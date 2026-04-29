"""
WebRTC Signaling Consumer via Django Channels
Handles: offer, answer, ICE exchange + Waiting Room admit flow
Token-validated WebSocket connections for security.
"""
import json
import logging
from datetime import datetime
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async

logger = logging.getLogger('innovaite')


@database_sync_to_async
def get_interview_by_room(room_id):
    """Fetch interview document for the given room_id."""
    try:
        from interviews.models import Interview
        return Interview.objects.get(room_id=room_id)
    except Exception:
        return None


@database_sync_to_async
def validate_user_room_access(interview, user_id: str, user_role: str) -> bool:
    """Return True only if this user is allowed to join the room."""
    if user_role == 'admin':
        return True
    allowed_ids = [interview.recruiter_id, interview.candidate_id]
    return user_id in [i for i in allowed_ids if i]


@database_sync_to_async
def set_interview_status(room_id: str, status: str):
    """Update interview status in DB."""
    try:
        from interviews.models import Interview
        interview = Interview.objects.get(room_id=room_id)
        if interview.status != status:
            interview.status = status
            interview.updated_at = datetime.utcnow()
            interview.save()
    except Exception:
        pass


@database_sync_to_async
def save_chat_message(room_id: str, message: dict):
    """Persist a chat message to MongoDB."""
    try:
        from interviews.models import Interview
        interview = Interview.objects.get(room_id=room_id)
        # Ensure timestamp is string for JSON serialization later
        message['timestamp'] = datetime.utcnow().isoformat()
        interview.update(push__chat_history=message)
    except Exception as e:
        logger.error(f'[WS] Persistent chat failed: {e}')


@database_sync_to_async
def get_chat_history(room_id: str):
    """Fetch all chat history for this interview."""
    try:
        from interviews.models import Interview
        interview = Interview.objects.get(room_id=room_id)
        return list(interview.chat_history or [])
    except Exception:
        return []


class SignalingConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        self.room_id = self.scope['url_route']['kwargs']['room_id']
        self.room_group_name = f'room_{self.room_id}'

        logger.info(f'[WS] Connection attempt for room: {self.room_id}')

        # Validate interview exists
        interview = await get_interview_by_room(self.room_id)
        if not interview:
            logger.warning(f'[WS] Room not found: {self.room_id}')
            await self.close(code=4004)
            return

        # Check token expiry
        if interview.token_expires_at and datetime.utcnow() > interview.token_expires_at:
            logger.warning(f'[WS] Expired token for room: {self.room_id}')
            await self.close(code=4010)
            return

        # Validate JWT and check user has permission
        user = self.scope.get('user')
        logger.info(f'[WS] User from scope: {user}, is_authenticated: {getattr(user, "is_authenticated", False)}')
        
        if not user or not getattr(user, 'is_authenticated', False):
            logger.warning(f'[WS] Unauthenticated connection attempt for room: {self.room_id}')
            await self.close(code=4001)
            return

        user_id = str(user.id)
        self.user_role = getattr(user, 'role', 'candidate')
        logger.info(f'[WS] User {user_id} (role: {self.user_role}) attempting to join room {self.room_id}')
        
        has_access = await validate_user_room_access(interview, user_id, self.user_role)
        if not has_access:
            logger.warning(f'[WS] Unauthorized room access: user={user_id} room={self.room_id}')
            await self.close(code=4003)
            return

        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()
        logger.info(f'[WS] Connection accepted for user {user_id} in room {self.room_id}')

        # --- PERSISTENT CHAT: Fetch and send history on join ---
        history = await get_chat_history(self.room_id)
        if history:
            await self.send(text_data=json.dumps({
                'type': 'chat_history',
                'history': history,
            }))
        # -----------------------------------------------------

        # ── Waiting Room Logic ──
        # Candidates enter "waiting" — they do NOT trigger peer_connected automatically.
        # They will send `request_admit` from the frontend.
        # Recruiters/admins trigger peer_connected normally.
        if self.user_role in ('recruiter', 'admin'):
            await self.channel_layer.group_send(
                self.room_group_name,
                {'type': 'peer_connected', 'channel': self.channel_name}
            )
            logger.info(f'[WS] Recruiter connected to room: {self.room_id}')
        else:
            # CRITICAL FIX: Candidate immediately notifies recruiter on connection
            # Don't wait for frontend to send request_admit
            await self.channel_layer.group_send(
                self.room_group_name,
                {'type': 'candidate_at_door', 'channel': self.channel_name, 'force_notify': True}
            )
            logger.info(f'[WS] Candidate entered waiting room: {self.room_id}, notification sent')

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)
        await self.channel_layer.group_send(
            self.room_group_name,
            {'type': 'peer_disconnected', 'channel': self.channel_name}
        )
        # If candidate disconnects while waiting, stop showing the "at door" notification
        if getattr(self, 'user_role', 'candidate') == 'candidate':
            await self.channel_layer.group_send(
                self.room_group_name,
                {'type': 'candidate_left_door', 'channel': self.channel_name}
            )
        # Mark interview as completed on abrupt disconnect (normal close = meeting already ended)
        if close_code not in (1000, 1001):
            await set_interview_status(self.room_id, 'completed')
        logger.info(f'[WS] Peer disconnected from room: {self.room_id} (code={close_code})')

    async def receive(self, text_data):
        """Forward signaling messages and handle waiting room flow."""
        try:
            data = json.loads(text_data)
        except json.JSONDecodeError:
            logger.warning(f'[WS] Invalid JSON from room: {self.room_id}')
            return

        msg_type = data.get('type')

        # ── Standard WebRTC signaling + real-time events ──
        if msg_type in ['offer', 'answer', 'ice-candidate', 'chat', 'ready', 'media-status-update', 'violation_alert']:
            # Save chat messages to DB for persistence
            if msg_type == 'chat':
                await save_chat_message(self.room_id, {
                    'sender_id': data.get('sender_id'),
                    'text': data.get('text'),
                    'from': data.get('from', 'peer'),
                })

            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'signaling_message',
                    'message': data,
                    'sender': self.channel_name,
                }
            )

        # ── Waiting Room: Candidate re-announces they are waiting ──
        elif msg_type == 'request_admit':
            # CRITICAL FIX: Add extensive logging and force notification
            logger.info(f'[WS] Candidate {self.channel_name} requesting admission in room {self.room_id}')
            # Relay to group so recruiter (who may have just joined) sees the request
            # CRITICAL: This should ALWAYS notify, even if notified before
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'candidate_at_door',
                    'channel': self.channel_name,
                    'force_notify': True,  # Force notification even if already notified
                }
            )
            logger.info(f'[WS] Sent candidate_at_door notification to group {self.room_group_name}')

        # ── Waiting Room: Recruiter joins and checks for waiting candidates ──
        elif msg_type == 'recruiter_joined':
            # When recruiter joins, trigger candidate to re-send request_admit
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'signaling_message',
                    'message': {'type': 'recruiter_ready'},
                    'sender': self.channel_name,
                }
            )

        # ── Waiting Room: Recruiter admits candidate ──
        elif msg_type == 'admit_candidate':
            # Reset notification flag for ALL recruiters in the room
            await self.channel_layer.group_send(
                self.room_group_name,
                {'type': 'reset_candidate_notification'}
            )
            # Tell candidate they are admitted (exclude recruiter via sender=self.channel_name)
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'signaling_message',
                    'message': {'type': 'admitted'},
                    'sender': self.channel_name,
                }
            )
            # Small delay to ensure 'admitted' is processed before 'peer-connected'
            import asyncio
            await asyncio.sleep(0.1)
            # Trigger peer_connected for BOTH users (candidate will create offer, recruiter will wait for it)
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'peer_connected',
                    'channel': None,  # Send to everyone so both know they're connected
                }
            )
            logger.info(f'[WS] Candidate admitted to room: {self.room_id}')

        # ── Waiting Room: Recruiter denies candidate ──
        elif msg_type == 'deny_candidate':
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'signaling_message',
                    'message': {'type': 'denied'},
                    'sender': self.channel_name,  # Exclude recruiter — only candidate gets it
                }
            )
            logger.info(f'[WS] Candidate denied entry to room: {self.room_id}')

        # ── End Meeting: Recruiter ends the meeting for everyone ──
        elif msg_type == 'end_meeting':
            # Update status in DB immediately
            await set_interview_status(self.room_id, 'completed')
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'signaling_message',
                    'message': {'type': 'end_meeting'},
                    'sender': None,  # Send to everyone including sender
                }
            )
            logger.info(f'[WS] Meeting ended by recruiter in room: {self.room_id}')
    # ── Event Handlers ──

    async def signaling_message(self, event):
        """Send message to WebSocket. If sender is None, send to ALL (broadcast)."""
        sender = event.get('sender')
        if sender is None or sender != self.channel_name:
            await self.send(text_data=json.dumps(event['message']))

    async def peer_connected(self, event):
        """Notify a peer that someone connected."""
        channel = event.get('channel')
        # If channel is None, send to everyone
        # Otherwise, only send to peers whose channel is different
        if channel is None or channel != self.channel_name:
            await self.send(text_data=json.dumps({'type': 'peer-connected'}))
            # Set interview to active when both peers are connected (only do this once)
            if channel is None:
                await set_interview_status(self.room_id, 'active')

    async def peer_disconnected(self, event):
        if event.get('channel') != self.channel_name:
            await self.send(text_data=json.dumps({'type': 'peer-disconnected'}))

    async def candidate_at_door(self, event):
        """Notify recruiter that a candidate is waiting in the lobby."""
        # Only send to recruiters/admins (not back to the candidate who sent it)
        if event.get('channel') != self.channel_name:
            if getattr(self, 'user_role', 'candidate') in ('recruiter', 'admin'):
                # CRITICAL FIX: Always send notification, remove the flag check
                # The flag was preventing notifications from being sent
                await self.send(text_data=json.dumps({'type': 'candidate_waiting'}))
                logger.info(f'[WS] Sent candidate_waiting notification to recruiter in room {self.room_id}')
                # Set flag after sending (for tracking only, not blocking)
                self._candidate_notified = True

    async def candidate_left_door(self, event):
        """Notify recruiter that a candidate is no longer waiting."""
        if event.get('channel') != self.channel_name:
            if getattr(self, 'user_role', 'candidate') in ('recruiter', 'admin'):
                await self.send(text_data=json.dumps({'type': 'candidate_left'}))
                if hasattr(self, '_candidate_notified'):
                    delattr(self, '_candidate_notified')

    async def reset_candidate_notification(self, event):
        """Reset the candidate notification flag for all recruiters."""
        if getattr(self, 'user_role', 'candidate') in ('recruiter', 'admin'):
            if hasattr(self, '_candidate_notified'):
                delattr(self, '_candidate_notified')
                logger.info(f'[WS] Reset candidate notification flag for recruiter in room {self.room_id}')
