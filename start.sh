#!/bin/bash
# InnovAIte Backend Startup Script
# Automatically binds to the Railway assigned port

echo "Starting Backend with Port: $PORT"

# Run migrations (optional, since using MongoDB)
# python manage.py migrate --noinput

# Start Daphne
exec daphne -b 0.0.0.0 -p "$PORT" core.asgi:application
