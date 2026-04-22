"""
Interview Monitoring Views - Real-time violation tracking and performance analysis
"""
import logging
from datetime import datetime
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from interviews.models import Interview, Violation
from core.openai_client import _call, _strip_json
import json

logger = logging.getLogger('innovaite')


class ViolationTrackingView(APIView):
    """
    POST /api/interviews/<interview_id>/violations/
    Track and store violations detected during interview
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, interview_id):
        try:
            interview = Interview.objects.get(id=interview_id)
        except Interview.DoesNotExist:
            return Response({'error': 'Interview not found'}, status=404)

        # Verify access
        user = request.user
        if str(user.id) not in [interview.recruiter_id, interview.candidate_id] and user.role != 'admin':
            return Response({'error': 'Access denied'}, status=403)

        data = request.data
        violation_type = data.get('type', 'UNKNOWN')
        description = data.get('description', 'Violation detected')
        severity = data.get('severity', 'MEDIUM')
        confidence = data.get('confidence', 0)
        details = data.get('details', {})

        # Create violation object
        violation = Violation(
            type=violation_type,
            description=description,
            timestamp=datetime.utcnow(),
            severity=severity,
            confidence=confidence,
            details=details
        )

        # Add to interview
        interview.violations.append(violation)
        
        # Update behavior score based on severity
        severity_penalties = {
            'LOW': 2,
            'MEDIUM': 5,
            'HIGH': 10,
            'CRITICAL': 20
        }
        penalty = severity_penalties.get(severity, 5)
        interview.behavior_score = max(0, interview.behavior_score - penalty)
        
        # Update monitoring stats
        if not interview.monitoring_stats:
            interview.monitoring_stats = {}
        
        violation_counts = interview.monitoring_stats.get('violation_counts', {})
        violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1
        interview.monitoring_stats['violation_counts'] = violation_counts
        interview.monitoring_stats['total_violations'] = len(interview.violations)
        interview.monitoring_stats['last_violation_time'] = datetime.utcnow().isoformat()
        
        interview.updated_at = datetime.utcnow()
        interview.save()

        logger.info(f"Violation recorded: {violation_type} for interview {interview_id}")

        return Response({
            'success': True,
            'violation': {
                'type': violation.type,
                'description': violation.description,
                'timestamp': violation.timestamp.isoformat(),
                'severity': violation.severity,
                'confidence': violation.confidence
            },
            'behavior_score': interview.behavior_score,
            'total_violations': len(interview.violations)
        })

    def get(self, request, interview_id):
        """Get all violations for an interview"""
        try:
            interview = Interview.objects.get(id=interview_id)
        except Interview.DoesNotExist:
            return Response({'error': 'Interview not found'}, status=404)

        # Verify access
        user = request.user
        if str(user.id) not in [interview.recruiter_id, interview.candidate_id] and user.role != 'admin':
            return Response({'error': 'Access denied'}, status=403)

        violations = [
            {
                'type': v.type,
                'description': v.description,
                'timestamp': v.timestamp.isoformat(),
                'severity': v.severity,
                'confidence': v.confidence,
                'details': v.details
            }
            for v in interview.violations
        ]

        return Response({
            'violations': violations,
            'total_violations': len(violations),
            'behavior_score': interview.behavior_score,
            'monitoring_stats': interview.monitoring_stats
        })


class PerformanceAnalysisView(APIView):
    """
    POST /api/interviews/<interview_id>/analyze-performance/
    Analyze candidate responses and generate performance report
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, interview_id):
        try:
            interview = Interview.objects.get(id=interview_id)
        except Interview.DoesNotExist:
            return Response({'error': 'Interview not found'}, status=404)

        # Only recruiter/admin can analyze
        user = request.user
        if user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Only recruiters can analyze performance'}, status=403)

        # Get transcript and responses
        transcript = interview.full_transcript or request.data.get('transcript', '')
        questions = interview.questions
        
        if not transcript or not questions:
            return Response({'error': 'No transcript or questions available'}, status=400)

        try:
            # Prepare analysis prompt
            questions_text = "\n".join([
                f"{i+1}. {q.text} (Expected: {', '.join(q.expected_keywords) if q.expected_keywords else 'N/A'})"
                for i, q in enumerate(questions)
            ])
            
            prompt = f"""Analyze this interview transcript and provide a detailed performance report.

QUESTIONS ASKED:
{questions_text}

CANDIDATE TRANSCRIPT:
{transcript[:4000]}

Provide analysis in the following format:
1. CORRECT ANSWERS: List questions answered correctly with brief explanation
2. INCORRECT/IRRELEVANT ANSWERS: List questions with poor responses
3. OVERALL PERFORMANCE SCORE: 0-100
4. STRENGTHS: Key strengths demonstrated
5. WEAKNESSES: Areas needing improvement
6. RECOMMENDATION: Hire/Consider/Reject with reasoning

Be specific and reference actual responses from the transcript."""

            # Use _call function from openai_client
            analysis = _call(prompt, user_id=str(user.id))

            # Parse and structure the analysis
            performance_metrics = {
                'analysis_text': analysis,
                'analyzed_at': datetime.utcnow().isoformat(),
                'transcript_length': len(transcript),
                'questions_count': len(questions),
                'violations_count': len(interview.violations),
                'behavior_score': interview.behavior_score
            }

            # Update interview with performance metrics
            interview.performance_metrics = performance_metrics
            interview.updated_at = datetime.utcnow()
            interview.save()

            logger.info(f"Performance analysis completed for interview {interview_id}")

            return Response({
                'success': True,
                'analysis': analysis,
                'performance_metrics': performance_metrics,
                'behavior_score': interview.behavior_score,
                'violations_summary': {
                    'total': len(interview.violations),
                    'by_severity': self._count_by_severity(interview.violations)
                }
            })

        except Exception as e:
            logger.error(f"Performance analysis failed: {str(e)}")
            return Response({'error': f'Analysis failed: {str(e)}'}, status=500)

    def _count_by_severity(self, violations):
        """Count violations by severity level"""
        counts = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
        for v in violations:
            counts[v.severity] = counts.get(v.severity, 0) + 1
        return counts


class ComprehensiveReportView(APIView):
    """
    GET /api/interviews/<interview_id>/comprehensive-report/
    Generate complete interview report with violations, performance, and behavior
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, interview_id):
        try:
            interview = Interview.objects.get(id=interview_id)
        except Interview.DoesNotExist:
            return Response({'error': 'Interview not found'}, status=404)

        # Verify access
        user = request.user
        if str(user.id) not in [interview.recruiter_id, interview.candidate_id] and user.role != 'admin':
            return Response({'error': 'Access denied'}, status=403)

        # Compile comprehensive report
        report = {
            'interview_id': str(interview.id),
            'title': interview.title,
            'candidate_id': interview.candidate_id,
            'scheduled_at': interview.scheduled_at.isoformat(),
            'status': interview.status,
            'duration_minutes': interview.duration_minutes,
            
            # Questions and Responses
            'questions': [
                {
                    'text': q.text,
                    'category': q.category,
                    'difficulty': q.difficulty
                }
                for q in interview.questions
            ],
            'total_questions': len(interview.questions),
            
            # Performance Metrics
            'performance_metrics': interview.performance_metrics,
            
            # Behavior Analysis
            'behavior_score': interview.behavior_score,
            'tab_switch_count': interview.tab_switch_count,
            
            # Violations
            'violations': [
                {
                    'type': v.type,
                    'description': v.description,
                    'timestamp': v.timestamp.isoformat(),
                    'severity': v.severity,
                    'confidence': v.confidence
                }
                for v in interview.violations
            ],
            'violations_summary': {
                'total': len(interview.violations),
                'by_type': self._count_by_type(interview.violations),
                'by_severity': self._count_by_severity(interview.violations)
            },
            
            # Monitoring Stats
            'monitoring_stats': interview.monitoring_stats,
            
            # Overall Assessment
            'overall_assessment': self._generate_assessment(interview),
            
            'generated_at': datetime.utcnow().isoformat()
        }

        return Response(report)

    def _count_by_type(self, violations):
        """Count violations by type"""
        counts = {}
        for v in violations:
            counts[v.type] = counts.get(v.type, 0) + 1
        return counts

    def _count_by_severity(self, violations):
        """Count violations by severity"""
        counts = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
        for v in violations:
            counts[v.severity] = counts.get(v.severity, 0) + 1
        return counts

    def _generate_assessment(self, interview):
        """Generate overall assessment based on all metrics"""
        behavior_score = interview.behavior_score
        violations_count = len(interview.violations)
        critical_violations = sum(1 for v in interview.violations if v.severity == 'CRITICAL')
        
        # Determine recommendation
        if critical_violations > 0:
            recommendation = 'REJECT'
            reason = f'{critical_violations} critical violation(s) detected'
        elif behavior_score < 50:
            recommendation = 'REJECT'
            reason = f'Poor behavior score: {behavior_score}/100'
        elif behavior_score < 70:
            recommendation = 'REVIEW'
            reason = f'Moderate behavior score: {behavior_score}/100'
        elif violations_count > 5:
            recommendation = 'REVIEW'
            reason = f'{violations_count} violations detected'
        else:
            recommendation = 'PROCEED'
            reason = f'Good behavior score: {behavior_score}/100'
        
        return {
            'recommendation': recommendation,
            'reason': reason,
            'behavior_score': behavior_score,
            'integrity_status': 'COMPROMISED' if critical_violations > 0 else 'ACCEPTABLE',
            'requires_hr_review': critical_violations > 0 or violations_count > 3
        }
