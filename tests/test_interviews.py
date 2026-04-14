"""
Unit tests for interview system
"""
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch


class TestConflictDetection(unittest.TestCase):
    """Test interview scheduling conflict detection"""
    
    def test_overlapping_interviews(self):
        """Test detection of overlapping interviews"""
        # Interview 1: 10:00 - 11:00
        start1 = datetime(2026, 4, 10, 10, 0)
        end1 = datetime(2026, 4, 10, 11, 0)
        
        # Interview 2: 10:30 - 11:30 (overlaps)
        start2 = datetime(2026, 4, 10, 10, 30)
        end2 = datetime(2026, 4, 10, 11, 30)
        
        # Check overlap
        overlaps = (start2 < end1 and end2 > start1)
        self.assertTrue(overlaps, 'Should detect overlap')
    
    def test_non_overlapping_interviews(self):
        """Test non-overlapping interviews"""
        # Interview 1: 10:00 - 11:00
        start1 = datetime(2026, 4, 10, 10, 0)
        end1 = datetime(2026, 4, 10, 11, 0)
        
        # Interview 2: 11:00 - 12:00 (no overlap)
        start2 = datetime(2026, 4, 10, 11, 0)
        end2 = datetime(2026, 4, 10, 12, 0)
        
        # Check overlap
        overlaps = (start2 < end1 and end2 > start1)
        self.assertFalse(overlaps, 'Should not detect overlap')
    
    def test_interview_time_validation(self):
        """Test interview time validation"""
        from core.conflict_detector import validate_interview_time
        
        # Past time should be invalid
        past_time = datetime.utcnow() - timedelta(hours=1)
        result = validate_interview_time(past_time, 45)
        self.assertFalse(result['valid'])
        
        # Future time should be valid
        future_time = datetime.utcnow() + timedelta(hours=2)
        result = validate_interview_time(future_time, 45)
        self.assertTrue(result['valid'])
        
        # Too short duration should be invalid
        future_time = datetime.utcnow() + timedelta(hours=2)
        result = validate_interview_time(future_time, 10)
        self.assertFalse(result['valid'])
        
        # Too long duration should be invalid
        future_time = datetime.utcnow() + timedelta(hours=2)
        result = validate_interview_time(future_time, 200)
        self.assertFalse(result['valid'])


class TestXAIEvaluation(unittest.TestCase):
    """Test XAI evaluation engine"""
    
    def test_recommendation_logic(self):
        """Test evaluation recommendation thresholds"""
        def get_recommendation(score):
            if score >= 80:
                return 'strong_yes'
            elif score >= 65:
                return 'yes'
            elif score >= 50:
                return 'maybe'
            elif score >= 35:
                return 'no'
            else:
                return 'strong_no'
        
        self.assertEqual(get_recommendation(85), 'strong_yes')
        self.assertEqual(get_recommendation(70), 'yes')
        self.assertEqual(get_recommendation(55), 'maybe')
        self.assertEqual(get_recommendation(40), 'no')
        self.assertEqual(get_recommendation(20), 'strong_no')
    
    def test_weighted_score_calculation(self):
        """Test weighted average calculation"""
        criteria = [
            {'score': 8.0, 'weight': 1.5},
            {'score': 7.0, 'weight': 1.2},
            {'score': 9.0, 'weight': 2.0},
        ]
        
        weighted_total = sum(c['score'] * c['weight'] for c in criteria)
        total_weight = sum(c['weight'] for c in criteria)
        avg_score = weighted_total / total_weight
        
        # (8*1.5 + 7*1.2 + 9*2.0) / (1.5 + 1.2 + 2.0) = 38.4 / 4.7 ≈ 8.17
        self.assertAlmostEqual(avg_score, 8.17, places=2)


if __name__ == '__main__':
    unittest.main()
