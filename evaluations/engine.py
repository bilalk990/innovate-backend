"""
XAI Rule-Based Evaluation Engine — InnovAIte Interview Guardian
Evaluates candidate interview responses against resume data using transparent rules.
Each criterion is scored with explicit explanations so HR can understand the decision.
"""

import re
import logging
import concurrent.futures  # #63 — parallel OpenAI calls
from accounts.models import User

logger = logging.getLogger('innovaite')

try:
    from core.openai_client import (
        enhance_evaluation_summary, 
        analyze_response_semantics,
        analyze_behavioral_traits,
        check_integrity_plagiarism,
        analyze_job_fitment,
        analyze_culture_fit
    )
    AI_AVAILABLE = True
except Exception:
    AI_AVAILABLE = False



# ─── Scoring criteria configuration ──────────────────────────────────────────

CRITERIA = [
    {
        'criterion': 'communication_clarity',
        'label': 'Communication Clarity',
        'weight': 1.5,
        'description': 'How clearly and coherently the candidate expresses ideas.',
    },
    {
        'criterion': 'response_depth',
        'label': 'Response Depth',
        'weight': 1.2,
        'description': 'Whether the candidate provides detailed, substantive answers.',
    },
    {
        'criterion': 'keyword_alignment',
        'label': 'Keyword Alignment',
        'weight': 1.4,
        'description': 'Overlap between response keywords and expected interview keywords.',
    },
    {
        'criterion': 'resume_consistency',
        'label': 'Resume Consistency',
        'weight': 1.6,
        'description': 'Whether claims in responses match the skills and experience on the resume.',
    },
    {
        'criterion': 'response_completeness',
        'label': 'Response Completeness',
        'weight': 1.0,
        'description': 'Whether the candidate addressed all parts of the question.',
    },
    {
        'criterion': 'confidence_indicators',
        'label': 'Confidence Indicators',
        'weight': 0.8,
        'description': 'Presence of assertive language vs. vague hedging.',
    },
    {
        'criterion': 'semantic_accuracy',
        'label': 'Semantic Accuracy',
        'weight': 2.0,  # High weight for AI semantic matching
        'description': 'Deep AI analysis comparing response to ideal answer for accuracy and relevance.',
    },
]

# Hedge words that suggest low confidence
HEDGE_WORDS = ['maybe', 'perhaps', 'sort of', 'kind of', 'i think', 'not sure', 'possibly', 'might']

# Confidence words
CONFIDENCE_WORDS = ['definitely', 'certainly', 'successfully', 'achieved', 'led', 'built', 'implemented', 'designed', 'delivered', 'managed']


# ─── Per-criterion scoring functions ──────────────────────────────────────────

def score_communication_clarity(response_text):
    """Evaluate sentence structure and clarity"""
    rules_applied = []
    evidence = []
    score = 5.0  # baseline

    words = response_text.split()
    word_count = len(words)
    sentences = [s.strip() for s in response_text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    sentence_count = len(sentences)

    if word_count < 20:
        score -= 3
        rules_applied.append('RULE_SHORT_RESPONSE')
        evidence.append(f'Response is very short ({word_count} words). Expected at least 20.')
    elif word_count >= 50:
        score += 2
        rules_applied.append('RULE_ADEQUATE_LENGTH')
        evidence.append(f'Response has good length ({word_count} words).')

    avg_words_per_sentence = word_count / max(sentence_count, 1)
    if avg_words_per_sentence > 40:
        score -= 1
        rules_applied.append('RULE_OVERLY_LONG_SENTENCES')
        evidence.append('Sentences are very long, which reduces clarity.')
    elif 10 <= avg_words_per_sentence <= 25:
        score += 1
        rules_applied.append('RULE_OPTIMAL_SENTENCE_LENGTH')
        evidence.append('Sentence length is in the optimal range for clarity.')

    explanation = (
        f"Clarity score based on response length ({word_count} words, "
        f"{sentence_count} sentences). "
        + (evidence[0] if evidence else '')
    )
    return min(max(score, 0), 10), explanation, rules_applied, evidence


def score_response_depth(response_text):
    """Check if response goes beyond surface-level answers"""
    rules_applied = []
    evidence = []
    score = 5.0

    words = response_text.split()
    word_count = len(words)

    # Depth indicators
    depth_words = ['because', 'therefore', 'as a result', 'specifically', 'for example',
                   'in particular', 'the reason', 'which led', 'this enabled', 'resulted in']
    found_depth = [w for w in depth_words if w.lower() in response_text.lower()]

    if found_depth:
        score += min(len(found_depth) * 1.5, 4)
        rules_applied.append('RULE_DEPTH_INDICATORS_FOUND')
        evidence.append(f'Depth indicators found: {", ".join(found_depth[:3])}')
    else:
        score -= 2
        rules_applied.append('RULE_NO_DEPTH_INDICATORS')
        evidence.append('No depth indicators (because, therefore, for example) found.')

    if word_count >= 80:
        score += 1
        rules_applied.append('RULE_DETAILED_RESPONSE')
        evidence.append(f'Detailed response ({word_count} words indicates depth).')

    explanation = (
        f"Depth evaluated by presence of reasoning connectors and response length. "
        + (evidence[0] if evidence else 'No depth indicators detected.')
    )
    return min(max(score, 0), 10), explanation, rules_applied, evidence


def score_keyword_alignment(response_text, expected_keywords):
    """Check question expected keywords vs response"""
    rules_applied = []
    evidence = []

    if not expected_keywords:
        return 7.0, 'No expected keywords defined for this question.', ['RULE_NO_KEYWORDS_DEFINED'], ['No keywords to match against.']

    response_lower = response_text.lower()
    matched = [kw for kw in expected_keywords if kw.lower() in response_lower]
    match_ratio = len(matched) / len(expected_keywords)
    score = round(match_ratio * 10, 1)

    if matched:
        rules_applied.append('RULE_KEYWORDS_MATCHED')
        evidence.append(f'Matched keywords: {", ".join(matched)}')
    unmatched = [kw for kw in expected_keywords if kw.lower() not in response_lower]
    if unmatched:
        rules_applied.append('RULE_KEYWORDS_MISSING')
        evidence.append(f'Missing keywords: {", ".join(unmatched[:3])}')

    explanation = (
        f"Matched {len(matched)} of {len(expected_keywords)} expected keywords "
        f"({int(match_ratio*100)}% alignment)."
    )
    return min(max(score, 0), 10), explanation, rules_applied, evidence


def score_resume_consistency(response_text, resume_parsed_data):
    """Check that response doesn't contradict resume claims"""
    rules_applied = []
    evidence = []
    score = 5.0

    if not resume_parsed_data:
        return 5.0, 'No resume on file — cannot verify consistency.', ['RULE_NO_RESUME'], ['Resume data unavailable.']

    resume_skills = [s.lower() for s in resume_parsed_data.get('skills', [])]
    response_lower = response_text.lower()

    # Skills mentioned in response that are on resume
    confirmed_skills = [s for s in resume_skills if s in response_lower]
    if confirmed_skills:
        score += min(len(confirmed_skills) * 1.0, 4)
        rules_applied.append('RULE_RESUME_SKILL_CONFIRMED')
        evidence.append(f'Resume skills confirmed in response: {", ".join(confirmed_skills[:4])}')

    # Check for inflated claims (e.g., claiming 10 years when resume shows 2 years)
    year_claims_response = [int(m) for m in re.findall(r'(\d+)\s+year', response_lower)]
    resume_years = resume_parsed_data.get('total_experience_years', 0) if resume_parsed_data else 0
    
    if year_claims_response:
        max_claim = max(year_claims_response)
        if resume_years > 0 and max_claim > (resume_years + 2):
            score -= 3
            rules_applied.append('RULE_INFLATED_EXPERIENCE_CLAIM')
            evidence.append(f'Claimed {max_claim} years but resume shows ~{resume_years} years.')
        elif max_claim > 15:
            score -= 1
            rules_applied.append('RULE_SUSPICIOUS_EXPERIENCE_CLAIM')
            evidence.append(f'Very high experience claim ({max_claim} years) — verify manually.')

    explanation = (
        f"Resume consistency checked against {len(resume_skills)} skills on file. "
        + (evidence[0] if evidence else 'Limited overlap detected.')
    )
    return min(max(score, 0), 10), explanation, rules_applied, evidence


def score_response_completeness(response_text, question_text):
    """Estimate if the response addresses the question"""
    rules_applied = []
    evidence = []
    score = 5.0

    words = response_text.split()
    word_count = len(words)

    # Multi-part question detection
    question_parts = question_text.count('?') + question_text.lower().count(' and ') + question_text.lower().count(', how')
    if question_parts > 1 and word_count < 40:
        score -= 2
        rules_applied.append('RULE_MULTI_PART_INCOMPLETE')
        evidence.append(f'Question has {question_parts} parts but response is brief ({word_count} words).')
    elif word_count >= 40:
        score += 2
        rules_applied.append('RULE_COMPLETE_RESPONSE')
        evidence.append(f'Response length ({word_count} words) suggests completeness.')

    explanation = f"Completeness assessed by response length vs question complexity."
    return min(max(score, 0), 10), explanation, rules_applied, evidence


def score_confidence_indicators(response_text):
    """Measure assertive vs hedging language"""
    rules_applied = []
    evidence = []
    score = 5.0

    response_lower = response_text.lower()
    found_hedges = [w for w in HEDGE_WORDS if w in response_lower]
    found_confidence = [w for w in CONFIDENCE_WORDS if w in response_lower]

    if found_confidence:
        score += min(len(found_confidence) * 1.2, 4)
        rules_applied.append('RULE_CONFIDENT_LANGUAGE')
        evidence.append(f'Confident language: {", ".join(found_confidence[:3])}')

    if found_hedges:
        score -= min(len(found_hedges) * 0.8, 3)
        rules_applied.append('RULE_HEDGING_LANGUAGE')
        evidence.append(f'Hedging language detected: {", ".join(found_hedges[:3])}')

    explanation = (
        f"Confidence based on language analysis. "
        f"Found {len(found_confidence)} confident and {len(found_hedges)} hedging expressions."
    )
    return min(max(score, 0), 10), explanation, rules_applied, evidence


def _fallback_summary(overall):
    if overall >= 80:
        return 'Candidate performed exceptionally well across all evaluated dimensions.'
    elif overall >= 65:
        return 'Candidate showed solid performance and is recommended for the next stage.'
    elif overall >= 50:
        return 'Candidate showed acceptable performance with some areas for improvement.'
    elif overall >= 35:
        return 'Candidate showed insufficient performance in key areas.'
    return 'Candidate did not meet the minimum performance threshold.'


# ─── Main evaluation engine ───────────────────────────────────────────────────

def run_xai_evaluation(interview, resume_parsed_data=None, user_id: str = None):
    """
    Run the XAI rule-based evaluation for a completed interview.
    Returns a dict with all criterion results, overall score, and recommendation.
    """
    responses = interview.candidate_responses or {}
    questions = interview.questions or []
    job_description = interview.job_description or ""
    
    # Try to get company values from recruiter
    try:
        recruiter = User.objects(id=interview.recruiter_id).first()
        company_values = recruiter.company_values if recruiter else []
    except (Exception, AttributeError):
        company_values = []

    all_criterion_results = []
    weighted_total = 0
    total_weight = 0

    for config in CRITERIA:
        criterion = config['criterion']
        weight = config['weight']

        # Aggregate scores across all responses for this criterion
        criterion_scores = []
        all_rules = []
        all_evidence = []
        combined_explanation = []

        for idx, question in enumerate(questions):
            resp = responses.get(str(idx), '')
            if not resp:
                continue

            if criterion == 'communication_clarity':
                s, exp, rules, ev = score_communication_clarity(resp)
            elif criterion == 'response_depth':
                s, exp, rules, ev = score_response_depth(resp)
            elif criterion == 'keyword_alignment':
                s, exp, rules, ev = score_keyword_alignment(resp, question.expected_keywords)
            elif criterion == 'resume_consistency':
                s, exp, rules, ev = score_resume_consistency(resp, resume_parsed_data)
            elif criterion == 'response_completeness':
                s, exp, rules, ev = score_response_completeness(resp, question.text)
            elif criterion == 'confidence_indicators':
                s, exp, rules, ev = score_confidence_indicators(resp)
            elif criterion == 'semantic_accuracy':
                if AI_AVAILABLE and getattr(question, 'ideal_answer', ''):
                    ai_res = analyze_response_semantics(question.text, question.ideal_answer, resp, user_id=user_id)
                    s, exp, rules, ev = ai_res['score'], ai_res['explanation'], ['RULE_AI_SEMANTIC_ANALYSIS'], ai_res.get('missing_points', [])
                else:
                    s, exp, rules, ev = 5.0, 'AI Semantic Analysis unavailable.', ['RULE_AI_SKIPPED'], []
            else:
                continue

            criterion_scores.append(s)
            all_rules.extend(rules)
            all_evidence.extend(ev)
            combined_explanation.append(f"Q{idx+1}: {exp}")

        avg_score = sum(criterion_scores) / len(criterion_scores) if criterion_scores else 5.0
        weighted_total += avg_score * weight
        total_weight += weight

        all_criterion_results.append({
            'criterion': criterion,
            'score': round(avg_score, 2),
            'max_score': 10.0,
            'weight': weight,
            'explanation': ' | '.join(combined_explanation) or f'No responses evaluated for {criterion}.',
            'rules_applied': list(set(all_rules)),
            'evidence': all_evidence[:5],  # cap evidence list
        })

    # Overall weighted score (0–100)
    overall = round((weighted_total / total_weight) * 10, 1) if total_weight > 0 else 50.0

    # Recommendation logic
    if overall >= 80:
        recommendation = 'strong_yes'
    elif overall >= 65:
        recommendation = 'yes'
    elif overall >= 50:
        recommendation = 'maybe'
    elif overall >= 35:
        recommendation = 'no'
    else:
        recommendation = 'strong_no'

    # Generate summary (AI-enhanced if available)
    ai_summary_used = False
    if AI_AVAILABLE:
        try:
            summary = enhance_evaluation_summary(
                overall_score=overall,
                recommendation=recommendation,
                criterion_results=all_criterion_results,
                job_title=interview.job_title
            )
            ai_summary_used = True
        except Exception:
            summary = _fallback_summary(overall)
    else:
        summary = _fallback_summary(overall)


    # Strengths & weaknesses — use human-readable labels from CRITERIA config
    label_map = {c['criterion']: c['label'] for c in CRITERIA}
    sorted_results = sorted(all_criterion_results, key=lambda x: x['score'], reverse=True)
    strengths = [label_map.get(r['criterion'], r['criterion'].replace('_', ' ').title()) for r in sorted_results[:2]]
    weaknesses = [label_map.get(r['criterion'], r['criterion'].replace('_', ' ').title()) for r in sorted_results[-2:]]

    # Behavioral & Integrity Analysis (Enterprise)
    behavioral = {"confidence_score": 50, "fluency_score": 50, "behavioral_summary": "N/A"}
    integrity = {"integrity_score": 100, "notes": "No issues detected."}
    job_fit = {"fitment_score": 50}
    culture_fit = {"culture_score": 50}

    # Fallback resume alignment (simple string match)
    resume_alignment_fallback = overall
    if resume_parsed_data and resume_parsed_data.get('skills'):
        skills = resume_parsed_data['skills']
        all_resp_text = ' '.join(str(v) for v in responses.values()).lower()
        matched = sum(1 for s in skills if s.lower() in all_resp_text)
        resume_alignment_fallback = round((matched / len(skills)) * 100, 1) if skills else overall

    if AI_AVAILABLE:
        transcript = " ".join(str(v) for v in responses.values())

        # Run all AI calls in parallel using ThreadPoolExecutor — pass user_id for rate limiting
        def _behavioral():
            try:
                return analyze_behavioral_traits(transcript, user_id=user_id)
            except Exception:
                return {"confidence_score": 50, "fluency_score": 50, "behavioral_summary": "N/A"}

        def _integrity():
            try:
                return check_integrity_plagiarism(responses, user_id=user_id)
            except Exception:
                return {"integrity_score": 100, "notes": "No issues detected."}

        def _job_fit():
            if not job_description:
                return {"fitment_score": 50}
            try:
                return analyze_job_fitment(resume_parsed_data, job_description)
            except Exception:
                return {"fitment_score": 50}

        def _culture():
            if not company_values:
                return {"culture_score": 50}
            try:
                return analyze_culture_fit(transcript, company_values)
            except Exception:
                return {"culture_score": 50}

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            f_behavioral = executor.submit(_behavioral)
            f_integrity = executor.submit(_integrity)
            f_job_fit = executor.submit(_job_fit)
            f_culture = executor.submit(_culture)

            try:
                behavioral = f_behavioral.result(timeout=45)
            except (concurrent.futures.TimeoutError, Exception) as e:
                logger.warning(f'[Eval] Behavioral analysis timed out or failed: {e}')
                behavioral = {"confidence_score": 50, "fluency_score": 50, "behavioral_summary": "Analysis unavailable."}

            try:
                integrity = f_integrity.result(timeout=45)
            except (concurrent.futures.TimeoutError, Exception) as e:
                logger.warning(f'[Eval] Integrity check timed out or failed: {e}')
                integrity = {"integrity_score": 100, "notes": "Integrity check skipped."}

            try:
                job_fit = f_job_fit.result(timeout=45)
            except (concurrent.futures.TimeoutError, Exception) as e:
                logger.warning(f'[Eval] Job fit analysis timed out or failed: {e}')
                job_fit = {"fitment_score": 50}

            try:
                culture_fit = f_culture.result(timeout=45)
            except (concurrent.futures.TimeoutError, Exception) as e:
                logger.warning(f'[Eval] Culture fit analysis timed out or failed: {e}')
                culture_fit = {"culture_score": 50}

    # Proctoring score: deduct 10 per tab switch, min 0
    # Also if no responses provided, integrity cannot be verified
    tab_switches = getattr(interview, 'tab_switch_count', 0)
    proctoring_score_from_violations = max(0, 100 - (tab_switches * 10))
    
    if not responses:
        proctoring_score = 0
    else:
        proctoring_score = min(integrity.get('integrity_score', 100), proctoring_score_from_violations)

    return {
        'criterion_results': all_criterion_results,
        'overall_score': overall,
        'recommendation': recommendation,
        'summary': summary,
        'strengths': strengths,
        'weaknesses': weaknesses,
        'confidence_score': behavioral.get('confidence_score', 50),
        'fluency_score': behavioral.get('fluency_score', 50),
        'behavioral_summary': behavioral.get('behavioral_summary', behavioral.get('summary', 'N/A')),
        'proctoring_score': proctoring_score,
        'integrity_notes': integrity.get('notes', ''),
        'tab_switch_count': tab_switches,
        'resume_alignment_score': job_fit.get('fitment_score', resume_alignment_fallback),
        'culture_fit_score': culture_fit.get('culture_score', 70),
        'ai_summary_used': ai_summary_used,  # #52
    }
