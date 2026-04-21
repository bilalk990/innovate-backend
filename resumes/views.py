"""
Resumes app views — upload, parse, retrieve
"""
import os
import re
import logging
import mongoengine
from datetime import datetime
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser
from resumes.models import Resume

logger = logging.getLogger('innovaite')

try:
    from core.openai_client import parse_resume_with_ai
    AI_AVAILABLE = True
except Exception:
    AI_AVAILABLE = False


def extract_text_from_file(file_path: str, ext: str) -> str:
    """
    Extract raw text from uploaded file.
    Supports: .txt, .pdf (via pdfminer), .docx (basic), .doc
    """
    try:
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

        elif ext == '.pdf':
            try:
                from pdfminer.high_level import extract_text as pdf_extract
                return pdf_extract(file_path)
            except ImportError:
                # pdfminer not installed — read as bytes and decode
                with open(file_path, 'rb') as f:
                    return f.read().decode('utf-8', errors='ignore')

        elif ext in ['.docx', '.doc']:
            # Basic DOCX text extraction without python-docx
            import zipfile
            import xml.etree.ElementTree as ET
            try:
                with zipfile.ZipFile(file_path, 'r') as z:
                    with z.open('word/document.xml') as doc:
                        tree = ET.parse(doc)
                        texts = [node.text for node in tree.iter() if node.text]
                        return ' '.join(texts)
            except Exception:
                with open(file_path, 'rb') as f:
                    return f.read().decode('utf-8', errors='ignore')
    except Exception:
        return ''


def calculate_quality_score(parsed_data: dict) -> int:
    """
    Calculate a Resume Quality/Strength score (0-100) based on parsed data richness.
    """
    score = 0
    
    # 1. Contact info (20%)
    if parsed_data.get('name'): score += 5
    if parsed_data.get('email'): score += 5
    if parsed_data.get('phone'): score += 5
    if parsed_data.get('linkedin') or parsed_data.get('github'): score += 5
    
    # 2. Summary (10%)
    if parsed_data.get('summary') and len(parsed_data.get('summary', '').split()) > 10:
        score += 10
    
    # 3. Skills (20%)
    skills_count = len(parsed_data.get('skills', []))
    if skills_count >= 10: score += 20
    elif skills_count >= 5: score += 10
    elif skills_count > 0: score += 5
    
    # 4. Experience (30%)
    exp_count = len(parsed_data.get('experience', []))
    if exp_count >= 3: score += 30
    elif exp_count >= 1: score += 15
    
    # 5. Education (20%)
    edu_count = len(parsed_data.get('education', []))
    if edu_count >= 2: score += 20
    elif edu_count >= 1: score += 10
    
    return min(score, 100)


def parse_resume(file_path: str, ext: str) -> dict:
    """
    Parse resume: try OpenAI GPT first, fall back to rule-based.
    """
    raw_text = extract_text_from_file(file_path, ext)
    if not raw_text.strip():
        return {}

    # Attempt AI parsing (OpenAI GPT)
    if AI_AVAILABLE:
        try:
            # CRITICAL FIX: Pass user_id for rate limiting (use 'system' for background parsing)
            ai_result = parse_resume_with_ai(raw_text, user_id='system')
            if ai_result and (ai_result.get('skills') or ai_result.get('name')):
                ai_result['parsed_by'] = 'openai-gpt'
                return ai_result
        except Exception as e:
            logger.warning(f'[Resume] AI parsing failed: {str(e)} — using rule-based fallback')

    # Rule-based fallback
    parsed = simple_resume_parser_from_text(raw_text)
    parsed['quality_score'] = calculate_quality_score(parsed)
    return parsed


def simple_resume_parser_from_text(text: str) -> dict:
    """Rule-based fallback parser — extracting name, email, phone, skills, education, experience."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # --- Contact Info ---
    email_match = re.search(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}', text)
    email = email_match.group(0) if email_match else ''

    phone_match = re.search(r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}', text)
    phone = phone_match.group(0) if phone_match else ''

    linkedin_match = re.search(r'linkedin\.com/in/[\w-]+', text, re.IGNORECASE)
    linkedin = linkedin_match.group(0) if linkedin_match else ''

    github_match = re.search(r'github\.com/[\w-]+', text, re.IGNORECASE)
    github = github_match.group(0) if github_match else ''

    # --- Skills: expanded 120+ keyword list ---
    skill_keywords = [
        # Languages
        'Python', 'JavaScript', 'TypeScript', 'Java', 'C++', 'C#', 'C', 'Go', 'Rust',
        'PHP', 'Ruby', 'Swift', 'Kotlin', 'Dart', 'Scala', 'R', 'MATLAB', 'Bash',
        # Frontend
        'React', 'Vue.js', 'Angular', 'Next.js', 'Nuxt.js', 'Svelte', 'HTML', 'CSS',
        'SASS', 'SCSS', 'Tailwind CSS', 'Bootstrap', 'Material UI', 'Chakra UI',
        # Backend
        'Node.js', 'Django', 'Flask', 'FastAPI', 'Spring Boot', 'Laravel', 'Express.js',
        'NestJS', 'Ruby on Rails', 'ASP.NET', 'Gin', 'Fiber',
        # Mobile
        'Flutter', 'React Native', 'SwiftUI', 'Jetpack Compose', 'Android', 'iOS',
        # Databases
        'MongoDB', 'PostgreSQL', 'MySQL', 'SQLite', 'Redis', 'Elasticsearch',
        'Cassandra', 'DynamoDB', 'Firebase', 'Firestore', 'Oracle', 'MS SQL',
        # Cloud & DevOps
        'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Terraform', 'Ansible',
        'CI/CD', 'Jenkins', 'GitHub Actions', 'GitLab CI', 'Nginx', 'Linux',
        'Heroku', 'DigitalOcean', 'Cloudflare', 'Vercel', 'Netlify',
        # AI/ML
        'Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch', 'Keras',
        'Scikit-learn', 'NLP', 'Computer Vision', 'OpenCV', 'Hugging Face',
        'LangChain', 'OpenAI', 'GPT', 'LLM', 'RAG', 'Vector Database',
        # APIs & Tools
        'REST API', 'GraphQL', 'gRPC', 'WebSocket', 'OAuth', 'JWT', 'Swagger',
        'Postman', 'Git', 'GitHub', 'GitLab', 'Jira', 'Confluence', 'Notion',
        # Soft Skills
        'Agile', 'Scrum', 'Kanban', 'Leadership', 'Communication', 'Problem Solving',
        'Team Collaboration', 'Project Management', 'Critical Thinking',
    ]
    found_skills = [s for s in skill_keywords if s.lower() in text.lower()]

    # --- Summary ---
    summary = ''
    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in ['summary', 'objective', 'profile', 'about']):
            summary = ' '.join(lines[i+1:i+5]) if i + 1 < len(lines) else ''
            break
    if not summary and lines:
        # Use first substantial paragraph as summary
        for line in lines[1:6]:
            if len(line.split()) > 8:
                summary = line
                break

    # --- Education (rule-based heuristics) ---
    education = []
    degree_keywords = ['bachelor', 'master', 'phd', 'b.sc', 'm.sc', 'b.e', 'm.e', 'b.s',
                       'm.s', 'mba', 'bba', 'b.tech', 'm.tech', 'diploma', 'associate']
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(kw in line_lower for kw in degree_keywords):
            year_match = re.search(r'(19|20)\d{2}', line)
            edu_entry = {
                'degree': line[:80],
                'institution': lines[i+1][:80] if i + 1 < len(lines) else '',
                'year': year_match.group(0) if year_match else '',
            }
            education.append(edu_entry)
            if len(education) >= 3:
                break

    # --- Experience (rule-based heuristics) ---
    experience = []
    exp_section = False
    exp_buffer = []
    for line in lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in ['experience', 'employment', 'work history', 'positions']):
            exp_section = True
            continue
        if any(kw in line_lower for kw in ['education', 'skills', 'certifications', 'projects', 'awards']):
            exp_section = False
        if exp_section and line:
            exp_buffer.append(line)
    # Parse experience buffer into entries (heuristic: line with date = new role)
    current_exp = {}
    for line in exp_buffer[:40]:
        date_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{4})[\w\s,\-–]+?(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{4}|Present)', line, re.IGNORECASE)
        if date_match:
            if current_exp.get('title'):
                experience.append(current_exp)
            current_exp = {'title': '', 'company': '', 'duration': date_match.group(0), 'years': 0, 'description': ''}
        elif not current_exp.get('title') and len(line.split()) <= 8:
            current_exp['title'] = line
        elif not current_exp.get('company') and len(line.split()) <= 6:
            current_exp['company'] = line
        elif current_exp.get('title'):
            current_exp['description'] = (current_exp.get('description', '') + ' ' + line).strip()[:200]
        if len(experience) >= 5:
            break
    if current_exp.get('title'):
        experience.append(current_exp)

    # --- Certifications ---
    certifications = []
    cert_keywords = ['certified', 'certification', 'certificate', 'aws certified', 'google certified', 'microsoft certified', 'pmp', 'cpa']
    for line in lines:
        if any(kw in line.lower() for kw in cert_keywords) and len(line) < 120:
            certifications.append(line)
            if len(certifications) >= 5:
                break

    parsed_result = {
        'name': lines[0] if lines else '',
        'email': email,
        'phone': phone,
        'linkedin': linkedin,
        'github': github,
        'skills': found_skills,
        'summary': summary,
        'total_experience_years': len(experience),  # rough estimate
        'education': education,
        'experience': experience,
        'certifications': certifications,
        'languages': [],
        'parsed_by': 'rule-based',
        'raw_text': text[:3000],
    }
    parsed_result['quality_score'] = calculate_quality_score(parsed_result)
    return parsed_result



class ResumeUploadView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        if request.user.role != 'candidate':
            return Response({'error': 'Only candidates can upload resumes.'}, status=403)

        if 'resume' not in request.FILES:
            return Response({'error': 'Resume file is required.'}, status=400)

        file = request.FILES['resume']
        
        # Validate file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if file.size > max_size:
            return Response({'error': 'File size must be less than 10MB.'}, status=400)
        
        allowed_extensions = ['.pdf', '.txt', '.docx', '.doc']
        ext = os.path.splitext(file.name)[1].lower()
        if ext not in allowed_extensions:
            return Response({'error': f'Unsupported file type. Use: {allowed_extensions}'}, status=400)

        # Save file
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'resumes', str(request.user.id))
        os.makedirs(upload_dir, exist_ok=True)
        filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file.name}"
        file_path = os.path.join(upload_dir, filename)
        with open(file_path, 'wb+') as dest:
            for chunk in file.chunks():
                dest.write(chunk)

        # Mark previous resumes as inactive
        Resume.objects(candidate_id=str(request.user.id)).update(is_active=False)

        # Save file first, return immediately with pending status
        resume = Resume(
            candidate_id=str(request.user.id),
            file_path=file_path,
            original_filename=file.name,
            file_size=file.size,
            parse_status='pending',
            is_active=True,
        )
        resume.save()

        # #53 — Parse resume asynchronously in background thread (non-blocking)
        import threading
        import logging
        logger = logging.getLogger('innovaite')

        def _parse_in_background(resume_id, fp, extension):
            try:
                from resumes.models import Resume as R
                r = R.objects.get(id=resume_id)
                r.parse_status = 'processing'  # Set to processing
                r.save()
                
                logger.info(f'[Resume] Starting parse for {resume_id}')
                parsed = parse_resume(fp, extension)
                
                # Log what was parsed
                logger.info(f'[Resume] Parsed data: name={parsed.get("name")}, skills_count={len(parsed.get("skills", []))}, parsed_by={parsed.get("parsed_by")}')
                
                r.parsed_data = parsed
                # CRITICAL FIX: Set to 'parsed' even if minimal data, as long as we have SOMETHING
                has_data = (parsed.get('skills') or parsed.get('name') or parsed.get('email') or 
                           parsed.get('experience') or parsed.get('education'))
                r.parse_status = 'parsed' if has_data else 'failed'
                r.parsed_by_ai = parsed.get('parsed_by') == 'openai-gpt'
                r.save()
                logger.info(f'[Resume] Successfully parsed resume {resume_id}, status={r.parse_status}')
            except Exception as e:
                logger.error(f'[Resume] Background parse failed for {resume_id}: {e}', exc_info=True)
                try:
                    from resumes.models import Resume as R
                    r = R.objects.get(id=resume_id)
                    r.parse_status = 'failed'
                    r.parsed_data = {'error': str(e)}
                    r.save()
                except Exception as save_err:
                    logger.error(f'[Resume] Failed to save error status: {save_err}')

        t = threading.Thread(
            target=_parse_in_background,
            args=(str(resume.id), file_path, ext),
            daemon=True
        )
        t.start()

        return Response(resume.to_dict(), status=201)


class ResumeListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        if user.role == 'candidate':
            resumes = Resume.objects(candidate_id=str(user.id))
        elif user.role in ['recruiter', 'admin']:
            candidate_id = request.query_params.get('candidate_id')
            if candidate_id:
                resumes = Resume.objects(candidate_id=candidate_id)
            else:
                resumes = Resume.objects.all()
        else:
            return Response({'error': 'Forbidden.'}, status=403)
        return Response([r.to_dict() for r in resumes])


class ResumeDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, resume_id):
        try:
            resume = Resume.objects.get(id=resume_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Resume not found.'}, status=404)
        user = request.user
        if user.role == 'candidate' and resume.candidate_id != str(user.id):
            return Response({'error': 'Forbidden.'}, status=403)
        return Response(resume.to_dict())


# ─────────────────────────────────────────────────────────────────────────────
# Feature 11 — AI Resume Generator
# ─────────────────────────────────────────────────────────────────────────────

class GenerateResumeView(APIView):
    """
    POST /api/resumes/generate/
    Candidate sends their profile data → OpenAI GPT returns polished resume JSON.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        if request.user.role != 'candidate':
            return Response({'error': 'Only candidates can generate resumes.'}, status=403)

        user = request.user
        job_target = request.data.get('job_target', '')
        
        logger.info(f"[ResumeGen] Generating resume for {user.email}. Target: {job_target}")
        
        try:
            from core.openai_client import generate_resume_content
            
            # Extract fields with safe defaults and explicit list conversion
            skills = list(getattr(user, 'detailed_skills', []))
            work = list(getattr(user, 'work_history', []))
            edu = list(getattr(user, 'education_history', []))
            
            # Simple check to see if profile is usable
            if not user.name and not skills and not work:
                return Response({'error': 'Your profile is empty. Please add some skills or experience before generating a resume.'}, status=400)

            result = generate_resume_content(
                name=user.name,
                email=user.email,
                phone=getattr(user, 'phone', '') or '',
                headline=getattr(user, 'headline', '') or '',
                bio=getattr(user, 'bio', '') or '',
                skills=skills,
                work_history=work,
                education_history=edu,
                location=getattr(user, 'location', '') or '',
                job_target=job_target,
            )
            
            if not result or not isinstance(result, dict) or 'name' not in result:
                logger.error(f"[ResumeGen] AI returned invalid response for {user.email}")
                return Response({'error': 'AI failed to generate resume. Please try adding more profile details.'}, status=500)

            return Response(result)
        except Exception as e:
            logger.error(f"[ResumeGen] Unexpected error for {user.email}: {str(e)}")
            error_msg = str(e).upper()
            if 'QUOTA' in error_msg or 'KEY' in error_msg:
                return Response({'error': 'AI service temporarily unavailable.'}, status=503)
            return Response({'error': f'Resume generation failed: {str(e)}'}, status=500)

