import sys
import os
from mongoengine import connect, Document, StringField, ListField, IntField, BooleanField, DateTimeField
from datetime import datetime

# Setup path to recognize backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Connect to MongoDB (Default localhost)
try:
    connect('innovai_db', host='127.0.0.1', port=27017)
    print("Locked on to MongoDB.")
except Exception as e:
    print(f"Connection failed: {e}")
    sys.exit(1)

class Job(Document):
    title = StringField(required=True)
    company_name = StringField(required=True)
    location = StringField(required=True)
    description = StringField(required=True)
    requirements = ListField(StringField(), default=[])
    job_type = StringField(choices=['Full Time', 'Part Time', 'Contract', 'Internship'], default='Full Time')
    salary_range = StringField(default='')
    is_active = BooleanField(default=True)
    created_at = DateTimeField(default=datetime.utcnow)
    
    meta = {'collection': 'jobs'}

# Seed Data
JOBS = [
    {
        "title": "Senior AI Architect",
        "company_name": "InnovAIte Technologies",
        "location": "San Francisco, CA",
        "description": "Lead the development of our next-gen neural interface. You will be responsible for scaling agentic architectures and ensuring low-latency inference across global nodes.",
        "requirements": ["Python", "PyTorch", "System Design", "Kubernetes", "Vector Databases", "LLMOps"],
        "job_type": "Full Time",
        "salary_range": "$180k - $250k"
    },
    {
        "title": "Full-Stack Engineer (React/Node)",
        "company_name": "Quantum Systems",
        "location": "London, UK (Remote)",
        "description": "Building high-performance dashboards for real-time data visualization. We need a tactical engineer who can handle high-complexity UI states and robust backend microservices.",
        "requirements": ["React", "Node.js", "TypeScript", "Tailwind CSS", "PostgreSQL", "Redis"],
        "job_type": "Full Time",
        "salary_range": "£80k - £120k"
    },
    {
        "title": "Product Designer (UI/UX)",
        "company_name": "Nebula Design Haus",
        "location": "Berlin, Germany",
        "description": "Crafting the 'Elite' look for our premium SaaS platforms. You should have a deep understanding of dark-mode aesthetics, tactical UI, and user experience psychology.",
        "requirements": ["Figma", "Design Systems", "Prototyping", "UI/UX Design", "Motion Design"],
        "job_type": "Contract",
        "salary_range": "€600 - €800 / day"
    },
    {
        "title": "Talent Acquisition Lead",
        "company_name": "Global Scale-up",
        "location": "New York, NY",
        "description": "Revolutionizing technical recruitment by leveraging AI-driven workflows. You will manage the full recruitment lifecycle for our engineering and leadership teams.",
        "requirements": ["Candidate Sourcing", "ATS Proficiency", "Negotiation", "Interview Scheduling", "Employer Branding"],
        "job_type": "Full Time",
        "salary_range": "$140k - $190k"
    },
    {
        "title": "Cyber Security Analyst",
        "company_name": "SafeNet Core",
        "location": "Singapore",
        "description": "Defending critical infrastructure against decentralized threats. Monitoring real-time traffic, performing penetration tests, and ensuring system integrity.",
        "requirements": ["Cyber Security", "Penetration Testing", "Security Audits", "Networking", "Python", "Cloud Security"],
        "job_type": "Full Time",
        "salary_range": "$100k - $150k"
    }
]

def seed():
    print("Initiating Job Seeding Protocol...")
    Job.objects.delete() # Clear existing
    for j_data in JOBS:
        job = Job(**j_data)
        job.save()
        print(f"Deplolyed Job: {job.title} @ {job.company_name}")
    print("Seeding Complete. 5 Professional Jobs Deployed.")

if __name__ == "__main__":
    seed()
