#!/usr/bin/env python
"""
Quick script to create admin user
Run: python create_admin_user.py
"""
import os
import sys
import django

# Setup Django
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

import bcrypt
from accounts.models import User

def create_admin():
    admin_email = 'admin@innovaite.com'
    admin_pass = 'Admin123!'
    
    existing = User.objects(email=admin_email).first()
    if existing:
        print(f"✓ Admin already exists: {admin_email}")
        existing.role = 'admin'
        existing.is_active = True
        existing.save()
        print(f"✓ Updated to admin role")
        return

    hashed = bcrypt.hashpw(admin_pass.encode(), bcrypt.gensalt()).decode()
    user = User(
        name='System Admin',
        email=admin_email,
        password=hashed,
        role='admin',
        is_active=True
    )
    user.save()
    print(f"✓ Created Admin User")
    print(f"  Email: {admin_email}")
    print(f"  Password: {admin_pass}")
    print(f"\nYou can now login at http://localhost:5173/login")

if __name__ == '__main__':
    create_admin()
