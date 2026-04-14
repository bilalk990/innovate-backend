import os
import bcrypt
import mongoengine as me
from accounts.models import User
from dotenv import load_dotenv

load_dotenv()

# Connect to MongoDB
me.connect(host=os.getenv('MONGODB_URI', 'mongodb://localhost:27017/innovaite_db'))

def setup_admin():
    admin_email = 'admin@innovaite.com'
    admin_pass = 'Admin123!'
    
    existing = User.objects(email=admin_email).first()
    if existing:
        print(f"Admin already exists: {admin_email}")
        existing.role = 'admin'
        existing.is_active = True
        existing.save()
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
    print(f"Created Admin: {admin_email} / {admin_pass}")

if __name__ == '__main__':
    setup_admin()
