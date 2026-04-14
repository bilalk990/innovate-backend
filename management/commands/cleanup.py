"""
Django management command for cleanup tasks
Run with: python manage.py cleanup
"""
from django.core.management.base import BaseCommand
from core.cleanup_service import run_all_cleanup_tasks


class Command(BaseCommand):
    help = 'Run cleanup tasks (orphaned files, expired tokens, etc.)'
    
    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting cleanup tasks...'))
        
        results = run_all_cleanup_tasks()
        
        self.stdout.write(self.style.SUCCESS(
            f'Cleanup completed:\n'
            f'  - Resumes cleaned: {results["resumes_cleaned"]}\n'
            f'  - Tokens cleaned: {results["tokens_cleaned"]}\n'
            f'  - Old interviews found: {results["old_interviews"]}'
        ))
