#!/bin/bash
# InnovAIte Backend Startup Script
# Automatically binds to the Railway assigned port

# Set Django settings module
export DJANGO_SETTINGS_MODULE=core.settings

echo "Starting Backend with Port: $PORT"
echo "Django Settings Module: $DJANGO_SETTINGS_MODULE"

# Run migrations (optional, since using MongoDB)
# python manage.py migrate --noinput

# Start Daphne with PORT from environment
exec daphne -b 0.0.0.0 -p ${PORT:-8000} core.asgi:application
