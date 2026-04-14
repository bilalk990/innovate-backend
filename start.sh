#!/bin/bash
# InnovAIte Backend Startup Script
# Automatically binds to the Railway assigned port

echo "Starting Backend with Port: $PORT"

# Run migrations (optional, since using MongoDB)
# python manage.py migrate --noinput

# Start Daphne with PORT from environment
exec daphne -b 0.0.0.0 -p ${PORT:-8000} core.asgi:application
