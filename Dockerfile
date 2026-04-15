FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DJANGO_SETTINGS_MODULE=core.settings

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project files
COPY . .

# Create media and static directories
RUN mkdir -p media/resumes staticfiles

# Collect static files (skips DB/secret checks via settings.py guard)
RUN python manage.py collectstatic --noinput || true

# Expose port
EXPOSE 8000

# Run daphne using shell form so $PORT expands from Railway env at runtime
CMD ["/bin/sh", "-c", "exec daphne -b 0.0.0.0 -p ${PORT:-8000} core.asgi:application"]
