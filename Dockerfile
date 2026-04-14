FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

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

# Make startup script executable
RUN chmod +x start.sh

# Create media and static directories
RUN mkdir -p media/resumes staticfiles

# Collect static files (Ping skipped in settings.py during this command)
RUN python manage.py collectstatic --noinput || true

# Expose port
EXPOSE 8000

# Run startup script using shell to properly expand environment variables
CMD ["/bin/bash", "./start.sh"]
