#!/bin/sh
echo "Starting InnovAIte Backend on port: ${PORT:-8000}"
echo "WebSocket support enabled via Daphne ASGI server"
exec daphne -b 0.0.0.0 -p ${PORT:-8000} --proxy-headers core.asgi:application
