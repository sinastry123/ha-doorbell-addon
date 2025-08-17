ARG BUILD_FROM=ghcr.io/home-assistant/aarch64-base:latest
FROM $BUILD_FROM

# Install system packages only
RUN apk add --no-cache \
    python3 \
    py3-pip \
    py3-numpy \
    py3-opencv \
    py3-requests \
    py3-pillow \
    ffmpeg

# Create virtual environment with system site packages access
RUN python3 -m venv --system-site-packages /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir pyftpdlib

# Set PATH to use virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /data

# Copy files
COPY doorbell_recognition.py /
COPY run.sh /

# Make executable
RUN chmod a+x /run.sh /doorbell_recognition.py

# Create directories
RUN mkdir -p /data/ftp_doorbell/faces /config/www/doorbell

# Expose ports
EXPOSE 2121 60000-60009

# Start
CMD ["/run.sh"]
