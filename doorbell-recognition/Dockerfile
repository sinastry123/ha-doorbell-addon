ARG BUILD_FROM
FROM $BUILD_FROM

# Install Python and dependencies using apk only
RUN apk add --no-cache \
    python3 \
    py3-pip \
    py3-numpy \
    py3-opencv \
    py3-requests \
    py3-pillow \
    ffmpeg \
    && pip3 install --break-system-packages --no-cache-dir \
        pyftpdlib

# Set working directory
WORKDIR /data

# Copy the Python script
COPY doorbell_recognition.py /
COPY run.sh /

# Make the scripts executable
RUN chmod a+x /run.sh
RUN chmod a+x /doorbell_recognition.py

# Create directories
RUN mkdir -p /data/ftp_doorbell/faces
RUN mkdir -p /config/www/doorbell

# Expose FTP ports
EXPOSE 2121 60000-60009

# Start the service
CMD ["/run.sh"]
