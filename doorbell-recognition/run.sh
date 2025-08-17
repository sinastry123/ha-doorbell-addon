#!/usr/bin/with-contenv bashio

# Get configuration from options
export COMPREFACE_URL=$(bashio::config 'compreface_url')
export DETECTION_API_KEY=$(bashio::config 'detection_api_key')
export RECOGNITION_API_KEY=$(bashio::config 'recognition_api_key')
export HA_TOKEN=$(bashio::config 'ha_token')
export NOTIFICATION_TARGET=$(bashio::config 'notification_target')
export TTS_ENTITY_ID=$(bashio::config 'tts_entity_id')
export TTS_KITCHEN_ENTITY_ID=$(bashio::config 'tts_kitchen_entity_id')
export FTP_USERNAME=$(bashio::config 'ftp_username')
export FTP_PASSWORD=$(bashio::config 'ftp_password')
export RECOGNITION_CONFIDENCE=$(bashio::config 'recognition_confidence')
export DETECTION_CONFIDENCE=$(bashio::config 'detection_confidence')
export MAX_FRAMES=$(bashio::config 'max_frames')

# Set Home Assistant URL (since we're running inside HA)
export HA_URL="http://supervisor/core"

# Set FTP directory to use add-on data directory
export FTP_ROOT_DIR="/data/ftp_doorbell"

bashio::log.info "Starting Doorbell Recognition System..."
bashio::log.info "CompreFace URL: ${COMPREFACE_URL}"
bashio::log.info "FTP Directory: ${FTP_ROOT_DIR}"
bashio::log.info "Notification Target: ${NOTIFICATION_TARGET}"

# Create directories if they don't exist
mkdir -p "${FTP_ROOT_DIR}/faces"
mkdir -p "/config/www/doorbell"

# Start the Python script
python3 /doorbell_recognition.py
