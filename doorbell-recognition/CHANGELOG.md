# Changelog

All notable changes to this add-on will be documented in this file.

## [1.0.0] - 2025-01-17

### Added
- Initial release of Doorbell Facial Recognition add-on
- FTP server for camera video uploads
- Facial recognition using CompreFace
- Real-time notifications with face images
- TTS announcements for visitor detection
- Optimized performance for Home Assistant Green
- Imgur image hosting integration
- Home Assistant sensor integration
- Support for Reolink cameras
- Configurable confidence thresholds
- Early exit optimization for high-confidence matches

### Features
- Process 10 frames per video (configurable 5-50)
- Mobile-optimized 480px frame resolution
- No emotion detection for maximum speed
- Automatic face cropping and image upload
- Motion detection fallback notifications
- Multi-device TTS support
- Clickable notification images

### Performance
- Processing time: 4-8 seconds per video
- Memory optimized for 1GB RAM systems
- ARM64 compatibility for Home Assistant Green
- Automatic cleanup of temporary files
