# Home Assistant Doorbell Facial Recognition Add-on

A complete doorbell system with FTP server and facial recognition optimized for Home Assistant Green.

## Features

- ✅ Built-in FTP server for camera uploads
- ✅ Facial recognition using CompreFace
- ✅ Optimized for Home Assistant Green performance
- ✅ Real-time notifications with face images
- ✅ TTS announcements
- ✅ Image hosting via Imgur
- ✅ Home Assistant dashboard integration

## Prerequisites

1. **CompreFace Add-on**: Install CompreFace from the add-on store first
2. **Long-lived Access Token**: Generate one in HA Profile settings
3. **Camera with FTP support**: Configure to upload to your HA Green IP on port 2121

## Configuration

All settings can be configured in the add-on configuration tab:

- **compreface_url**: URL to your CompreFace instance (default: http://192.168.50.248:8000)
- **detection_api_key**: Your CompreFace detection API key
- **recognition_api_key**: Your CompreFace recognition API key
- **ha_token**: Your Home Assistant long-lived access token
- **notification_target**: Your mobile device for notifications (e.g., mobile_app_harrys_iphone)
- **tts_entity_id**: Primary speaker for announcements
- **tts_kitchen_entity_id**: Secondary speaker for announcements
- **ftp_username/password**: Credentials for camera FTP uploads
- **recognition_confidence**: Minimum confidence for face recognition (0.1-1.0)
- **detection_confidence**: Minimum confidence for face detection (0.1-1.0)
- **max_frames**: Number of frames to analyze per video (5-50)

## Camera Setup

Configure your Reolink camera to upload to:
- **FTP Server**: Your Home Assistant Green IP address
- **Port**: 2121
- **Username**: reolink (or as configured)
- **Password**: camera123 (or as configured)
- **Upload Path**: / (root)

## Performance

Optimized for Home Assistant Green:
- Processes 10 frames per video by default
- Automatic early exit on high-confidence matches (90%+)
- Mobile-optimized frame resolution (480px)
- No emotion detection for maximum speed
- Processing time: 4-8 seconds per video

## Troubleshooting

**FTP Issues**: Check that port 2121 is not blocked by your router/firewall
**CompreFace Not Found**: Ensure CompreFace add-on is running and accessible
**No Notifications**: Verify your HA token and notification target are correct
**Performance Issues**: Reduce max_frames to 5-8 for slower hardware

## Support

Visit the Home Assistant Community Forum for support and discussions.
