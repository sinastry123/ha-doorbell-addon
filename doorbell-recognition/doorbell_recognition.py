#!/usr/bin/env python3
"""
Optimized Home Assistant Doorbell Facial Recognition Add-on
Removed emotion detection for maximum speed and simplicity
"""

import os
import time
import datetime
import json
import cv2
import tempfile
import threading
import requests
import urllib3
import random
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer
from pyftpdlib.authorizers import DummyAuthorizer

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================================
# CONFIGURATION FROM ENVIRONMENT
# ================================

# Configuration from Environment Variables (for HA Add-on) or Defaults
FTP_ROOT_DIR = os.environ.get('FTP_ROOT_DIR', '/data/ftp_doorbell')
FTP_PORT = 2121
FTP_USERNAME = os.environ.get('FTP_USERNAME', 'reolink')
FTP_PASSWORD = os.environ.get('FTP_PASSWORD', 'camera123')

# CompreFace Settings
COMPREFACE_URL = os.environ.get('COMPREFACE_URL', 'http://homeassistant.local:8000')
DETECTION_API_KEY = os.environ.get('DETECTION_API_KEY', '169e3bdc-ae84-40ec-a26b-e3fe3f144631')
RECOGNITION_API_KEY = os.environ.get('RECOGNITION_API_KEY', 'c67f39c9-606f-40dc-8e72-c9758d20ab7b')

# Home Assistant Settings  
HA_URL = os.environ.get('HA_URL', 'http://supervisor/core')  # Use supervisor API when running as add-on
HA_TOKEN = os.environ.get('HA_TOKEN', 'your_ha_token_here')
NOTIFICATION_TARGET = os.environ.get('NOTIFICATION_TARGET', 'mobile_app_harrys_iphone')
TTS_ENTITY_ID = os.environ.get('TTS_ENTITY_ID', 'media_player.caravan_roof_wifi')
TTS_KITCHEN_ENTITY_ID = os.environ.get('TTS_KITCHEN_ENTITY_ID', 'media_player.kitchen_wifi')

# Recognition Settings - OPTIMIZED
RECOGNITION_CONFIDENCE_THRESHOLD = float(os.environ.get('RECOGNITION_CONFIDENCE', '0.7'))
DETECTION_CONFIDENCE_THRESHOLD = float(os.environ.get('DETECTION_CONFIDENCE', '0.6'))
MAX_FRAMES_TO_ANALYZE = int(os.environ.get('MAX_FRAMES', '10'))  # HA Add-on optimized
EARLY_EXIT_CONFIDENCE = 0.90  # Stop processing if we get excellent match

# Simple TTS Phrases (no emotion)
KNOWN_PHRASES = [
    "{name} was at the front gate {time_ago}",
    "{name} appeared at the front gate {time_ago}",
    "{name} showed up at the front gate {time_ago}",
    "{name} arrived at the front gate {time_ago}",
    "{name} visited the front gate {time_ago}"
]

UNKNOWN_PHRASES = [
    "An unknown person was at the front gate {time_ago}",
    "A visitor appeared at the front gate {time_ago}",
    "Someone was at the front gate {time_ago}",
    "An unidentified person showed up at the front gate {time_ago}"
]

@dataclass
class RecognitionResult:
    """Simplified recognition result data"""
    person_name: str
    confidence: float
    status: str
    processing_time: float
    face_image_url: str = ""

class DoorbellFTPHandler(FTPHandler):
    """Custom FTP handler for doorbell uploads"""
    
    _recognition_system = None
    
    def on_file_received(self, file_path):
        """Process uploaded files"""
        try:
            file_name = Path(file_path).name
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            print(f"[FTP] ✅ Upload: {file_name} ({file_size:.1f} MB)")
            
            # Process video files
            if file_name.lower().endswith(('.mp4', '.avi', '.mov')):
                if self._recognition_system:
                    thread = threading.Thread(
                        target=self._recognition_system.process_video,
                        args=(file_path,),
                        daemon=True
                    )
                    thread.start()
                    
        except Exception as e:
            print(f"[FTP] ❌ Error: {e}")

class DoorbellRecognitionSystem:
    def __init__(self):
        self.ftp_server = None
        self.processing_lock = threading.Lock()
        
    def log(self, message: str, level: str = "INFO"):
        """Simple logging"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    # ================================
    # FTP SERVER
    # ================================
    
    def setup_directories(self):
        """Create required directories"""
        Path(FTP_ROOT_DIR).mkdir(exist_ok=True)
        Path(FTP_ROOT_DIR, "faces").mkdir(exist_ok=True)
        return FTP_ROOT_DIR
    
    def get_host_ip(self):
        """Get Host IP address (works in HA add-on environment)"""
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            # Fallback for HA add-on environment
            try:
                hostname = socket.gethostname()
                ip = socket.gethostbyname(hostname)
                return ip
            except:
                return "192.168.50.193"  # Final fallback
    
    def start_ftp_server(self):
        """Start FTP server"""
        try:
            # Setup
            ftp_root = self.setup_directories()
            host_ip = self.get_host_ip()
            
            # Create authorizer
            authorizer = DummyAuthorizer()
            authorizer.add_user(FTP_USERNAME, FTP_PASSWORD, ftp_root, perm="elradfmwMT")
            
            # Create handler
            handler = DoorbellFTPHandler
            handler.authorizer = authorizer
            handler.banner = "Doorbell FTP Server"
            handler.timeout = 300
            handler.max_upload_file_size = 200 * 1024 * 1024
            handler.permit_foreign_addresses = True
            handler.masquerade_address = host_ip
            handler.passive_ports = range(60000, 60010)
            handler._recognition_system = self
            
            # Create server
            self.ftp_server = FTPServer((host_ip, FTP_PORT), handler)
            
            # Enable socket reuse
            import socket
            self.ftp_server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Start server thread
            server_thread = threading.Thread(target=self.ftp_server.serve_forever, daemon=True)
            server_thread.start()
            
            self.log(f"🚀 FTP server started on {host_ip}:{FTP_PORT}")
            self.log(f"📁 Upload directory: {ftp_root}")
            return True
            
        except Exception as e:
            self.log(f"❌ FTP server failed: {e}", "ERROR")
            return False

    # ================================
    # FILE PROCESSING
    # ================================
    
    def find_video_file(self, original_path: str) -> Optional[str]:
        """Find the actual video file - ALWAYS GET THE NEWEST"""
        # Check if original exists first
        if os.path.exists(original_path):
            self.log(f"📝 Using original file: {Path(original_path).name}")
            return original_path
        
        # Find the NEWEST file that matches the pattern
        directory = Path(original_path).parent
        original_name = Path(original_path).stem
        
        try:
            # Extract base pattern and timestamp from original
            timestamp_match = re.search(r'(\d{14})$', original_name)
            if timestamp_match:
                original_timestamp = timestamp_match.group(1)
                base_pattern = original_name[:original_name.rfind('_')]
                
                self.log(f"🔍 Looking for newest file matching: {base_pattern}_*")
                
                # Find ALL matching files and get the newest one
                matching_files = []
                
                for video_file in directory.glob("*.mp4"):
                    if video_file.stem.startswith(base_pattern):
                        # Extract timestamp
                        file_match = re.search(r'(\d{14})$', video_file.stem)
                        if file_match:
                            file_timestamp = file_match.group(1)
                            file_mtime = video_file.stat().st_mtime
                            
                            matching_files.append({
                                'path': str(video_file),
                                'name': video_file.name,
                                'timestamp': file_timestamp,
                                'mtime': file_mtime
                            })
                
                if matching_files:
                    # Sort by modification time (newest first)
                    matching_files.sort(key=lambda x: x['mtime'], reverse=True)
                    
                    # Get the newest file
                    newest_file = matching_files[0]
                    self.log(f"📝 Found newest matching file: {newest_file['name']}")
                    return newest_file['path']
            
            # Fallback: Get the most recently modified .mp4 file
            self.log("🔍 Fallback: Looking for most recent .mp4 file...")
            
            recent_files = []
            current_time = time.time()
            
            for video_file in directory.glob("*.mp4"):
                try:
                    file_mtime = video_file.stat().st_mtime
                    # Only consider files modified in the last 5 minutes
                    if current_time - file_mtime < 300:
                        recent_files.append((video_file, file_mtime))
                except:
                    continue
            
            if recent_files:
                # Sort by modification time, get newest
                recent_files.sort(key=lambda x: x[1], reverse=True)
                newest_file = recent_files[0][0]
                self.log(f"📝 Using most recent file: {newest_file.name}")
                return str(newest_file)
                
        except Exception as e:
            self.log(f"⚠️ File search error: {e}", "WARNING")
        
        self.log("❌ Could not find any suitable video file", "ERROR")
        return None
    
    def process_video(self, video_path: str):
        """Main video processing function - OPTIMIZED"""
        with self.processing_lock:
            start_time = time.time()
            
            try:
                self.log(f"🎬 Processing: {Path(video_path).name}")
                
                # Wait briefly for camera operations
                time.sleep(0.5)  # Reduced wait time
                
                # Find actual video file
                actual_path = self.find_video_file(video_path)
                if not actual_path:
                    self.log("❌ Video file not found", "ERROR")
                    return
                
                # Extract frames
                frames = self.extract_frames(actual_path)
                if not frames:
                    self.log("❌ No frames extracted", "ERROR")
                    return
                
                # Detect and recognize faces (no emotion processing)
                result = self.recognize_faces(frames)
                if not result:
                    self.send_motion_notification(start_time)
                    return
                
                result.processing_time = time.time() - start_time
                
                # Send notifications
                self.send_notifications(result, start_time)
                
                total_time = time.time() - start_time
                self.log(f"✅ Complete: {result.person_name} ({result.status}) in {total_time:.1f}s")
                
            except Exception as e:
                self.log(f"❌ Processing error: {e}", "ERROR")
            finally:
                # Cleanup frames
                if 'frames' in locals():
                    for frame in frames:
                        try:
                            if os.path.exists(frame):
                                os.unlink(frame)
                        except:
                            pass

    # ================================
    # COMPUTER VISION - OPTIMIZED
    # ================================
    
    def extract_frames(self, video_path: str) -> List[str]:
        """Extract frames from video - OPTIMIZED"""
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_count == 0:
                cap.release()
                return []
            
            # Calculate frame positions - focus on middle 60% where faces are clearest
            start_frame = int(frame_count * 0.2)
            end_frame = int(frame_count * 0.8)
            interval = max(1, (end_frame - start_frame) // MAX_FRAMES_TO_ANALYZE)
            
            frames = []
            self.log(f"🖼️ Extracting {MAX_FRAMES_TO_ANALYZE} frames from {frame_count} frame video...")
            
            for i in range(start_frame, end_frame, interval):
                if len(frames) >= MAX_FRAMES_TO_ANALYZE:
                    break
                    
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if ret:
                    # Save frame optimized for speed
                    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                    frame_path = temp_file.name
                    temp_file.close()
                    
                    # Resize frame for faster processing (mobile optimized)
                    height, width = frame.shape[:2]
                    if width > 480:  # Even smaller for mobile optimization
                        scale = 480.0 / width
                        new_width = 480
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 70])  # Further reduced for speed
                    frames.append(frame_path)
            
            cap.release()
            self.log(f"✅ Extracted {len(frames)} frames")
            return frames
            
        except Exception as e:
            self.log(f"❌ Frame extraction error: {e}", "ERROR")
            return []
    
    def recognize_faces(self, frame_paths: List[str]) -> Optional[RecognitionResult]:
        """Detect and recognize faces - NO EMOTION PROCESSING"""
        detection_url = f"{COMPREFACE_URL}/api/v1/detection/detect"
        recognition_url = f"{COMPREFACE_URL}/api/v1/recognition/recognize"
        headers = {"x-api-key": DETECTION_API_KEY}
        
        best_result = None
        best_frame_path = None
        best_face_data = None
        
        self.log(f"🔍 Analyzing {len(frame_paths)} frames for faces...")
        
        for frame_path in frame_paths:
            try:
                # Face detection
                with open(frame_path, 'rb') as f:
                    files = {"file": ("image.jpg", f, "image/jpeg")}
                    resp = requests.post(detection_url, headers=headers, files=files, timeout=8)
                    
                    if resp.status_code != 200:
                        continue
                    
                    faces = resp.json().get('result', [])
                    if not faces:
                        continue
                    
                    # Get best face
                    best_face = max(faces, key=lambda x: x.get('box', {}).get('probability', 0))
                    confidence = best_face.get('box', {}).get('probability', 0)
                    
                    if confidence < DETECTION_CONFIDENCE_THRESHOLD:
                        continue
                
                # Face recognition
                headers['x-api-key'] = RECOGNITION_API_KEY
                with open(frame_path, 'rb') as f:
                    files = {"file": ("image.jpg", f, "image/jpeg")}
                    resp = requests.post(recognition_url, headers=headers, files=files, timeout=8)
                    
                    if resp.status_code != 200:
                        continue
                    
                    result = resp.json().get('result', [])
                    if not result:
                        continue
                    
                    subjects = result[0].get('subjects', [])
                    if not subjects:
                        continue
                    
                    # Get best match
                    best_match = max(subjects, key=lambda x: x.get('similarity', 0))
                    similarity = best_match.get('similarity', 0)
                    name = best_match.get('subject', 'Unknown')
                    
                    # Determine status
                    if similarity > RECOGNITION_CONFIDENCE_THRESHOLD:
                        status = "recognized"
                    else:
                        name = "Unknown"
                        status = "unknown"
                    
                    # Create result WITHOUT emotion
                    current_result = RecognitionResult(
                        person_name=name,
                        confidence=similarity,
                        status=status,
                        processing_time=0
                    )
                    
                    # Keep best result based on recognition confidence
                    if not best_result or similarity > best_result.confidence:
                        best_result = current_result
                        best_frame_path = frame_path
                        best_face_data = best_face
                        
                        # Early termination if we find a very high confidence match
                        if similarity > EARLY_EXIT_CONFIDENCE:
                            self.log(f"🎯 Excellent match found ({similarity:.3f}), stopping analysis early")
                            break
                    
                headers['x-api-key'] = DETECTION_API_KEY  # Reset header
                
            except Exception as e:
                self.log(f"⚠️ Recognition error: {e}", "WARNING")
                continue
        
        # Save and upload the best face image
        if best_result and best_frame_path and best_face_data:
            try:
                face_url = self.save_and_upload_face_image(best_frame_path, best_face_data, best_result.person_name)
                best_result.face_image_url = face_url or ""
            except Exception as e:
                self.log(f"⚠️ Face save error: {e}", "WARNING")
                best_result.face_image_url = ""
            
            self.log(f"✅ Final result: {best_result.person_name} ({best_result.confidence:.3f})")
        
        return best_result
    
    def save_and_upload_face_image(self, frame_path: str, face_data: Dict, person_name: str) -> Optional[str]:
        """Save and upload face image to Imgur"""
        try:
            # Load frame
            frame = cv2.imread(frame_path)
            if frame is None:
                return None
            
            # Get bounding box
            box = face_data.get('box', {})
            x1 = max(0, int(box.get('x_min', 0)))
            y1 = max(0, int(box.get('y_min', 0)))
            x2 = int(box.get('x_max', frame.shape[1]))
            y2 = int(box.get('y_max', frame.shape[0]))
            
            # Add padding
            padding = 30
            height, width = frame.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)
            
            # Crop face
            face_crop = frame[y1:y2, x1:x2]
            
            # Resize for web viewing
            h, w = face_crop.shape[:2]
            if w > 400:
                scale = 400.0 / w
                new_w = 400
                new_h = max(1, int(h * scale))
                face_crop = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Save locally
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"face_{person_name}_{timestamp}.jpg"
            face_path = Path(FTP_ROOT_DIR) / "faces" / filename
            
            cv2.imwrite(str(face_path), face_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            self.log(f"📸 Face saved: {filename}")
            
            # Upload to Imgur
            upload_url = self.upload_to_imgur(str(face_path), filename)
            
            # Also copy to HA local folder for reliable display
            try:
                # Use config/www/doorbell for HA add-on environment
                ha_local_path = Path("/config/www/doorbell")
                ha_local_path.mkdir(parents=True, exist_ok=True)
                local_file = ha_local_path / filename
                
                import shutil
                shutil.copy2(str(face_path), str(local_file))
                self.log(f"📁 Also saved locally: /local/doorbell/{filename}")
                
            except Exception as e:
                self.log(f"⚠️ Local copy failed: {e}", "WARNING")
                
            return upload_url
            
        except Exception as e:
            self.log(f"❌ Face save error: {e}", "ERROR")
            return None
    
    def upload_to_imgur(self, image_path: str, filename: str) -> Optional[str]:
        """Upload image to Imgur"""
        try:
            self.log("📤 Uploading to Imgur...")
            
            url = 'https://api.imgur.com/3/image'
            headers = {'Authorization': 'Client-ID 546c25a59c58ad7'}
            
            with open(image_path, 'rb') as f:
                files = {'image': f}
                resp = requests.post(url, files=files, headers=headers, timeout=30)
                
                if resp.status_code == 200:
                    result = resp.json()
                    if result.get('success'):
                        upload_url = result['data']['link']
                        self.log("✅ Imgur upload successful")
                        return upload_url
                
            self.log("❌ Imgur upload failed", "ERROR")
            return None
            
        except Exception as e:
            self.log(f"❌ Imgur upload error: {e}", "ERROR")
            return None

    # ================================
    # NOTIFICATIONS - SIMPLIFIED
    # ================================
    
    def format_time_ago(self, start_time: float) -> str:
        """Format elapsed time"""
        seconds = int(time.time() - start_time)
        
        if seconds < 60:
            return f"{seconds} seconds ago" if seconds != 1 else "1 second ago"
        
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        
        if minutes == 1:
            if remaining_seconds == 0:
                return "1 minute ago"
            elif remaining_seconds == 1:
                return "1 minute and 1 second ago"
            else:
                return f"1 minute and {remaining_seconds} seconds ago"
        else:
            if remaining_seconds == 0:
                return f"{minutes} minutes ago"
            elif remaining_seconds == 1:
                return f"{minutes} minutes and 1 second ago"
            else:
                return f"{minutes} minutes and {remaining_seconds} seconds ago"
    
    def get_simple_message(self, result: RecognitionResult, time_ago: str) -> str:
        """Generate simple TTS message without emotion"""
        if result.status == "recognized":
            template = random.choice(KNOWN_PHRASES)
            return template.format(name=result.person_name, time_ago=time_ago)
        else:
            template = random.choice(UNKNOWN_PHRASES)
            return template.format(time_ago=time_ago)
    
    def send_notifications(self, result: RecognitionResult, start_time: float):
        """Send mobile and TTS notifications - SIMPLIFIED"""
        time_ago = self.format_time_ago(start_time)
        simple_message = self.get_simple_message(result, time_ago)
        
        # Send mobile notification
        self.send_mobile_notification(result, time_ago)
        
        # Send TTS
        self.send_tts(simple_message)
        
        # Update Home Assistant sensor
        self.update_ha_sensor(result, time_ago, simple_message)
        
        self.log(f"🔊 TTS: '{simple_message}'")
    
    def send_mobile_notification(self, result: RecognitionResult, time_ago: str):
        """Send mobile notification with clickable image"""
        headers = {
            "Authorization": f"Bearer {HA_TOKEN}",
            "Content-Type": "application/json"
        }
        
        if result.status == "recognized":
            title = f"🚪 Doorbell: {result.person_name}"
            if result.face_image_url:
                message = f"Person: {result.person_name}\nConfidence: {result.confidence:.1%}\nTime: {time_ago}\n\nTap to view face image"
            else:
                message = f"Person: {result.person_name}\nConfidence: {result.confidence:.1%}\nTime: {time_ago}"
        else:
            title = "🚪 Doorbell: Unknown Person"
            if result.face_image_url:
                message = f"Unknown person detected\nConfidence: {result.confidence:.1%}\nTime: {time_ago}\n\nTap to view face image"
            else:
                message = f"Unknown person detected\nConfidence: {result.confidence:.1%}\nTime: {time_ago}"
        
        payload = {
            "message": message,
            "title": title,
            "data": {"push": {"sound": "default"}}
        }
        
        # Add clickable image if available
        if result.face_image_url:
            payload["data"]["url"] = result.face_image_url
            payload["data"]["actions"] = [
                {"action": "VIEW_IMAGE", "title": "View Face", "uri": result.face_image_url}
            ]
        
        try:
            url = f"{HA_URL}/api/services/notify/{NOTIFICATION_TARGET}"
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
            if resp.status_code == 200:
                self.log("✅ Mobile notification sent")
            else:
                self.log(f"⚠️ Mobile notification failed: {resp.status_code}", "WARNING")
        except Exception as e:
            self.log(f"❌ Mobile notification error: {e}", "ERROR")
    
    def send_tts(self, message: str):
        """Send TTS using proven method"""
        headers = {
            "Authorization": f"Bearer {HA_TOKEN}",
            "Content-Type": "application/json"
        }
        
        devices = [TTS_ENTITY_ID, TTS_KITCHEN_ENTITY_ID]
        
        # Turn on devices
        for device in devices:
            try:
                url = f"{HA_URL}/api/services/media_player/turn_on"
                requests.post(url, headers=headers, json={"entity_id": device}, timeout=5)
            except:
                pass
        
        # Wait for wake up
        time.sleep(2)
        
        # Send TTS
        success_count = 0
        for device in devices:
            try:
                url = f"{HA_URL}/api/services/tts/speak"
                payload = {
                    "entity_id": "tts.google_translate_en_com",
                    "media_player_entity_id": device,
                    "message": message,
                    "cache": True
                }
                
                resp = requests.post(url, headers=headers, json=payload, timeout=15)
                if resp.status_code == 200:
                    success_count += 1
                    
            except Exception as e:
                self.log(f"⚠️ TTS error for {device}: {e}", "WARNING")
        
        if success_count > 0:
            self.log(f"✅ TTS sent to {success_count} devices")
        else:
            self.log("❌ TTS failed on all devices", "ERROR")
    
    def update_ha_sensor(self, result: RecognitionResult, time_ago: str, message: str):
        """Update Home Assistant sensor"""
        headers = {
            "Authorization": f"Bearer {HA_TOKEN}",
            "Content-Type": "application/json"
        }
        
        sensor_data = {
            "state": result.person_name,
            "attributes": {
                "confidence": round(result.confidence * 100, 1),
                "status": result.status,
                "time_ago": time_ago,
                "last_message": message,
                "face_image_url": result.face_image_url,
                "detection_time": datetime.datetime.now().isoformat(),
                "friendly_name": "Last Doorbell Visitor",
                "icon": "mdi:account-check" if result.status == "recognized" else "mdi:account-question",
                "entity_picture": result.face_image_url if result.face_image_url else None
            }
        }
        
        try:
            url = f"{HA_URL}/api/states/sensor.doorbell_last_visitor"
            resp = requests.post(url, headers=headers, json=sensor_data, timeout=10)
            if resp.status_code in [200, 201]:
                self.log("📊 HA sensor updated")
            else:
                self.log(f"⚠️ HA sensor update failed: {resp.status_code}", "WARNING")
        except Exception as e:
            self.log(f"⚠️ HA sensor error: {e}", "WARNING")
    
    def send_motion_notification(self, start_time: float):
        """Send notification for motion without face"""
        time_ago = self.format_time_ago(start_time)
        
        # Create motion result
        motion_result = RecognitionResult(
            person_name="Motion",
            confidence=0.0,
            status="no_face",
            processing_time=0.0
        )
        
        # Send simple motion notification
        headers = {
            "Authorization": f"Bearer {HA_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "message": f"Motion detected but no face recognized\nTime: {time_ago}",
            "title": "🚪 Doorbell: Motion Detected",
            "data": {"push": {"sound": "default"}}
        }
        
        try:
            url = f"{HA_URL}/api/services/notify/{NOTIFICATION_TARGET}"
            requests.post(url, headers=headers, json=payload, timeout=10)
            self.log("📱 Motion notification sent")
        except Exception as e:
            self.log(f"⚠️ Motion notification error: {e}", "WARNING")
        
        # Update HA sensor for motion
        motion_message = f"Motion was detected at the front gate {time_ago}"
        self.update_ha_sensor(motion_result, time_ago, motion_message)

    # ================================
    # MAIN SYSTEM
    # ================================
    
    def start_system(self):
        """Start the complete system"""
        import signal
        
        def signal_handler(signum, frame):
            self.log("🛑 Shutting down...")
            self.stop_system()
            exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        self.log("🏠 Starting HA Green Doorbell Recognition System...")
        
        if not self.start_ftp_server():
            return False
        
        self.log("✅ System started successfully on Home Assistant Green!")
        self.log("⚡ Optimized for maximum speed - no emotion detection")
        self.log("📊 Home Assistant integration with image display")
        self.log("📸 Imgur image hosting configured")
        self.log("👁️ Monitoring for uploads...")
        
        try:
            while True:
                time.sleep(60)
                self.log("💓 System running...")
        except KeyboardInterrupt:
            self.stop_system()
    
    def stop_system(self):
        """Stop the system"""
        if self.ftp_server:
            try:
                self.ftp_server.close_all()
                time.sleep(1)
                self.log("✅ FTP server stopped")
            except:
                pass

def main():
    """Main entry point - HA Add-on compatible"""
    print("🏠 Home Assistant Doorbell Recognition Add-on")
    print("=" * 50)
    print("✅ FTP Server")
    print("✅ Facial Recognition with CompreFace")
    print("⚡ Optimized for Maximum Speed")
    print("✅ Simple TTS Phrases")
    print("✅ Home Assistant Integration")
    print("✅ Reliable Imgur Image Hosting")
    print()
    
    # Test CompreFace connection
    try:
        resp = requests.get(f"{COMPREFACE_URL}/api/v1/recognition/faces", timeout=5)
        print("✅ CompreFace connection successful")
    except:
        print("⚠️ CompreFace not detected - install CompreFace add-on first")
        print("   System will work but facial recognition will be limited")
    
    print()
    print("🚀 Starting system automatically in HA add-on mode...")
    
    system = DoorbellRecognitionSystem()
    system.start_system()

if __name__ == "__main__":
    main()
