# VAF Multistream Processing Service

## 📋 Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Folder Structure](#folder-structure)
- [Core Components](#core-components)
- [Features](#features)
- [Notification System](#notification-system)
- [Configuration](#configuration)
- [Installation](#installation)
- [Usage](#usage)
- [API Integration](#api-integration)
- [Troubleshooting](#troubleshooting)

---

## Overview

**VAF Multistream Processing** is a real-time video analytics service that processes multiple RTSP/HTTP video streams in parallel using YOLO object detection and SORT tracking. It supports configurable features like line intrusion detection with flexible notification systems (sync/async).

### Key Capabilities
- ✅ **Parallel Stream Processing**: Multiple streams captured concurrently using threading
- ✅ **Shared Model Inference**: Single YOLO model processes all streams efficiently
- ✅ **Per-Stream Tracking**: Independent SORT trackers maintain object continuity per stream
- ✅ **Configurable Features**: Line intrusion detection with time bounds, direction, size filters
- ✅ **Class-Based Filtering**: Alert only for specific object classes (person, car, etc.)
- ✅ **Dual Notification Modes**: Sync (local file saving) or Async (webhook + SMS)
- ✅ **API-Driven Configuration**: Dynamic feature configuration from remote API

---

## System Architecture

### Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    VAF MULTISTREAM PROCESSING                        │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Stream Thread 1 │     │  Stream Thread 2 │     │  Stream Thread N │
│   (RTSP/HTTP)    │     │   (RTSP/HTTP)    │     │   (RTSP/HTTP)    │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                      ┌───────────▼───────────┐
                      │  Latest Frame Buffer  │
                      │    (Thread-Safe)      │
                      └───────────┬───────────┘
                                  │
                      ┌───────────▼───────────┐
                      │   MAIN THREAD LOOP    │
                      │  (Inference Server)   │
                      └───────────┬───────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
    ┌───────▼────────┐   ┌────────▼────────┐   ┌──────▼──────┐
    │  YOLO Detector │   │  SORT Tracker   │   │   Features  │
    │  (Shared Model)│   │  (Per-Stream)   │   │ (Detection) │
    └───────┬────────┘   └────────┬────────┘   └──────┬──────┘
            │                     │                    │
            └─────────────────────┼────────────────────┘
                                  │
                      ┌───────────▼───────────┐
                      │  Line Intrusion Check │
                      │   • Time Bounds       │
                      │   • Direction Filter  │
                      │   • Size Filter       │
                      │   • Class Filter ✓    │
                      └───────────┬───────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │  Notification Generation  │
                    │                           │
                    │  use_async_notifications  │
                    │         ?                 │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │                           │
           ┌────────▼────────┐      ┌──────────▼──────────┐
           │  SYNC MODE      │      │  ASYNC MODE         │
           │  (Default)      │      │  (Optional)         │
           │                 │      │                     │
           │ • Local file    │      │ • Queue alerts      │
           │   saving        │      │ • Background workers│
           │ • Immediate I/O │      │ • Webhook + SMS     │
           │ • Blocks stream │      │ • Non-blocking ✓    │
           └─────────────────┘      └─────────────────────┘
```

### Thread Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         THREAD LAYOUT                                │
└─────────────────────────────────────────────────────────────────────┘

[Stream Capture Threads - Parallel]
    Thread 1: RTSP Stream 73 → Frame Buffer 73
    Thread 2: RTSP Stream 74 → Frame Buffer 74
    Thread N: RTSP Stream N  → Frame Buffer N

[Main Thread - Sequential Processing]
    while True:
        1. Get latest frames from all buffers (non-blocking)
        2. For each frame:
           a. YOLO detection (shared model)
           b. SORT tracking (per-stream tracker)
           c. Feature processing (line intrusion)
           d. Notification (sync/async)
        3. Update annotated frames for display
        4. Display all streams (if enabled)

[Async Notification Threads - Parallel] (if use_async_notifications=True)
    Worker 1: Process alert queue → Send webhook + SMS
    Worker 2: Process alert queue → Send webhook + SMS
```

---

## Folder Structure

```
vaf-multistream-processing/
│
├── main.py                          # Entry point - main processing loop
├── requirements.txt                 # Python dependencies
├── .env                            # Environment variables (secrets)
├── yolo11n_custom.pt               # YOLO model weights
│
├── api/                            # API integration
│   └── config_fetcher.py           # Fetch stream configs from remote API
│
├── config/                         # Configuration
│   ├── settings.py                 # App settings (model path, etc.)
│   ├── constants.py                # Environment variables (Twilio, webhook)
│   └── stream_config.py            # Stream configuration utilities
│
├── core/                           # Core processing
│   └── frame_processor.py          # YOLO detection + SORT tracking logic
│
├── features/                       # Detection features
│   └── line_intrusion_detector/
│       ├── __init__.py
│       └── line_intrusion_detector.py  # Line crossing detection logic
│
├── notification/                   # Alert notification system
│   ├── local_notification_handler.py   # Sync mode - local file saving
│   ├── async_notification_handler.py   # Async mode - queue + workers
│   └── vaf_alert_service.py            # Webhook + SMS delivery
│
├── stream/                         # Stream management
│   ├── multistream_manager.py      # Multi-stream capture + display
│   ├── stream_handler.py           # Single stream capture + reconnection
│   └── motion_detector.py          # Motion detection (optional)
│
├── Utils/                          # Utilities
│   ├── logger.py                   # Logging configuration
│   ├── Line/                       # Line geometry utilities
│   ├── ObjectDetectors/            # YOLO detector wrapper
│   └── Trackers/                   # SORT tracker implementation
│
├── output/                         # Output directories
│   ├── intrusions/                 # Sync mode alerts (default)
│   └── vaf_alerts/                 # Async mode alerts
│       └── line_intrusion/         # Per-feature subdirectories
│
├── logs/                           # Application logs
└── videos/                         # Test videos (optional)
```

---

## Core Components

### 1. **MultiStreamManager** (`stream/multistream_manager.py`)

Manages parallel stream capture using threading.

**Key Features:**
- One thread per video stream
- Thread-safe frame buffers
- Automatic reconnection on stream failure
- Latest-frame-only strategy (no buffering)

**Usage:**
```python
stream_sources = {
    "73": "rtsp://camera1/stream",
    "74": "rtsp://camera2/stream"
}

manager = MultiStreamManager(stream_sources, enable_display=True)
manager.start_all()  # Start capture threads

frames = manager.get_all_latest_frames()  # Get latest frames
manager.update_annotated_frame(stream_id, annotated_frame)
manager.display_all_streams()  # Show all streams
manager.stop_all()  # Stop all threads
```

---

### 2. **FrameProcessor** (`core/frame_processor.py`)

Central processing unit with shared detection and per-stream tracking.

**Key Features:**
- **Shared YOLO Model**: One model instance for all streams (efficient)
- **Per-Stream SORT Trackers**: Independent tracking per stream
- **Class Detection**: Detects ALL classes, filtering happens at feature level
- **Track History**: Maintains position history for line crossing

**Usage:**
```python
processor = FrameProcessor(
    model_path="yolo11n_custom.pt",
    conf_thresh=0.45
)

annotated_frame = processor.process_frame(
    frame=frame,
    stream_id="74",
    features=[line_detector_instance]
)
```

**Important:** 
- Model detects ALL classes (person, car, truck, etc.)
- Features filter alerts based on `alert_classes` parameter

---

### 3. **LineIntrusionDetector** (`features/line_intrusion_detector/`)

Detects when tracked objects cross a virtual line.

**Key Features:**
- **Line Crossing Detection**: Uses track position history
- **Direction Filtering**: Left, right, or any direction
- **Time Bounds**: Only detect during specific hours
- **Size Filtering**: Filter by object size (bbox dimensions)
- **Class Filtering** ✓: Alert only for specific classes (e.g., ["person", "car"])
- **Dual Notification Modes**: Sync (local) or Async (webhook+SMS)

**Configuration Parameters:**

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `line_coords` | List[Tuple] | Line endpoints [(x1,y1), (x2,y2)] | `[(100, 200), (500, 200)]` |
| `instance_id` | str | Unique instance identifier | `"32"` |
| `stream_id` | str | Stream identifier | `"74"` |
| `direction_to_use` | int | 0=any, 1=left, 2=right | `1` |
| `time_bound_start` | str | Start time (HH:MM:SS) | `"09:00:00"` |
| `time_bound_end` | str | End time (HH:MM:SS) | `"17:00:00"` |
| `bounding_box_start` | Tuple | Reference size start (x, y) | `(100, 100)` |
| `bounding_box_end` | Tuple | Reference size end (x, y) | `(200, 200)` |
| `precision_factor` | int | Size tolerance (0-100%) | `20` |
| `alert_classes` | List[str] | Classes to alert on | `["person", "car"]` |
| `use_async_notifications` | bool | Async (True) or Sync (False) | `True` |

**Alert Classes Filter:**

```python
# Example: Alert only for "person" crossing the line
detector = LineIntrusionDetector(
    line_coords=[(100, 200), (500, 200)],
    instance_id="32",
    stream_id="74",
    alert_classes=["person"],  # ✓ Only alert for person
    use_async_notifications=False
)

# If a car crosses, it will be ignored
# If a person crosses, alert will be generated
```

**How Class Filtering Works:**

1. YOLO detects ALL objects (person, car, truck, etc.)
2. SORT tracks ALL detected objects
3. Line crossing is checked for ALL tracks
4. **Class filter is applied**: If `alert_classes` is set, only objects matching these classes trigger alerts
5. Alert generated only if object class is in `alert_classes` list

---

## Features

### Line Intrusion Detection

**What it does:**
Detects when objects cross a virtual line drawn on the video frame.

**Processing Flow:**

```
1. Check Time Bounds
   ↓ (if within time range)
2. Get Tracked Objects with Class Labels
   ↓
3. For each track:
   ↓
4. Check Class Filter (alert_classes)
   ↓ (if class matches or no filter)
5. Check Line Intersection
   ↓ (if crossed)
6. Check Direction Filter
   ↓ (if direction matches)
7. Check Size Filter
   ↓ (if size matches)
8. Generate Alert
   ↓
9. Sync or Async Notification
```

**Example Scenarios:**

```python
# Scenario 1: Alert for any person crossing left
LineIntrusionDetector(
    line_coords=[(100, 200), (500, 200)],
    instance_id="entry_line",
    stream_id="73",
    direction_to_use=1,  # Left only
    alert_classes=["person"],  # Person only ✓
    use_async_notifications=False
)

# Scenario 2: Alert for cars and trucks crossing (any direction) during business hours
LineIntrusionDetector(
    line_coords=[(200, 100), (200, 400)],
    instance_id="parking_entry",
    stream_id="74",
    direction_to_use=0,  # Any direction
    time_bound_start="08:00:00",
    time_bound_end="18:00:00",
    alert_classes=["car", "truck"],  # Cars and trucks ✓
    use_async_notifications=True
)

# Scenario 3: Alert for any object (no class filter)
LineIntrusionDetector(
    line_coords=[(300, 150), (300, 450)],
    instance_id="general_line",
    stream_id="75",
    alert_classes=None,  # All classes
    use_async_notifications=False
)
```

---

## Notification System

The service supports **two notification modes**: Sync (default) and Async (optional).

### Sync Mode (Default) - Local File Saving

**When to use:**
- Testing and development
- No external notification needed
- Want immediate file saving
- Acceptable to block stream briefly

**How it works:**
```python
# In main.py
detector = LineIntrusionDetector(
    ...,
    use_async_notifications=False  # Sync mode (default)
)

# On line crossing:
# → File saved immediately to output/intrusions/
# → Stream processing pauses briefly during file I/O
# → Simple and straightforward
```

**Output:**
```
output/intrusions/
├── stream74_obj1_20250130_123456.jpg
├── stream74_obj2_20250130_124532.jpg
└── ...
```

**Pros:**
✅ Simple setup (no configuration needed)
✅ No external dependencies
✅ Immediate file saving

**Cons:**
❌ Blocks stream during file I/O (~100-500ms)
❌ No webhook or SMS notifications
❌ Can cause frame drops at high alert rates

---

### Async Mode (Optional) - Webhook + SMS

**When to use:**
- Production deployments
- Need webhook/SMS notifications
- High alert frequency
- Cannot afford frame drops
- Multiple streams with many alerts

**How it works:**
```python
# In main.py
detector = LineIntrusionDetector(
    ...,
    use_async_notifications=True  # Async mode ✓
)

# On line crossing:
# → Alert queued immediately (<1ms)
# → Stream processing continues (non-blocking)
# → Background workers send webhook + SMS
# → Images saved with metadata
```

**Architecture:**

```
Alert Event
    ↓
AsyncNotificationHandler (Shared across all detectors)
    ↓
Queue (max 100 alerts)
    ↓
Background Workers (2 workers)
    ↓
┌─────────────────┬─────────────────┐
│  Webhook POST   │  Twilio SMS     │
│  (async HTTP)   │  (threaded)     │
│  + Image Base64 │  + Alert text   │
└─────────────────┴─────────────────┘
```

**Configuration:**

Add to `.env`:
```bash
# Webhook URL (required)
VAF_WEBHOOK_URL=https://your-endpoint.com/vaf-alerts

# Twilio SMS (optional)
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_FROM_PHONE=+1234567890
TO_PHONE_NUMBERS=+1234567890,+0987654321

# Location identifier
LOCATION=55CPW
```

**Webhook Payload:**

```json
{
  "alert_type": "Line Intrusion",
  "stream_id": "74",
  "instance_id": "32",
  "object_id": "1",
  "location": "55CPW",
  "direction": "left",
  "severity": "high",
  "timestamp": "2025-01-30T12:34:56.789Z",
  "image": "base64_encoded_jpeg_string...",
  "image_filename": "vaf_Line_Intrusion_74_20250130_123456.jpg",
  "image_encoding": "jpeg_base64",
  "image_bytes": 45678,
  "metadata": {
    "bbox": [100, 200, 300, 400],
    "direction": "left",
    "feature_type": "line_intrusion"
  }
}
```

**SMS Format:**
```
🚨 VAF ALERT: Line Intrusion
Location: 55CPW
Stream: 74
Instance: 32
Object ID: 1
Direction: left
Time: 2025-01-30 12:34:56 UTC
```

**Output:**
```
output/vaf_alerts/
└── line_intrusion/
    ├── stream74_obj1_inst32_20250130_123456.jpg
    ├── stream74_obj1_inst32_20250130_123456_meta.json
    └── ...
```

**Pros:**
✅ Non-blocking (stream continues at full FPS)
✅ Webhook + SMS delivery
✅ Retry logic with exponential backoff
✅ Cooldown prevention (120s per stream)
✅ Queue management (no memory issues)
✅ Graceful shutdown (waits for queue)

**Cons:**
❌ Requires external webhook endpoint
❌ Optional Twilio account for SMS
❌ More complex setup

---

### Comparison: Sync vs Async

| Feature | Sync Mode | Async Mode |
|---------|-----------|------------|
| **File Saving** | ✅ Immediate | ✅ Background |
| **Stream Blocking** | ❌ Yes (~100-500ms) | ✅ No (<1ms) |
| **Webhook** | ❌ No | ✅ Yes |
| **SMS** | ❌ No | ✅ Yes (optional) |
| **Queue Management** | ❌ No | ✅ Yes (100 alerts) |
| **Cooldown** | ❌ No | ✅ Yes (120s) |
| **Setup Complexity** | ✅ Simple | ⚠️ Medium |
| **Dependencies** | ✅ None | ⚠️ Webhook, Twilio |
| **Best For** | Testing, Low-traffic | Production, High-traffic |

---

## Configuration

### Environment Variables (`.env`)

```bash
# ===== APP CONFIGURATION =====
BASE_URL=https://api.camzify.com
API_TOKEN=your_api_token_here
MODEL_PATH=yolo11n_custom.pt

# ===== ASYNC NOTIFICATIONS (Optional) =====
# Webhook endpoint for alerts
VAF_WEBHOOK_URL=https://your-webhook-endpoint.com/vaf-alerts

# Twilio SMS configuration (optional)
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_FROM_PHONE=+1234567890
TO_PHONE_NUMBERS=+1234567890,+0987654321

# Location identifier
LOCATION=55CPW
```

### API Configuration Format

The service fetches stream configurations from a remote API. Expected format:

```json
{
  "73": {
    "rtsp_url": "rtsp://camera1.example.com/stream",
    "features": {
      "line_intrusion_detector": [
        {
          "instance_id": "32",
          "line_coords": [[100, 200], [500, 200]],
          "direction_to_use": 1,
          "time_bound_start": "00:00:00",
          "time_bound_end": "23:59:59",
          "bounding_box_start": null,
          "bounding_box_end": null,
          "precision_factor": 0,
          "alert_classes": ["person"]
        }
      ]
    }
  },
  "74": {
    "rtsp_url": "rtsp://camera2.example.com/stream",
    "features": {
      "line_intrusion_detector": [
        {
          "instance_id": "33",
          "line_coords": [[200, 100], [200, 400]],
          "direction_to_use": 0,
          "time_bound_start": "08:00:00",
          "time_bound_end": "18:00:00",
          "alert_classes": ["car", "truck"]
        }
      ]
    }
  }
}
```

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, recommended)
- RTSP camera streams or video files

### Setup

1. **Clone Repository**
```bash
cd E:\PyCharmProjects\DEUTICS-GLOBAL
git clone <repository-url> vaf-multistream-processing
cd vaf-multistream-processing
```

2. **Create Virtual Environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Environment**
```bash
# Copy example and edit
cp .env.example .env
# Edit .env with your settings
```

5. **Download YOLO Model**
```bash
# Place your custom YOLO model in the root directory
# Or it will download default yolo11n.pt
```

---

## Usage

### Basic Usage (Sync Mode)

```bash
python main.py
```

This will:
1. Fetch stream configurations from API
2. Start capture threads for each stream
3. Process frames using shared YOLO model
4. Detect line intrusions
5. Save alerts locally (sync mode)
6. Display annotated streams (if enabled)

### Enable Async Notifications

**Option 1: Edit main.py**
```python
detector = LineIntrusionDetector(
    ...,
    use_async_notifications=True  # Enable async
)
```

**Option 2: Configure via API**
Add `use_async_notifications: true` to API configuration.

### Select Specific Streams

Edit `main.py`:
```python
stream_ids = [73, 74]  # Specific streams
# OR
stream_ids = None  # All active streams
```

### Disable Display

Edit `main.py`:
```python
enable_display = False
```

---

## API Integration

### ConfigFetcher Usage

```python
from api.config_fetcher import ConfigFetcher

fetcher = ConfigFetcher()

# Fetch specific streams with specific features
configs = fetcher.fetch_configs(
    feature_endpoints=["line_intrusion_detector"],
    stream_ids=[73, 74],
    is_active=True
)

# Fetch all active streams
configs = fetcher.fetch_configs(
    feature_endpoints=["line_intrusion_detector"],
    stream_ids=None,
    is_active=True
)
```

### Expected API Response

```json
{
  "stream_id": {
    "rtsp_url": "rtsp://...",
    "https_url": "https://...",  # Alternative
    "features": {
      "line_intrusion_detector": [
        {
          "instance_id": "string",
          "line_coords": [[x1, y1], [x2, y2]],
          "direction_to_use": 0|1|2,
          "time_bound_start": "HH:MM:SS",
          "time_bound_end": "HH:MM:SS",
          "bounding_box_start": [x, y] | null,
          "bounding_box_end": [x, y] | null,
          "precision_factor": 0-100,
          "alert_classes": ["person", "car"] | null
        }
      ]
    }
  }
}
```

---

## Troubleshooting

### Common Issues

#### 1. **No Alerts Generated**

**Possible Causes:**
- Class filter blocking alerts
- Time bounds outside current time
- Direction filter not matching
- Size filter too restrictive

**Solution:**
```python
# Debug: Remove all filters
detector = LineIntrusionDetector(
    line_coords=[(100, 200), (500, 200)],
    instance_id="debug",
    stream_id="74",
    direction_to_use=0,  # Any direction
    time_bound_start="00:00:00",
    time_bound_end="23:59:59",
    alert_classes=None,  # All classes
    use_async_notifications=False
)
```

Check logs:
```
[LineIntrusion:74:debug] Track 1 (person) crossed left
```

#### 2. **Async Notifications Not Working**

**Check `.env`:**
```bash
VAF_WEBHOOK_URL=https://your-endpoint.com/alerts  # Must be set
```

**Check logs:**
```
Shared AsyncNotificationHandler initialized  # Should appear
✅ Notification workers started: 2
```

**Test webhook:**
```bash
curl -X POST $VAF_WEBHOOK_URL \
  -H "Content-Type: application/json" \
  -d '{"test": "alert"}'
```

#### 3. **Stream Not Connecting**

**Check stream URL:**
```python
# Test with VLC or ffplay first
ffplay rtsp://your-camera/stream
```

**Check logs:**
```
Stream 74 thread started
Stream successfully opened: rtsp://...
```

**Reconnection:**
- Service automatically retries failed streams
- Check camera credentials and network

#### 4. **Low FPS / Frame Drops**

**Causes:**
- Too many streams for available CPU/GPU
- Sync notifications blocking processing
- Display window updates slowing down

**Solutions:**
```python
# 1. Disable display
enable_display = False

# 2. Use async notifications
use_async_notifications=True

# 3. Reduce streams or increase hardware
```

#### 5. **Alert Class Not Working**

**Verify YOLO detects the class:**
```python
# Check frame_processor output
print(tracked_objects)
# Output: [{'label': 'person', 'tracker_id': 1, ...}]
```

**Check class name matches exactly:**
```python
alert_classes=["person"]  # ✅ Correct
alert_classes=["Person"]  # ❌ Wrong (case-sensitive)
alert_classes=["people"]  # ❌ Wrong (YOLO uses "person")
```

**Common YOLO class names:**
- `person`
- `car`
- `truck`
- `bus`
- `motorcycle`
- `bicycle`

#### 6. **Queue Full Warnings**

```
⚠️ Notification queue full, dropping alert: stream=74
```

**Solutions:**
```python
# Increase queue size
AsyncNotificationHandler(
    queue_size=200,  # Default: 100
    workers=4        # Default: 2
)
```

Or reduce alert frequency:
```python
# Add cooldown via alert service config
config={"cooldown_seconds": 180}  # Default: 120
```

---

## Performance Tips

### 1. **Optimize Stream Count**
- **Recommended:** 4-8 streams per GPU
- **Max:** Depends on resolution and FPS

### 2. **Reduce Model Inference**
```python
# Lower confidence threshold = more detections
conf_thresh=0.45  # Default

# Higher threshold = fewer false positives
conf_thresh=0.60
```

### 3. **Async vs Sync**
- Use **Async** for >2 streams or high alert rates
- Use **Sync** for testing or low-traffic scenarios

### 4. **Display Optimization**
```python
# Disable display for maximum performance
enable_display = False
```

### 5. **Logging Level**
```python
# In Utils/logger.py
# Set to WARNING or ERROR in production
logging.INFO  # Development
logging.WARNING  # Production
```

---

## Logging

Logs are saved to `logs/` directory.

**Log Levels:**
```python
DEBUG   # Detailed debugging info
INFO    # General information
WARNING # Alerts and important events
ERROR   # Errors and exceptions
```

**Key Log Messages:**

```
# Startup
=== VAF Multistream Processing (API Mode) ===
Loaded configurations for 2 streams
Stream 74 thread started
Shared AsyncNotificationHandler initialized

# Processing
[LineIntrusion:74:32] Track 1 (person) crossed left
✅ Webhook delivered (status=200)
✅ SMS sent to +1234567890 SID=SM123...

# Issues
⚠️ Cooldown active for stream 74: 45.2s < 120s
⚠️ Notification queue full, dropping alert
❌ Stream 73 error: Connection timeout
```

---

## License

[Your License Here]

---

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review this README
3. Contact support team

---

**Last Updated:** 2025-01-30  
**Version:** 1.0.0  
**Author:** VAF Development Team
