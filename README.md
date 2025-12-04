# ParkAlisto - Multi-Lot Smart Parking System

**ParkAlisto** provides real-time parking lot monitoring across multiple camera feeds using computer vision. The system detects vehicles with YOLOv5, tracks occupancy with confirmation timers, captures license plate screenshots, and serves a responsive dashboard with live combined video streams.

---

**Copyright:** Owned by University of St. La Salle students, Bacolod.

---

## Features

* **Multi-video processing** with parallel threads for simultaneous lot monitoring
* **YOLOv5 vehicle detection** (cars, motorcycles, buses, trucks) with 0.25 confidence threshold
* **Parking spot occupancy tracking** with 10s confirmation timers and 5s screenshot delays
* **EasyOCR license plate recognition** with advanced preprocessing and multiple OCR attempts
* **Vehicle color detection** using KMeans clustering on dominant BGR colors
* **Combined MJPEG video stream** from all cameras with dynamic grid layout
* **Background processing queue** (3 workers) for plate/color analysis without blocking video
* **Real-time dashboard** with lot stats, global queue, parking history, and plate screenshot modals
* **Dashboard features**: Dark/light theme toggle, offline support, fullscreen video, service worker caching
* **REST APIs**: `/data`, `/parkinghistory`, and `/platescreenshots/<filename>`

---

## Quick Start

1. **Ensure Python 3.9+** with a virtual environment and install dependencies:

```bash
pip install torch torchvision ultralytics opencv-python easyocr flask numpy scikit-learn
```

2. **Prepare parking lot mask images**

   * White spots on black background (PNG format)
   * Match each video’s resolution

3. **Save scripts**

   * Main server: `parkalisto.py`
   * Dashboard: `statusdisplay.html` (from paste-2.txt)

4. **Run the server**:

```bash
python parkalisto.py
```

5. **Setup dialog** prompts for:

   * Number of lots (1-10)
   * Lot name
   * Video file (MP4/AVI/MOV)
   * Mask image per lot

6. **Access dashboard**:

   * Dashboard: [http://localhost:5000](http://localhost:5000)
   * Video feed: [http://localhost:5000/videofeed](http://localhost:5000/videofeed)

---

## Configuration

Adjust these constants in `parkalisto.py` before running:

| Parameter                           | Default                  | Description                    |
| ----------------------------------- | ------------------------ | ------------------------------ |
| OCCUPANCY_CONFIRMATION_TIME         | 10.0s                    | Time to confirm vehicle parked |
| VACANCY_CONFIRMATION_TIME           | 10.0s                    | Time to confirm spot vacant    |
| SCREENSHOT_DELAY_AFTER_CONFIRMATION | 5.0s                     | Delay before plate screenshot  |
| FRAME_SKIP_RATE                     | 10                       | Process every 10th frame       |
| NUM_PROCESSING_THREADS              | 3                        | Background OCR/color workers   |
| PROCESSING_WIDTH/HEIGHT             | 416x416                  | YOLO input resize              |
| VEHICLE_CLASSES                     | car/motorcycle/bus/truck | COCO classes detected          |

* Screenshots are saved to `./platescreenshots/` directory

---

## Dashboard Usage

* **Global Status**: Total available spots and processing queue size
* **Lot Cards**: Real-time occupancy, pending changes, last update per lot
* **Live Feed**: Combined multi-camera MJPEG stream (hover for fullscreen)
* **History Log**: Events with plate numbers, colors, vehicle types, and screenshot viewer
* **APIs**: Poll `/data` every 2s for stats, `/parkinghistory` every 3s for logs
* **Offline Support**: Cached content with network status indicators

---

## Architecture

```
Video Feeds (parallel threads)
     ↓
YOLO Detection + Mask Matching
     ↓
Spot Trackers (per spot confirmation)
     ↓
Flask APIs → Tailwind Dashboard (Vue-like reactivity)
     ↓
Combined MJPEG Stream + History Log
```

* Each lot runs an independent `processvideoloop()` thread
* `updatecombinedframe()` stacks frames dynamically (side-by-side or grid)
* Background workers handle OCR/color via shared `processingqueue.Queue()`

---

## Performance Notes

* Optimized for MacBook with PyTorch CPU (GPU auto-detected if available)
* Frame skip + resize keeps CPU <50% on multi-core systems
* Queue prevents video lag during heavy OCR load
* Dashboard updates independently (2s stats, 3s history) to minimize flicker

---

## Troubleshooting

* **No video**: Check file paths, FFmpeg support in OpenCV
* **Poor OCR**: Improve lighting/angle; system tries multiple preprocess methods
* **High CPU**: Increase `FRAME_SKIP_RATE` or reduce resolution
* **Mask issues**: Ensure white parking spots on black, matching video aspect ratio

---

## Contributing

Built for ParkAlisto production deployment. Contributions welcome for:

* RTSP streaming support
* Cloud APIs
* Mobile PWA enhancements
