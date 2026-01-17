# ParkAlisto - Professional Project Structure

## Overview
ParkAlisto is a multi-lot parking analytics system that uses computer vision (YOLOv5) and OCR to track parking occupancy, detect license plates, and identify vehicle colors in real-time.

## Features
- **Real-time Occupancy Tracking**: detailed monitoring of parking spots.
- **License Plate Recognition**: Automatic detection of plate numbers.
- **Vehicle Color Detection**: Classifies vehicle colors.
- **Web Dashboard**: Interactive interface for monitoring and configuration.
- **History Logging**: Searchable history of parked vehicles.

## Setup Instructions

1. **Clone and Navigate:**
   ```bash
   git clone <repository>
   cd ParkAlisto
   ```

2. **Create Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

5. **Run Application:**
   ```bash
   python run.py
   ```

6. **Access Dashboard:**
   Open http://localhost:5000 in your browser.

## Docker Deployment

```bash
docker-compose up --build
```

## Directory Structure
See `[Structure Description]` in the main documentation or file tree.
