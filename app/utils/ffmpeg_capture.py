import cv2
import subprocess
import numpy as np
import json
import time

class FFmpegVideoCapture:
    """
    A specific replacement for cv2.VideoCapture that uses ffmpeg subprocess
    to pipe raw video frames for RTSP streams.
    This works around OpenCV backend issues with specific RTSP camera configurations.
    """
    def __init__(self, source):
        self.source = source
        self.pipe = None
        self.width = 0
        self.height = 0
        self.fps = 0
        self._is_opened = False
        
        # 1. Probe for stream info
        if not self._probe_stream():
            print(f"[FFmpeg] Failed to probe stream: {source}")
            return

        # 2. Start ffmpeg process
        self._start_ffmpeg()
        
    def _probe_stream(self):
        """Use ffprobe to get resolution and fps."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate",
            "-of", "json",
            self.source
        ]
        
        try:
            print(f"[FFmpeg] Probing stream: {self.source}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print(f"[FFmpeg] Probe failed: {result.stderr}")
                return False
                
            info = json.loads(result.stdout)
            stream = info['streams'][0]
            
            self.width = int(stream['width'])
            self.height = int(stream['height'])
            
            # FPS calculation
            fps_str = stream.get('avg_frame_rate', '30/1')
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                self.fps = num / den if den != 0 else 30
            else:
                self.fps = float(fps_str)
                
            print(f"[FFmpeg] Probed: {self.width}x{self.height} @ {self.fps}fps")
            return True
            
        except Exception as e:
            print(f"[FFmpeg] Probe error: {e}")
            return False

    def _start_ffmpeg(self):
        """Start the ffmpeg subprocess."""
        command = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-i", self.source,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an",  # No audio
            "-"
        ]
        
        try:
            print(f"[FFmpeg] Starting process for: {self.source}")
            self.pipe = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10**7  # 10MB buffer
            )
            self._is_opened = True
        except Exception as e:
            print(f"[FFmpeg] Start error: {e}")
            self._is_opened = False

    def isOpened(self):
        return self._is_opened

    def read(self):
        if not self._is_opened or self.pipe is None:
            return False, None
            
        try:
            # Read exact frame size
            frame_size = self.width * self.height * 3
            raw_frame = self.pipe.stdout.read(frame_size)
            
            if len(raw_frame) != frame_size:
                print("[FFmpeg] Incomplete frame read or stream ended.")
                self._is_opened = False
                return False, None
                
            frame = np.frombuffer(raw_frame, np.uint8).reshape((self.height, self.width, 3))
            
            # IMPORTANT: Return a copy to ensure it's writable/safe for OpenCV
            return True, frame.copy()
            
        except Exception as e:
            print(f"[FFmpeg] Read error: {e}")
            self._is_opened = False
            return False, None

    def release(self):
        if self.pipe:
            self.pipe.terminate()
            try:
                self.pipe.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.pipe.kill()
            self.pipe = None
        self._is_opened = False
        print("[FFmpeg] Released resource.")

    def get(self, prop_id):
        # Map common OpenCV properties
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        elif prop_id == cv2.CAP_PROP_FPS:
            return self.fps
        return 0
