import threading
import cv2 as cv
import time
import datetime
import Utils.CameraControl as cc
from Utils.FrameTimingNormalizer import create_frame_timing_normalizer
from collections import deque
import itertools


class BufferedFrameCapture(threading.Thread):
    def __init__(self, capture, buffer_size=250, fps=25, remove_jitter=False, name='BufferedFrameCapture'):
        self.capture = capture
        assert self.capture.isOpened()

        self.buffer_size = buffer_size
        self.fps = fps

        self.remove_jitter = remove_jitter
        if self.remove_jitter:
            self.normalizer = create_frame_timing_normalizer(window_size=fps*60, fps=fps)

        # Capture to grab latency
        # if timstamp is late, increase latency. If ts is early, decrease latency.
        self.device_buffer = 4 # Experimentally established for imx291 buffer size (does not set the buffer)
        self.system_latency = 0.13 # Experimentally established network + machine latency
        self.total_latency = self.device_buffer / self.fps + self.system_latency

        self.frames = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)

        self.running = False
        super().__init__(name=name)
    
    def isOpened(self):
        # Proxy the call to the underlying VideoCapture object
        return self.capture.isOpened()

    def flush_buffer(self):
        # Flush the network buffer
        i = 0
        j = 0
        interval = 1 / self.fps
        counter = 5 * self.fps # the number of seconds to be stable before releasing

        while True:
            print(f"\r\033[KFlushing {j} frames! {(counter-i-1) / self.fps}", end="", flush=True)
            j += 1

            t0 = time.time()
            self.capture.grab()
            t1 = time.time()
            self.capture.retrieve()
            t2 = time.time()

            grab_wait = t1 - t0
            retreive_time = t2 - t1
            
            # Wait for frame rate to stabilize before releasing
            if grab_wait >= (interval - retreive_time) * 0.95:
                i += 1
                if i >= counter:
                    print("\nNetwork buffer empty. GO!")
                    break

    def run(self):
        # Spinning wheel characters
        wheel = itertools.cycle(['-', '\\', '|', '/'])

        while self.running:
            # using grab > time > retrieve instead of read > time for more accurate time capture
            if self.capture.grab():
                raw_timestamp = time.time()
                success, img = self.capture.retrieve()
                if success:
                    self.frames.append(img)

                    if self.remove_jitter:
                        corrected_timestamp = self.normalizer.correct_timestamp(raw_timestamp)
                        self.timestamps.append(corrected_timestamp - self.total_latency)
                    else:
                        self.timestamps.append(raw_timestamp - self.total_latency)

                    print(f"\rCapturing! Buffer: {len(self.timestamps)} / {self.buffer_size} {next(wheel)}  ", end="", flush=True)
            else:
                print("Failed to grab a frame. Waiting...")
                time.sleep(0.5/self.fps)

    
    def read(self):
        """Block until the next frame and its timestamp are available from the buffer."""
        while self.running:
            if self.frames and self.timestamps:
                return True, (self.frames.popleft(), self.timestamps.popleft())
            else:
                time.sleep(1/self.fps)
        
        # If the capture has stopped running, return False
        return False, (None, None)
    
    def start_capture(self):
        self.flush_buffer()
        self.running = True
        self.start()

    def release(self, timeout=None):
        self.running = False
        self.join(timeout=timeout)
        self.capture.release()


def overlay_timestamp(frame, timestamp):
    # Convert Unix timestamp to a datetime object
    dt = datetime.datetime.utcfromtimestamp(timestamp)
    # Format the datetime object to include milliseconds
    readable_timestamp = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + ' UTC' # Truncate to milliseconds
    font = cv.FONT_HERSHEY_SIMPLEX
    height, _, _ = frame.shape
    cv.putText(frame, f"{readable_timestamp}", (10, height - 6), font, 0.4, (0, 0, 0), 2, cv.LINE_AA)
    cv.putText(frame, f"{readable_timestamp}", (10, height - 6), font, 0.4, (255, 255, 255), 1, cv.LINE_AA)


# Test
def main():
    fps = 25
    # IP Address of the camera
    camera_ip = '192.168.42.10'

    # Command and options
    cmd = 'SetParam'
    opts = ['Encode', 'Video', 'FPS', fps]

    # Call the function
    cc.cameraControl(camera_ip, cmd, opts)

    # Replace with your camera's RTSP URL
    rtsp_url = "rtsp://192.168.42.10:554/user=admin&password=&channel=1&stream=0.sdp"
    

    cap = cv.VideoCapture(rtsp_url)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

    # Assuming 1280x720 resolution; adjust to match your camera's resolution
    out = cv.VideoWriter('output.mp4', cv.VideoWriter_fourcc(*'mp4v'), 25.0, (1280, 720))

    buffered_capture = BufferedFrameCapture(cap, buffer_size=250, fps=fps, remove_jitter=True)

    # Start capturing with initial buffer flush
    buffered_capture.start_capture()
    for i in range(256):
        ret, (frame, timestamp) = buffered_capture.read()
        print(f"\n{i}: {timestamp}")
        if ret:
                overlay_timestamp(frame, timestamp)
                
                out.write(frame)
                #cv.imshow("Frame", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):  # 'q' key
            break
        
    buffered_capture.release()
    out.release()
    cv.destroyAllWindows()
    print("\nFile saved!")

if __name__ == '__main__':
    main()