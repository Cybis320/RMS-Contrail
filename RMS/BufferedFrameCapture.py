from multiprocessing import Process, Queue, Event, Value
import cv2
import time
import datetime
import Utils.CameraControl as cc
import itertools


class BufferedFrameCapture(Process):
    def __init__(self, deviceID, buffer_size=250, fps=25, name='BufferedFrameCapture'):
        super().__init__(name=name)
        self.deviceID = deviceID
        
        self.device_opened = Event()
        self.stop_event = Event()

        self.buffer_size = buffer_size
        self.fps = fps

        # Capture to grab latency
        # if timstamp is late, increase latency. If ts is early, decrease latency.
        self.device_buffer = 4 # Experimentally established for imx291 buffer size (does not set the buffer)
        self.system_latency = 0.105 # Experimentally established network + machine latency
        self.total_latency = self.device_buffer / self.fps + self.system_latency

        self.frame_queue = Queue(maxsize=buffer_size)
        
    
    def isOpened(self):
        return hasattr(self, 'capture') and self.capture.isOpened()


    def run(self):
        self.capture = cv2.VideoCapture(self.deviceID)
        # Flush the network buffer
        i = 0
        j = 0
        interval = 1 / self.fps
        counter = 5 * self.fps # the number of seconds to be stable before releasing

        while True:
            print(f"\r\033[KFlushing {j} frames! {(counter-i-1) // self.fps}", end="", flush=True)
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

        assert self.capture.isOpened()
        if self.capture.isOpened():
            self.device_opened.set()

        # Spinning wheel characters
        wheel = itertools.cycle(['-', '\\', '|', '/'])
        timeout = 0.5 / self.fps
        i = 0

        while not self.stop_event.is_set():
            if self.frame_queue.full():
                # Discard the oldest item (non-blocking get)
                try:
                    self.frame_queue.get(timeout=2)
                    i += 1
                except Exception as e:
                    print("Exception occurred while attempting to drop frame:", str(e))
                    print("Exception type:", type(e).__name__)

            # using grab > time > retrieve instead of read > time for more accurate time capture
            if self.capture.grab():
                raw_timestamp = time.time()
                success, img = self.capture.retrieve()
                if success:
                    corrected_timestamp = raw_timestamp - self.total_latency
                    try:
                        self.frame_queue.put((img, corrected_timestamp), timeout=timeout)
                    except:
                        print("Failed to store frame!")

                    print(f"\rCapturing! Buffer: {self.frame_queue.qsize()} / {self.buffer_size}, , Total dropped frames: {i} {next(wheel)}  ", end="", flush=True)
                else:
                    print("Failed to grab a frame. Waiting...")
                    time.sleep(timeout)


    def read(self):
        """Block until the next frame and its timestamp are available from the buffer."""
        while not self.stop_event.is_set():
            try:
                return True, self.frame_queue.get(timeout=1)
            except Exception as e:
                print("Exception occurred while waiting for frame:", str(e))
                print("Exception type:", type(e).__name__)

        # If the capture has stopped running, return False
        return False, (None, None)

    
    def start_capture(self):
        self.running = True
        self.start()

    def release(self, timeout=None):
        self.stop_event.set()
        self.join(timeout=timeout)
        self.capture.release()


def overlay_timestamp(frame, timestamp):
    # Convert Unix timestamp to a datetime object
    dt = datetime.datetime.utcfromtimestamp(timestamp)
    # Format the datetime object to include milliseconds
    readable_timestamp = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + ' UTC' # Truncate to milliseconds
    font = cv2.FONT_HERSHEY_SIMPLEX
    height, _, _ = frame.shape
    cv2.putText(frame, f"{readable_timestamp}", (10, height - 6), font, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{readable_timestamp}", (10, height - 6), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


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
    
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Assuming 1280x720 resolution; adjust to match your camera's resolution
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (1280, 720))

    buffered_capture = BufferedFrameCapture(cap, buffer_size=250, fps=fps, remove_jitter=True)

    # Start capturing with initial buffer flush
    buffered_capture.start_capture()
    for i in range(256):
        ret, (frame, timestamp) = buffered_capture.read()
        print(f"\n{i}: {timestamp}")
        if ret:
                overlay_timestamp(frame, timestamp)
                
                out.write(frame)
                #cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' key
            break
        
    buffered_capture.release()
    out.release()
    cv2.destroyAllWindows()
    print("\nFile saved!")

if __name__ == '__main__':
    main()