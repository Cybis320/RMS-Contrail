from multiprocessing import Process, Queue, Event, Value, shared_memory
import cv2
import time
import datetime
import Utils.CameraControl as cc
import itertools
import numpy as np

class BufferedFrameCapture(Process):
    def __init__(self, deviceID, buffer_size=250, fps=25, name='BufferedFrameCapture'):
        super().__init__(name=name)
        self.deviceID = deviceID
        
        self.device_opened = Event()
        self.stop_event = Event()
        self.frame_available = Event()


        self.buffer_size = buffer_size
        self.fps = fps

        # Capture to grab latency
        # if timstamp is late, increase latency. If ts is early, decrease latency.
        self.device_buffer = 4 # Experimentally established for imx291 buffer size (does not set the buffer)
        self.system_latency = 0.105 # Experimentally established network + machine latency
        self.total_latency = self.device_buffer / self.fps + self.system_latency

        #  frame properties (modify as needed)
        self.frame_height = 720
        self.frame_width = 1280
        self.frame_channels = 3  # RGB

        # Calculate frame size in bytes
        self.frame_size = self.frame_height * self.frame_width * self.frame_channels
        self.frame_dtype = np.uint8

        # Shared memory for frames
        self.frames_memory = shared_memory.SharedMemory(create=True, size=self.frame_size * buffer_size)
        self.frames_buffer = np.ndarray((buffer_size, self.frame_height, self.frame_width, self.frame_channels), dtype=self.frame_dtype, buffer=self.frames_memory.buf)

        # Shared memory for metadata
        self.metadata_size = buffer_size * 8  # Assuming 8 bytes per timestamp
        self.metadata_memory = shared_memory.SharedMemory(create=True, size=self.metadata_size)
        self.metadata_buffer = np.ndarray((buffer_size,), dtype='d', buffer=self.metadata_memory.buf)  # 'd' for double

        # Initialize pointers for the circular buffer
        self.write_pointer = Value('i', 0)
        self.read_pointer = Value('i', 0)
        
    
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

        # Spinning wheel characters
        wheel = itertools.cycle(['-', '\\', '|', '/'])
        timeout = 0.5 / self.fps
        i = 0

        assert self.capture.isOpened()
        if self.capture.isOpened():
            self.device_opened.set()

        while not self.stop_event.is_set():
            next_write_position = (self.write_pointer.value + 1) % self.buffer_size
            # print(f"run: write pos: {self.write_pointer.value}, read pos: {self.read_pointer.value}")
            if next_write_position == self.read_pointer.value:
                print("Buffer is full. Overwriting oldest frame.")
                with self.read_pointer.get_lock():
                    self.read_pointer.value = (self.read_pointer.value + 1) % self.buffer_size

            # using grab > time > retrieve instead of read > time for more accurate time capture
            if self.capture.grab():
                raw_timestamp = time.time()
                success, img = self.capture.retrieve()
                if success:
                    corrected_timestamp = raw_timestamp - self.total_latency
                    try:
                        # Write frame and timestamp to shared memory
                        np.copyto(self.frames_buffer[self.write_pointer.value % self.buffer_size], img)
                        self.metadata_buffer[self.write_pointer.value % self.buffer_size] = corrected_timestamp
                        with self.write_pointer.get_lock():
                            self.write_pointer.value = (self.write_pointer.value + 1) % self.buffer_size
                        self.frame_available.set()                       
                    except:
                        print("Failed to store frame!")
                    
                    # Calculate the number of frames currently in the buffer
                    buffer_occupancy = (self.write_pointer.value - self.read_pointer.value) % self.buffer_size

                    print(f"\rCapturing! Buffer: {buffer_occupancy} / {self.buffer_size}, Total dropped frames: {i} {next(wheel)}  ", end="", flush=True)
                else:
                    print("Failed to grab a frame. Waiting...")
                    time.sleep(timeout)


    def read(self):
        """Block until the next frame and its timestamp are available from the buffer."""
        while not self.stop_event.is_set():
            # print(f"read: write pos: {self.write_pointer.value}, read pos: {self.read_pointer.value}")
            # Wait until a frame is available or the stop event is set
            if not self.frame_available.wait(timeout=1) and not self.stop_event.is_set():
                continue
            try:
                with self.read_pointer.get_lock():
                    if self.read_pointer.value == self.write_pointer.value:
                        # No new frame available yet
                        continue

                    frame = np.copy(self.frames_buffer[self.read_pointer.value % self.buffer_size])
                    timestamp = self.metadata_buffer[self.read_pointer.value % self.buffer_size]
                    self.read_pointer.value = (self.read_pointer.value + 1) % self.buffer_size

                # Clear the frame available event
                self.frame_available.clear()
                return True, (frame, timestamp)
            except Exception as e:
                print("Exception occurred while waiting for frame:", str(e))
                print("Exception type:", type(e).__name__)

        # If the capture has stopped running, return False
        return False, (None, None)
    
    def start_capture(self):
        self.start()

    def release(self, timeout=None):
        # Signal the process to stop
        self.stop_event.set()

        # Wait for the process to finish
        self.join(timeout=timeout)

        # Close and unlink the shared memory blocks
        self.frames_memory.close()
        self.frames_memory.unlink()
        self.metadata_memory.close()
        self.metadata_memory.unlink()
        print("Shared memory unlinked and closed.")

        # Release the capture device
        if hasattr(self, 'capture') and self.capture.isOpened():
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