# RPi Meteor Station
# Copyright (C) 2015  Dario Zubovic
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function, division, absolute_import

import os
# Set GStreamer debug level. Use '2' for warnings in production environments.
os.environ['GST_DEBUG'] = 'rtspsrc:2'
#os.environ['GST_DEBUG_FILE'] = '/home/pi/RMS_data/gst_debug.log'
import sys
import re
import time
import numpy as np
from math import floor
import logging
import datetime
import os.path
from multiprocessing import Process, Event, Value

import cv2
from RMS.Misc import ping

# Get the logger from the main module
log = logging.getLogger("logger")

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
except ImportError as e:
    log.info('Could not import gi: {}. Using OpenCV.'.format(e))


class BufferedCapture(Process):
    """ Capture from device to buffer in memory.
    """
    
    running = False
    
    def __init__(self, array1, startTime1, array2, startTime2, config, video_file=None):
        """ Populate arrays with (startTime, frames) after startCapture is called.
        
        Arguments:
            array1: numpy array in shared memory that is going to be filled with frames
            startTime1: float in shared memory that holds time of first frame in array1
            array2: second numpy array in shared memory
            startTime2: float in shared memory that holds time of first frame in array2

        Keyword arguments:
            video_file: [str] Path to the video file, if it was given as the video source. None by default.

        """
        
        super(BufferedCapture, self).__init__()
        self.array1 = array1
        self.startTime1 = startTime1
        self.array2 = array2
        self.startTime2 = startTime2
        
        self.startTime1.value = 0
        self.startTime2.value = 0

        self.video_file = video_file

        
        self.config = config
        self.media_backend_override = False

        # Debug section
        self.timestamps = []
        self.window_size = 6
        self.expected_fps2 = 25
        self.dropped_frames2 = 0
        self.last_timestamp = None
        
        # Custom buffer
        self.pts_buffer_size = 25*30 # 30 sec 
        self.pts_buffer = []
        self.average_interval = 0
        self.base_pts = 0
        self.frame_count = 0

        # A frame will be considered dropped if it was late more then half a frame
        self.time_for_drop = 2.0*(1.0/config.fps)

        self.dropped_frames = Value('i', 0)
        self.device = None
        self.pipeline = None
        self.start_timestamp = 0
        self.frame_shape = None
        self.convert_to_gray = False


        # TIMESTAMP LATENCY
        #
        # Experimentally establish device_buffer and device_latency
        #
        # For example:
        #
        # RPi4, GStream, IMX291, 720p @ 25 FPS, VBR
        #     self.device_buffer = 1
        #     self.system_latency = 0.01
        #
        # If timestamp is late, increase latency. If it is early, decrease latency.
        # Formula is: timestamp = time.time() - total_latency

        # TODO: Incorporate variables in .config

        self.device_buffer = 1 # Experimentally measured buffer size (does not set the buffer)
        if self.config.height == 1080:
            self.system_latency = 0.02 # seconds. Experimentally measured latency
        else:
            self.system_latency = 0.01 # seconds. Experimentally measured latency
        self.total_latency = self.device_buffer / self.config.fps + (self.config.fps - 5) / 2000 + self.system_latency



    def save_image_to_disk(self, filename, img_path, img, i):
        try:
            cv2.imwrite(img_path, img)
            log.info(f"Saving completed: i={i}: {filename}")
        except Exception as e:
            log.info(f"Error, could not save image to disk: {e}")
    

    def update_fps(self, frame_timestamp):
        if self.last_timestamp is not None:
            time_diff = frame_timestamp - self.last_timestamp

        self.last_timestamp = frame_timestamp
        self.timestamps.append(frame_timestamp)
        
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)
        
        if len(self.timestamps) > 1:
            elapsed_time = self.timestamps[-1] - self.timestamps[0]
            fps = (len(self.timestamps) - 1) / elapsed_time
            delta_t = time.time() - frame_timestamp
            sys.stdout.write(f"\rMoving Average FPS: {fps:.4f} | Delta_t: {delta_t:.4f}, time diff {time_diff:.6f}")
            sys.stdout.flush()


    def startCapture(self, cameraID=0):
        """ Start capture using specified camera.
        
        Arguments:
            cameraID: ID of video capturing device (ie. ID for /dev/video3 is 3). Default is 0.
            
        """
        
        self.cameraID = cameraID
        self.exit = Event()
        self.start()
    

    def stopCapture(self):
        """ Stop capture.
        """
        
        self.exit.set()

        time.sleep(1)

        log.info("Joining capture...")

        # Wait for the capture to join for 60 seconds, then terminate
        for i in range(60):
            if self.is_alive():
                time.sleep(1)
            else:
                break

        if self.is_alive():
            log.info('Terminating capture...')
            self.terminate()

        return self.dropped_frames.value


    def device_is_opened(self):
        if self.device is None:
            return False
        try:
            if isinstance(self.device, cv2.VideoCapture):
                return self.device.isOpened()
            else:
                state = self.device.get_state(Gst.CLOCK_TIME_NONE).state
                if state == Gst.State.PLAYING:
                    return True
                else:
                    return False
        except Exception as e:
            log.error('Error checking device status: {}'.format(e))
            return False


    def smooth_pts(self, new_pts):

        self.frame_count += 1

        # Append new raw pts
        self.pts_buffer.append(new_pts)

        # Ensure pts buffer doesn't exceed its maximum size
        if len(self.pts_buffer) > self.pts_buffer_size:
            self.pts_buffer.pop(0)
        
        current_buffer_size = len(self.pts_buffer)

        # On initial run or after a reset
        if current_buffer_size == 1:
            self.base_pts = new_pts
            self.frame_count = 1
            smoothed_pts = new_pts

        else:
            # Calculate average interval from raw intervals
            average_interval = (new_pts - self.base_pts) / (self.frame_count - 1)

            # Calculate delay for each PTS and store along with index
            delays = [(self.pts_buffer[i] - self.pts_buffer[0] - i * average_interval, i) for i in range(current_buffer_size)]

            # Number of segments - aiming for 10, but it could be less if the array is small
            num_segments = 10
            segment_size = max(1, current_buffer_size // num_segments)  # Ensure at least 1 per segment

            smoothed_pts_values = []

            for segment in range(num_segments):
                start_idx = segment * segment_size
                end_idx = start_idx + segment_size if segment < num_segments - 1 else current_buffer_size

                # Extract the current segment
                current_segment = delays[start_idx:end_idx]

                if not current_segment:
                    break  # No more data to process

                # Find the smallest delay in the current segment
                _, idx = min(current_segment, key=lambda x: x[0])

                # Calculate smoothed PTS for this smallest delay
                smoothed_pts = self.pts_buffer[idx] + (current_buffer_size - idx - 1) * average_interval
                smoothed_pts_values.append(smoothed_pts)

            # Average the smoothed PTS values from each segment
            smoothed_pts = sum(smoothed_pts_values) / len(smoothed_pts_values) if smoothed_pts_values else 0
            sys.stdout.write(f"\raverage interval: {average_interval/1e6:.3f} ms, delta: {(smoothed_pts - new_pts) / 1e6:.3f} ms")
            sys.stdout.flush()

        return smoothed_pts



    def read(self):
        '''
        Retrieve frames and timestamp.
        :param device: The video capture device or file.
        :return: tuple (ret, frame, timestamp) where ret is a boolean indicating success,
                 frame is the captured frame, and timestamp is the frame timestamp.
        '''
        ret, frame, timestamp = False, None, None

        # Read Video file frame
        if self.video_file is not None:
            ret, frame = self.device.read()
            if ret:
                timestamp = None # assigned later
        
        # Read capture device frame
        else:
            # GStreamer
            if self.config.media_backend == 'gst' and not self.media_backend_override:
                sample = self.device.emit("pull-sample")
                if sample:
                    buffer = sample.get_buffer()
                    gst_timestamp_ns = buffer.pts  # GStreamer timestamp in nanoseconds

                    # Sanity Checking pts value
                    max_expected_ns = 24 * 60 * 60 * 1e9
                    if gst_timestamp_ns > max_expected_ns or gst_timestamp_ns <= 0:
                        # Log this event, and drop frame
                        log.info("Unexpected PTS value: {}.".format(gst_timestamp_ns))
                        return False, None, None
                    
                    
                    ret, map_info = buffer.map(Gst.MapFlags.READ)
                    if ret:
                        # If all channels contains colors, or there is only one channel, keep channel(s) 
                        if not self.convert_to_gray:
                            frame = np.ndarray(shape=self.frame_shape, buffer=map_info.data, dtype=np.uint8)

                        # If channels contains no colors, discard two channels
                        else:
                            bgr_frame = np.ndarray(shape=self.frame_shape, buffer=map_info.data, dtype=np.uint8)
                            
                            # select a specific channel
                            gray_frame = bgr_frame[:, :, 0]
                                                        
                            frame = gray_frame

                        # Smooth raw timestamp
                        smoothed_pts = self.smooth_pts(gst_timestamp_ns)
                        timestamp = self.start_timestamp + (smoothed_pts / 1e9)
                    else:
                        log.info("Gst Buffer did not contain a frame.")
                        return False, None, None
                    
                    buffer.unmap(map_info)
                else:
                    log.info("Gst device did not emit a sample.")
                    return False, None, None

            # OpenCV
            else:
                ret, frame = self.device.read()
                if ret:
                    timestamp = time.time()
                
        return ret, frame, timestamp


    def extract_rtsp_url(self, input_string):
        # Define the regular expression pattern
        pattern = r'(rtsp://.*?\.sdp)/?'

        # Search for the pattern in the input string
        match = re.search(pattern, input_string)

        # Extract and format the RTSP URL
        if match:
            rtsp_url = match.group(1)  # Extract the matched URL
            if not rtsp_url.endswith('/'):
                rtsp_url += '/'  # Add '/' if it's missing
            return rtsp_url
        else:
            return None  # Return None if no RTSP URL is found
            

    def is_grayscale(self, frame):
        # Check if the R, G, and B channels are equal
        b, g, r = cv2.split(frame)
        if np.array_equal(r, g) and np.array_equal(g, b):
            return True
        return False


    def create_gstream_device(self, video_format):
        """
        Creates a GStreamer pipeline for capturing video from an RTSP source and 
        initializes playback with specific configurations.

        The method also sets an initial timestamp for the pipeline's operation.

        Parameters:
        - video_format (str): The desired video format for the conversion, 
        e.g., 'BGR', 'GRAY8', etc.

        Returns:
        - Gst.Element: The appsink element of the created GStreamer pipeline, 
        which can be used for further processing of the captured video frames.
        """

        device_url = self.extract_rtsp_url(self.config.deviceID)
        # device_str = ("rtspsrc  buffer-mode=1 latency=1000 default-rtsp-version=17 protocols=tcp tcp-timeout=5000000 retry=5 "
        #               "location=\"{}\" ! rtpjitterbuffer latency=1000 mode=1 ! "
        #               "rtph264depay ! h264parse ! avdec_h264").format(device_url)

        device_str = ("rtspsrc  buffer-mode=1 protocols=tcp tcp-timeout=5000000 retry=5 "
                      "location=\"{}\" ! "
                      "rtph264depay ! h264parse ! avdec_h264").format(device_url)

        conversion = "videoconvert ! video/x-raw,format={}".format(video_format)
        pipeline_str = ("{} ! queue leaky=downstream max-size-buffers=100 max-size-bytes=0 max-size-time=0 ! "
                        "{} ! queue max-size-buffers=100 max-size-bytes=0 max-size-time=0 ! "
                        "appsink max-buffers=100 drop=true sync=0 ts-offset=1000000000 name=appsink").format(device_str, conversion)

        
        self.pipeline = Gst.parse_launch(pipeline_str)

        self.pipeline.set_state(Gst.State.PLAYING)
        self.start_timestamp = time.time() - self.total_latency
        start_time_str = datetime.datetime.fromtimestamp(self.start_timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')
        log.info("Start time is {}".format(start_time_str))

        return self.pipeline.get_by_name("appsink")


    def initVideoDevice(self):
        """ Initialize the video device. """

        # Use a file as the video source
        if self.video_file is not None:
            self.device = cv2.VideoCapture(self.video_file)

        # Use a device as the video source
        else:

            # If an analog camera is used, skip the ping
            ip_cam = False
            if "rtsp" in str(self.config.deviceID):
                ip_cam = True


            if ip_cam:

                ### If the IP camera is used, check first if it can be pinged

                # Extract the IP address
                ip = re.findall(r"[0-9]+(?:\.[0-9]+){3}", self.config.deviceID)

                # Check if the IP address was found
                if ip:
                    ip = ip[0]

                    # Try pinging 5 times
                    ping_success = False

                    for i in range(500):

                        print('Trying to ping the IP camera...')
                        ping_success = ping(ip)

                        if ping_success:
                            log.info("Camera IP ping successful!")
                            break

                        time.sleep(5)

                    if not ping_success:
                        log.error("Can't ping the camera IP!")
                        return False

                else:
                    log.error("Can't find the camera IP!")
                    return False


            # Init the video device
            log.info("Initializing the video device...")
            log.info("Device: " + str(self.config.deviceID))

            if self.config.media_backend == 'gst':
                try:
                    log.info("Initialize GStreamer Device.")
                    # Initialize GStreamer
                    Gst.init(None)

                    # Create and start a GStreamer pipeline
                    self.device = self.create_gstream_device('BGR')

                    # Reset pts buffer
                    self.pts_buffer = []

                    # Determine the shape of the GStream
                    sample = self.device.emit("pull-sample")
                    buffer = sample.get_buffer()
                    if sample:
                        buffer = sample.get_buffer()
                        ret, map_info = buffer.map(Gst.MapFlags.READ)

                        if ret:

                            # Get caps and extract video information
                            caps = sample.get_caps()
                            structure = caps.get_structure(0) if caps else None

                            if structure:

                                # Extract width, height, and format
                                width = structure.get_value('width')
                                height = structure.get_value('height')

                                self.frame_shape = (height, width, 3)
                                frame = np.ndarray(shape=self.frame_shape, buffer=map_info.data, dtype=np.uint8)
                                # If frame is grayscale, set convert_to_gray flag
                                if self.is_grayscale(frame):
                                    self.convert_to_gray = True

                                log.info("Video format: BGR, {}P, color: {}".format(height, not self.convert_to_gray))
                                    
                            else:
                                log.error("Could not determine frame shape.")
                                self.media_backend_override = True
                                self.release_resources()
                        else:
                            log.error("Could not obtain frame.")
                            self.media_backend_override = True
                            self.release_resources()
                    else:
                        log.error("Could not obtain sample.")
                        self.media_backend_override = True
                        self.release_resources()

                except Exception as e:
                    log.info("Could not initialize GStreamer. Initialize OpenCV Device instead. Error: {}".format(e))
                    self.media_backend_override = True
                    self.release_resources()

            if self.config.media_backend == 'v4l2':
                try:
                    log.info("Initialize v4l2 Device.")
                    self.device = cv2.VideoCapture(self.config.deviceID, cv2.CAP_V4L2)
                    self.device.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                except Exception as e:
                    log.info("Could not initialize v4l2. Initialize OpenCV Device instead. Error: {}".format(e))
                    self.media_backend_override = True
                    self.release_resources()


            elif self.config.media_backend == 'cv2' or self.media_backend_override:
                log.info("Initialize OpenCV Device.")
                self.device = cv2.VideoCapture(self.config.deviceID)

        return True


    def release_resources(self):
        """Releases resources for GStreamer and OpenCV devices."""
        if self.pipeline:
            try:
                self.pipeline.set_state(Gst.State.NULL)
                time.sleep(5)
                log.info('GStreamer Video device released!')
            except Exception as e:
                log.error('Error releasing GStreamer pipeline: {}'.format(e))
                
        if self.device:
            try:
                if isinstance(self.device, cv2.VideoCapture):
                    self.device.release()
                    log.info('OpenCV Video device released!')
            except Exception as e:
                log.error('Error releasing OpenCV device: {}'.format(e))
            finally:
                self.device = None  # Reset device to None after releasing


    def run(self):
        """ Capture frames.
        """
        
        # Init the video device
        while not self.exit.is_set() and not self.initVideoDevice():
            log.info('Waiting for the video device to be connect...')
            time.sleep(5)

        # Create dir to save jpg files
        stationID = str(self.config.stationID)
        date_string = time.strftime("%Y%m%d_%H%M%S", time.gmtime(time.time()))
        dirname = f"JPG_{stationID}_"+ date_string
        dirname = os.path.join(self.config.data_dir, self.config.jpg_dir, dirname)

        # Create the directory
        os.makedirs(dirname, exist_ok=True)

        if self.device is None:

            log.info('The video source could not be opened!')
            self.exit.set()
            return False


        # Wait until the device is opened
        device_opened = False
        for i in range(20):
            time.sleep(1)
            if self.device_is_opened():
                device_opened = True
                break


        # If the device could not be opened, stop capturing
        if not device_opened:
            log.info('The video source could not be opened!')
            self.exit.set()
            return False

        else:
            log.info('Video device opened!')


        # Keep track of the total number of frames
        total_frames = 0


        # For video devices only (not files), throw away the first 10 frames
        if self.video_file is None and isinstance(self.device, cv2.VideoCapture):

            first_skipped_frames = 10
            for i in range(first_skipped_frames):
                self.read()

            total_frames = first_skipped_frames


        # If a video file was used, set the time of the first frame to the time read from the file name
        if self.video_file is not None:
            time_stamp = "_".join(os.path.basename(self.video_file).split("_")[1:4])
            time_stamp = time_stamp.split(".")[0]
            video_first_time = datetime.datetime.strptime(time_stamp, "%Y%m%d_%H%M%S_%f")
            log.info("Using a video file: " + self.video_file)
            log.info("Setting the time of the first frame to: " + str(video_first_time))

            # Convert the first time to a UNIX timestamp
            video_first_timestamp = (video_first_time - datetime.datetime(1970, 1, 1)).total_seconds()


        # Use the first frame buffer to start - it will be flip-flopped between the first and the second
        #   buffer during capture, to prevent any data loss
        buffer_one = True

        wait_for_reconnect = False

        last_frame_timestamp = False
        
        # Run until stopped from the outside
        while not self.exit.is_set():


            # Wait until the compression is done (only when a video file is used)
            if self.video_file is not None:
                
                wait_for_compression = False

                if buffer_one:
                    if self.startTime1.value == -1:
                        wait_for_compression = True
                else:
                    if self.startTime2.value == -1:
                        wait_for_compression = True

                if wait_for_compression:
                    log.debug("Waiting for the {:d}. compression thread to finish...".format(int(not buffer_one) + 1))
                    time.sleep(0.1)
                    continue



            
            if buffer_one:
                self.startTime1.value = 0
            else:
                self.startTime2.value = 0
            

            # If the video device was disconnected, wait 5s for reconnection
            if wait_for_reconnect:

                print('Reconnecting...')

                while not self.exit.is_set() and not self.initVideoDevice():

                    log.info('Waiting for the video device to be reconnected...')

                    time.sleep(5)


                    if self.device is None:
                        print("The video device couldn't be connected! Retrying...")
                        continue


                    if self.exit.is_set():
                        break

                    # Read the frame
                    log.info("Reading frame...")
                    ret, _, _ = self.read()
                    log.info("Frame read!")

                    # If the connection was made and the frame was retrieved, continue with the capture
                    if ret:
                        log.info('Video device reconnected successfully!')
                        wait_for_reconnect = False
                        break


                wait_for_reconnect = False


            t_frame = 0
            t_assignment = 0
            t_convert = 0
            t_block = time.time()
            max_frame_interval_normalized = 0.0
            max_frame_age_seconds = 0.0


            # Capture a block of 256 frames
            block_frames = 256

            log.info('Grabbing a new block of {:d} frames...'.format(block_frames))

            # Debug code
            while False:
                ret, frame, frame_timestamp = self.read()
                if not ret:
                    break  # Exit the loop if the frame read was unsuccessful
                self.update_fps(frame_timestamp)
                
            for i in range(block_frames):


                # Read the frame (keep track how long it took to grab it)
                t1_frame = time.time()
                ret, frame, frame_timestamp = self.read()
                t_frame = time.time() - t1_frame


                # If the video device was disconnected, wait for reconnection
                if (self.video_file is None) and (not ret):

                    log.info('Frame grabbing failed, video device is probably disconnected!')
                    self.release_resources()
                    wait_for_reconnect = True
                    break


                # If a video file is used, compute the time using the time from the file timestamp
                if self.video_file is not None:
                
                    frame_timestamp = video_first_timestamp + total_frames/self.config.fps

                    # print("tot={:6d}, i={:3d}, fps={:.2f}, t={:.8f}".format(total_frames, i, self.config.fps, frame_timestamp))


                # Set the time of the first frame
                if i == 0:

                    # Initialize last frame timestamp if it's not set
                    if not last_frame_timestamp:
                        last_frame_timestamp = frame_timestamp
                    
                    # Always set first frame timestamp in the beginning of the block
                    first_frame_timestamp = frame_timestamp

                if self.video_file is None:

                    # If a video device is used, save a jpg every nth frames
                    # if i % 64 == 0:   > img every 2.56s, 3.7GB per day @ 25 fps
                    # if i % 128 == 0:   > img every 5.12s, 1.9GB per day @ 25 fps
                    # if i == 0:   > img every 10.24s, 0.9GB per day @ 25 fps
                    if i % 128 == 0:

                        # Generate the name for the file
                        date_string = time.strftime("%Y%m%d_%H%M%S", time.gmtime(frame_timestamp))

                        # Calculate miliseconds
                        millis = int((frame_timestamp - floor(frame_timestamp))*1000)
                        
                        # Create the filename
                        filename = f"{stationID}_"+ date_string + "_" + str(millis).zfill(3) + ".jpg"

                        img_path = os.path.join(dirname, filename)

                        # Save the image to disk
                        try:
                            self.save_image_to_disk(filename, img_path, frame,i)
                        except:
                            log.error("Could not save {:s} to disk!".format(filename))

                # If the end of the video file was reached, stop the capture
                if self.video_file is not None:
                    if (frame is None) or (not self.device_is_opened()):

                        log.info("End of video file!")
                        log.debug("Video end status:")
                        log.debug("Frame:" + str(frame))
                        log.debug("Device open:" + str(self.device_is_opened()))

                        self.exit.set()
                        time.sleep(0.1)
                        break


                # Check if frame is dropped if it has been more than 1.5 frames than the last frame
                elif (frame_timestamp - last_frame_timestamp) >= self.time_for_drop:
                    
                    # Calculate the number of dropped frames
                    n_dropped = int((frame_timestamp - last_frame_timestamp)*self.config.fps)

                    self.dropped_frames.value += n_dropped

                    if self.config.report_dropped_frames:
                        log.info("{}/{} frames dropped or late! Time for frame: {:.3f}, convert: {:.3f}, assignment: {:.3f}".format(
                            str(n_dropped), str(self.dropped_frames.value), t_frame, t_convert, t_assignment))

                # If cv2:
                if self.config.media_backend != 'gst' and not self.media_backend_override:
                    # Calculate the normalized frame interval between the current and last frame read, normalized by frames per second (fps)
                    frame_interval_normalized = (frame_timestamp - last_frame_timestamp) / (1 / self.config.fps)
                    # Update max_frame_interval_normalized for this cycle
                    max_frame_interval_normalized = max(max_frame_interval_normalized, frame_interval_normalized)

                # If GStreamer:
                else:
                    # Calculate the time difference between the current time and the frame's timestamp
                    frame_age_seconds = time.time() - frame_timestamp
                    # Update max_frame_age_seconds for this cycles
                    max_frame_age_seconds = max(max_frame_age_seconds, frame_age_seconds)

                # On the last loop, report late or dropped frames
                if i == block_frames - 1:
                    # For cv2, show elapsed time since frame read to assess loop performance
                    if self.config.media_backend != 'gst' and not self.media_backend_override:
                        log.info("Block's max frame interval: {:.3f} (normalized). Run's late frames: {}".format(max_frame_interval_normalized, self.dropped_frames.value))
                    
                    # For GStreamer, show elapsed time since frame capture to assess sink fill level
                    else:
                        log.info("Block's max frame age: {:.3f} seconds. Run's dropped frames: {}".format(max_frame_age_seconds, self.dropped_frames.value))

                last_frame_timestamp = frame_timestamp




                ### Convert the frame to grayscale ###

                t1_convert = time.time()

                # Convert the frame to grayscale
                #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Convert the frame to grayscale
                if len(frame.shape) == 3:

                    # If a color image is given, take the green channel
                    if frame.shape[2] == 3:

                        gray = frame[:, :, 1]

                    # If UYVY image given, take luma (Y) channel
                    elif self.config.uyvy_pixelformat and (frame.shape[2] == 2):
                        gray = frame[:, :, 1]

                    # Otherwise, take the first available channel
                    else:
                        gray = frame[:, :, 0]

                else:
                    gray = frame


                # Cut the frame to the region of interest (ROI)
                gray = gray[self.config.roi_up:self.config.roi_down, \
                    self.config.roi_left:self.config.roi_right]

                # Track time for frame conversion
                t_convert = time.time() - t1_convert


                ### ###




                # Assign the frame to shared memory (track time to do so)
                t1_assign = time.time()
                if buffer_one:
                    self.array1[i, :gray.shape[0], :gray.shape[1]] = gray
                else:
                    self.array2[i, :gray.shape[0], :gray.shape[1]] = gray

                t_assignment = time.time() - t1_assign



                # Keep track of all captured frames
                total_frames += 1




            if self.exit.is_set():
                wait_for_reconnect = False
                log.info('Capture exited!')
                break


            if not wait_for_reconnect:

                # Set the starting value of the frame block, which indicates to the compression that the
                # block is ready for processing
                if buffer_one:
                    self.startTime1.value = first_frame_timestamp

                else:
                    self.startTime2.value = first_frame_timestamp

                log.info('New block of raw frames available for compression with starting time: {:s}'.format(str(first_frame_timestamp)))

            
            # Switch the frame block buffer flags
            buffer_one = not buffer_one
            if self.config.report_dropped_frames:
                log.info('Estimated FPS: {:.3f}'.format(block_frames/(time.time() - t_block)))
        

        log.info('Releasing video device...')
        self.release_resources()
    
