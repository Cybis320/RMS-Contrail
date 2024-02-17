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
from scipy.signal import butter, filtfilt
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
        
        self.config = config

        self.timestamps = []
        self.window_size = 6
        self.expected_fps2 = 25
        self.dropped_frames2 = 0
        
        self.last_timestamp = None
        self.video_file = video_file
        self.pts_buffer_size = 500
        self.actual_fps = 29.976 * config.fps / 30
        self.raw_intervals = [1e9 / self.actual_fps for _ in range(self.pts_buffer_size)]
        self.pts_buffer = []
        self.last_smoothed_pts = None
        self.tolerance = 100000 # ns

        # A frame will be considered dropped if it was late more then half a frame
        self.time_for_drop = 1.5*(1.0/config.fps)

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
            self.system_latency = 0.027 # seconds. Experimentally measured latency
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
            expected_interval = 1.0 / self.expected_fps2
            if time_diff > 1.2 * expected_interval:
                dropped = int(time_diff / expected_interval) - 1
                self.dropped_frames2 += dropped

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


    def update_and_filter_pts(self, new_pts):
        # # Update raw intervals if there's at least one previous raw pts
        # if self.pts_buffer:
        #     new_interval = new_pts - self.pts_buffer[-1]
        #     self.raw_intervals.append(new_interval)
        #     # Ensure raw_intervals buffer doesn't exceed its maximum size
        #     if len(self.raw_intervals) > self.pts_buffer_size:
        #         self.raw_intervals.pop(0)

        # # Calculate median interval from raw intervals
        # median_interval = np.median(self.raw_intervals)

        # if self.last_smoothed_pts is not None:
        #     expected_next_pts = self.last_smoothed_pts + median_interval
        #     lower_bound = expected_next_pts - self.tolerance
        #     upper_bound = expected_next_pts + self.tolerance

        #     # Condition 1: Within tolerance
        #     if lower_bound <= new_pts <= upper_bound:
        #         smoothed_pts = new_pts
        #     # Condition 2: More than twice the interval
        #     elif new_pts - self.pts_buffer[-1] > 2 * median_interval:
        #         smoothed_pts = new_pts
        #         print("Detected Dropped Frames")
        #     # Condition 3: Below or above but less than 2x, adjust within tolerance
        #     else:
        #         # Adjust, but not beyond new_pts
        #         if new_pts > upper_bound:
        #             smoothed_pts = min(expected_next_pts, expected_next_pts + self.tolerance)
        #         else:
        #             smoothed_pts = max(expected_next_pts, expected_next_pts - self.tolerance)
        # else:
        #     # First point, no smoothing
        #     smoothed_pts = new_pts

        # # Update last smoothed pts and pts buffer
        # self.last_smoothed_pts = smoothed_pts
        # self.pts_buffer.append(new_pts)  # Append new raw pts
        # # Ensure pts buffer doesn't exceed its maximum size
        # if len(self.pts_buffer) > self.pts_buffer_size:
        #     self.pts_buffer.pop(0)

        return new_pts



    def read(self):
        '''
        Retrieve frames and timestamp.
        :param device: The video capture device or file.
        :return: tuple (ret, frame, timestamp) where ret is a boolean indicating success,
                 frame is the captured frame, and timestamp is the frame timestamp.
        '''
        ret, frame, timestamp = False, None, None

        if self.video_file is not None:
            ret, frame = self.device.read()
            if ret:
                timestamp = None # assigned later
        
        else:
            if self.config.force_v4l2 or self.config.force_cv2:
                ret, frame = self.device.read()
                if ret:
                    timestamp = time.time()
            else:
                sample = self.device.emit("pull-sample")
                if sample:
                    buffer = sample.get_buffer()
                    gst_timestamp_ns = buffer.pts  # GStreamer timestamp in nanoseconds

                    # Validate gst_timestamp_ns to be within a reasonable range 
                    max_expected_ns = 24 * 60 * 60 * 1e9
                    if gst_timestamp_ns > max_expected_ns or gst_timestamp_ns <= 0:
                        # Log this event, handle error, or take corrective action
                        log.info("Unexpected PTS value: {}.".format(gst_timestamp_ns))
                        return False, None, None
                    smoothed_pts = self.update_and_filter_pts(gst_timestamp_ns)
                    
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

                        buffer.unmap(map_info)
                        timestamp = self.start_timestamp + (smoothed_pts / 1e9)
                
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
        device_str = ("rtspsrc  buffer-mode=1 latency=1000 default-rtsp-version=17 protocols=tcp tcp-timeout=5000000 retry=5 "
                      "location=\"{}\" ! rtpjitterbuffer latency=1000 mode=1 ! "
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

        # use a file as the video source
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
            if self.config.force_v4l2:
                log.info("Initialize v4l2 Device.")
                self.device = cv2.VideoCapture(self.config.deviceID, cv2.CAP_V4L2)
                self.device.set(cv2.CAP_PROP_CONVERT_RGB, 0)

            elif self.config.force_cv2 and not self.config.force_v4l2:
                log.info("Initialize OpenCV Device.")
                self.device = cv2.VideoCapture(self.config.deviceID)

            else:
                try:
                    log.info("Initialize GStreamer Device.")
                    # Initialize GStreamer
                    Gst.init(None)

                    # Create and start a GStreamer pipeline
                    self.device = self.create_gstream_device('BGR')

                    # Determine the shape of the GStream
                    sample = self.device.emit("pull-sample")
                    buffer = sample.get_buffer()
                    ret, _ = buffer.map(Gst.MapFlags.READ)
                    if ret:
                        # Get caps and extract video information
                        caps = sample.get_caps()
                        structure = caps.get_structure(0) if caps else None

                        if structure:

                            # Extract width, height, and format
                            width = structure.get_value('width')
                            height = structure.get_value('height')
                            video_format = structure.get_value('format')

                            # Determine the shape based on format
                            if video_format in ['RGB', 'BGR']:
                                self.frame_shape = (height, width, 3)  # RGB or BGR
                                ret, frame, _ = self.read()

                                if ret:
                                    # If frame is grayscale, stop and restart the pipeline in GRAY8 format
                                    if self.is_grayscale(frame):
                                        self.convert_to_gray = True
                                    log.info("Video format: {}, {}P, color: {}".format(video_format, height, not self.convert_to_gray))

                                else:
                                    log.error("Could not determine BGR frame shape.")
                                    return False

                            elif video_format == 'GRAY8':
                                self.frame_shape = (height, width)  # Grayscale
                                log.info("Video format: {}, {}P".format(video_format, height))
                                
                            else:
                                log.error("Unsupported video format: {}.".format(video_format))
                                return False
                        else:
                            log.error("Could not determine frame shape.")
                            return False
                    else:
                        log.error("Could not obtain frame.")
                        return False
                except Exception as e:
                    log.info("Could not initialize GStreamer. Initialize OpenCV Device instead. Error: {}".format(e))
                    self.device = cv2.VideoCapture(self.config.deviceID)
        return True


    def release_resources(self):
        """Releases resources for GStreamer and OpenCV devices."""
        if self.pipeline:
            try:
                self.pipeline.set_state(Gst.State.NULL)
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
                if self.config.force_v4l2 or self.config.force_cv2:
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
                    if self.config.force_v4l2 or self.config.force_cv2:
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
    
