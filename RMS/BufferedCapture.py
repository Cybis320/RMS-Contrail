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
os.environ['GST_DEBUG'] = '3'

import re
import time
import logging
import datetime
import os.path
from multiprocessing import Process, Event

from math import floor

# pylint: disable=E1101
import cv2

import numpy as np

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

from RMS.Misc import ping

# Get the logger from the main module
log = logging.getLogger("logger")


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

        self.video_file = video_file

        # A frame will be considered dropped if it was late more then half a frame
        self.time_for_drop = 1.9/config.fps

        self.dropped_frames = 0

        self.pipeline = None
        self.start_timestamp = 0
        self.frame_shape = None

    def save_image_and_log_time(self, filename, img_path, img, i):
        try:
            cv2.imwrite(img_path, img)
            log.info(f"Saving completed: i={i}: {filename}")
        except Exception as e:
            log.info(f"Error in save_image_and_log_time: {e}")


    def startCapture(self, cameraID=0):
        """ Start capture using specified camera.
        
        Arguments:
            cameraID: ID of video capturing device (ie. ID for /dev/video3 is 3). Default is 0.
            
        """
        
        self.cameraID = cameraID
        self.exit = Event()
        self.start()
    

    def read(self, device):
        '''
        Retrieve frames and timestamp.
        :param device: The video capture device or file.
        :return: tuple (ret, frame, timestamp) where ret is a boolean indicating success,
                 frame is the captured frame, and timestamp is the frame timestamp.
        '''
        ret, frame, timestamp = False, None, None

        if self.video_file is not None:
            ret, frame = device.read()
            if ret:
                timestamp = time.time()
        
        else:
            if self.config.force_v4l2:
                ret, frame = device.read()
                if ret:
                    timestamp = time.time()
            else:
                sample = device.emit("pull-sample")
                if sample:
                    buffer = sample.get_buffer()
                    gst_timestamp_ns = buffer.pts  # GStreamer timestamp in nanoseconds

                    ret, map_info = buffer.map(Gst.MapFlags.READ)
                    if ret:
                        frame = np.ndarray(shape=self.frame_shape, buffer=map_info.data, dtype=np.uint8)
                        buffer.unmap(map_info)
                        timestamp = self.start_timestamp + (gst_timestamp_ns / 1e9)
                
        return ret, frame, timestamp
    

    def is_grayscale(self, frame):
        # Check if the R, G, and B channels are equal
        b, g, r = cv2.split(frame)
        if np.array_equal(r, g) and np.array_equal(g, b):
            return True
        return False


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


    def initVideoDevice(self):
        """ Initialize the video device. """

        device = None

        # use a file as the video source
        if self.video_file is not None:
            device = cv2.VideoCapture(self.video_file)

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
                        return None

                else:
                    return None



            # Init the video device
            log.info("Initializing the video device...")
            log.info("Device: " + str(self.config.deviceID))
            if self.config.force_v4l2:
                device = cv2.VideoCapture(self.config.deviceID, cv2.CAP_V4L2)
                device.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            else:
                log.info("Initialize GStreamer Device: ")
                #device = cv2.VideoCapture(self.config.deviceID)
                # Initialize GStreamer
                Gst.init(None)

                deviceID = self.config.deviceID
                conversion = "videoconvert ! video/x-raw,format=BGR"
                pipeline_str = (f"rtspsrc location={deviceID} protocols=udp ! rtph264depay ! avdec_h264 ! {conversion} ! appsink name=appsink")

                # Create GStreamer pipeline
                self.pipeline = Gst.parse_launch(pipeline_str)
                self.start_timestamp = time.time()

                # Start the pipeline
                self.pipeline.set_state(Gst.State.PLAYING)
                device = self.pipeline.get_by_name("appsink")

                # Determine the shape of the GStream
                sample = device.emit("pull-sample")
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
                            ret, frame, _ = self.read(device)

                            # If frame is grayscale, stop and restart the pipeline in GRAY8 format
                            if self.is_grayscale(frame):
                                self.pipeline.set_state(Gst.State.NULL)
                                conversion = "videoconvert ! video/x-raw,format=GRAY8"
                                pipeline_str = (f"rtspsrc location={deviceID} protocols=udp ! rtph264depay ! avdec_h264 ! {conversion} ! appsink name=appsink")

                                # Create GStreamer pipeline
                                self.pipeline = Gst.parse_launch(pipeline_str)
                                self.start_timestamp = time.time()

                                # Start the pipeline
                                self.pipeline.set_state(Gst.State.PLAYING)
                                device = self.pipeline.get_by_name("appsink")
                                self.frame_shape = (height, width)
                        
                        elif video_format == 'GRAY8':
                            self.frame_shape = (height, width)  # Grayscale
                            
                        else:
                            log.error(f"Unsupported video format: {video_format}.")
                    else:
                        log.error("Could not determine frame shape.")
                else:
                    log.error("Could not obtain frame.")

        return device


    def run(self):
        """ Capture frames.
        """
        
        # Init the video device
        device = self.initVideoDevice()

        # For video devices only (not files)

        # Create dir to save uncompressed images (for Contrails)
        stationID = str(self.config.stationID)
        date_string = time.strftime("%Y%m%d_%H%M%S", time.gmtime(time.time()))
        dirname = f"UC_{stationID}_"+ date_string
        dirname = os.path.join(self.config.data_dir, self.config.contrails_dir, dirname)

        # Create the directory
        os.makedirs(dirname, exist_ok=True)


        if device is None:

            log.info('The video source could not be opened!')
            self.exit.set()
            return False


        # # Wait until the device is opened
        # device_opened = False
        # for i in range(20):
        #     time.sleep(1)
        #     if device.isOpened():
        #         device_opened = True
        #         break


        # # If the device could not be opened, stop capturing
        # if not device_opened:
        #     log.info('The video source could not be opened!')
        #     self.exit.set()
        #     return False

        # else:
        #     log.info('Video device opened!')


        # Keep track of the total number of frames
        total_frames = 0            



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

        # For video devices only (not files)
        # if self.video_file is None:
        #     # Record the start time in seconds since the Unix epoch
        #     start_timestamp = time.time()

        #     # Start the pipeline
        #     self.pipeline.set_state(Gst.State.PLAYING)

        
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

                while not self.exit.is_set():

                    log.info('Waiting for the video device to be reconnected...')

                    time.sleep(5)

                    # Reinit the video device
                    device = self.initVideoDevice()


                    if device is None:
                        print("The video device couldn't be connected! Retrying...")
                        continue


                    if self.exit.is_set():
                        break

                    # Read the frame
                    log.info("Reading frame...")
                    ret, _, _ = self.read(device)
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
            t_contrail = 0
            t_block = time.time()

            # Capture a block of 256 frames
            block_frames = 256
            
            log.info('Grabbing a new block of {:d} frames...'.format(block_frames))


            for i in range(block_frames):

                # Read the frame (keep track how long it took to grab it)
                t1_frame = time.time()
                ret, frame, frame_timestamp = self.read(device)
                t_frame = time.time() - t1_frame


                # If the video device was disconnected, wait for reconnection
                if (self.video_file is None) and (not ret):

                    log.info('Frame grabbing failed, video device is probably disconnected!')

                    wait_for_reconnect = True
                    break


                # If a video device is used, get the current time
                if self.video_file is None:

                    # If a video device is used, save a uncompressed frame every nth frames (for Contrails)
                    # if i % 64 == 0:   > img every 2.56s, 3.7GB per day @ 25 fps
                    # if i % 128 == 0:   > img every 5.12s, 1.9GB per day @ 25 fps
                    # if i == 0:   > img every 10.24s, 0.9GB per day @ 25 fps
                    t1_contrail = time.time()
                    if i % 128 == 0:
                        # Generate the name for the file
                        date_string = time.strftime("%Y%m%d_%H%M%S", time.gmtime(frame_timestamp))

                        # Calculate miliseconds
                        millis = int((frame_timestamp - floor(frame_timestamp))*1000)
                        
                        # Create the filename
                        filename = f"UC_{stationID}_"+ date_string + "_" + str(millis).zfill(3) + ".jpg"

                        img_path = os.path.join(dirname, filename)

                        # Save the image to disk
                        try:
                            self.save_image_and_log_time(filename, img_path, frame,i)
                        except:
                            log.error("Could not save {:s} to disk!".format(filename))
                    t_contrail = time.time() - t1_contrail

                # If a video file is used, compute the time using the time from the file timestamp
                else:
                    frame_timestamp = video_first_timestamp + total_frames/self.config.fps

                    # print("tot={:6d}, i={:3d}, fps={:.2f}, t={:.8f}".format(total_frames, i, self.config.fps, frame_timestamp))

                    
                # Set the time of the first frame
                if i == 0:
                    # Initialize last frame timestamp if it's not set
                    if not last_frame_timestamp:
                        last_frame_timestamp = frame_timestamp
                    
                    # Always set first frame timestamp in the beginning of the block
                    first_frame_timestamp = frame_timestamp

                    # Calculate elapsed time since frame capture to assess sink fill level
                    frame_age_seconds = time.time() - frame_timestamp
                    log.info(f"Frame is {frame_age_seconds:.3f}s old.")


                # If the end of the video file was reached, stop the capture
                if self.video_file is not None: 
                    if (frame is None) or (not device.isOpened()):

                        log.info("End of video file!")
                        log.debug("Video end status:")
                        log.debug("Frame:" + str(frame))
                        log.debug("Device open:" + str(device.isOpened()))

                        self.exit.set()
                        time.sleep(0.1)
                        break


                # Check if frame is dropped if it has been more than 1.5 frames than the last frame
                elif (frame_timestamp - last_frame_timestamp) >= self.time_for_drop:
                    
                    # Calculate the number of dropped frames
                    n_dropped = int((frame_timestamp - last_frame_timestamp)*self.config.fps)
                    self.dropped_frames += n_dropped

                    if self.config.report_dropped_frames:
                        log.info(f"{str(n_dropped)}/{str(self.dropped_frames)} frames dropped! Time for frame: {t_frame:.3f}, contrail: {t_contrail:.3f}, convert: {t_convert:.3f}, assignment: {t_assignment:.3f}")

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

        # Check if using GStreamer and release resources
        if hasattr(self, 'pipeline') and self.pipeline:
            try:
                self.pipeline.set_state(Gst.State.NULL)
                log.info('GStreamer Video device released!')
            except Exception as e:
                log.error(f'Error releasing GStreamer pipeline: {e}')

        # Check if using OpenCV and release resources
        if 'device' in locals() and device:
            try:
                if isinstance(device, cv2.VideoCapture):
                    device.release()
                    log.info('OpenCV Video device released!')
            except Exception as e:
                log.error(f'Error releasing OpenCV device: {e}')


    