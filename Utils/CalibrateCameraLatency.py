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

# Usage: (sudo privilege and vRMS modules needed)
# sudo /home/pi/vRMS/bin/python /home/pi/source/RMS/Utils/CalibrateCameraLatency.py -c /home/pi/source/RMS/.config

from __future__ import print_function, division, absolute_import

import os
# Set GStreamer debug level. Use '2' for warnings in production environments.
os.environ['GST_DEBUG'] = '3'

import re
import time
import argparse
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
import RMS.ConfigReader as cr



class BufferedCapture(Process):
    """ Capture from device to buffer in memory.
    """
    
    running = False
    
    def __init__(self, config, video_file=None):
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
        
        self.config = config

        self.video_file = video_file

        # A frame will be considered dropped if it was late more then half a frame
        self.time_for_drop = 1.9/config.fps

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


        self.pipeline = None
        self.start_timestamp = 0
        self.pulse_timestamp = 0
        self.frame_shape = None
        self.records = []

        # Define the LED
        self.led = "/sys/class/leds/PWR"
        self.current_trigger = None
        

    def write_to_file(self, path, value):
        with open(path, 'w') as f:
            f.write(value)


    def save_image_to_disk(self, filename, img_path, img, i):
        try:
            cv2.imwrite(img_path, img)
            # print(f"Saving completed: i={i}: {filename}")
        except Exception as e:
            print(f"Error, could not save image to disk: {e}")


    def startCapture(self, cameraID=0):
        """ Start capture using specified camera.
        
        Arguments:
            cameraID: ID of video capturing device (ie. ID for /dev/video3 is 3). Default is 0.
            
        """
        
        self.cameraID = cameraID
        self.exit = Event()

        # Save current trigger
        with open(f"{self.led}/trigger", 'r') as f:
            self.current_trigger = f.read().strip()
        
        self.write_to_file(f"{self.led}/brightness", "0")
        
        self.start()
    

    def stopCapture(self):
        """ Stop capture.
        """
        
        self.exit.set()

        time.sleep(1)

        # Restore the LED trigger
        self.write_to_file(f"{self.led}/brightness", "1")

        self.write_to_file(f"{self.led}/trigger", self.current_trigger)

        if self.is_alive():
            print('Terminating capture...')
            self.terminate()

        return self.dropped_frames


    def device_isOpened(self, device):
        if device is None:
            return False
        try:
            if isinstance(device, cv2.VideoCapture):
                return device.isOpened()
            else:
                state = device.get_state(Gst.CLOCK_TIME_NONE).state
                if state == Gst.State.PLAYING:
                    return True
                else:
                    return False
        except Exception as e:
            print(f'Error checking device status: {e}')
            return False


    def read(self, device):
        '''
        Retrieve frames and timestamp.
        :param device: The video capture device or file.
        :return: tuple (ret, frame, timestamp) where ret is a boolean indicating success,
                 frame is the captured frame, and timestamp is the frame timestamp.
        '''
        ret, frame, timestamp, gst_timestamp_ns = False, None, None, None

        if self.video_file is not None:
            ret, frame = device.read()
            if ret:
                timestamp = None # assigned later
        
        else:
            if self.config.force_v4l2 or self.config.force_cv2:
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
                
        return ret, frame, timestamp, gst_timestamp_ns


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
        device_str = ("rtspsrc protocols=tcp tcp-timeout=5000000 retry=5 "
                    f"location=\"{device_url}\" !"
                    "rtph264depay ! h264parse ! avdec_h264")

        conversion = f"videoconvert ! video/x-raw,format={video_format}"
        pipeline_str = (f"{device_str} ! {conversion} ! "
                        "appsink max-buffers=25 drop=true sync=1 name=appsink")
        
        self.pipeline = Gst.parse_launch(pipeline_str)

        self.pipeline.set_state(Gst.State.PLAYING)
        self.start_timestamp = time.time() - self.total_latency
        print(f"Start time is {datetime.datetime.fromtimestamp(self.start_timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')}")

        return self.pipeline.get_by_name("appsink")


    def initVideoDevice(self):
        """ Initialize the video device. """

        device = None

        # use a file as the video source
        if self.video_file is not None:
            print('The video source could not be opened!')
            self.exit.set()
            return False

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
                            print("Camera IP ping successful!")
                            break

                        time.sleep(5)

                    if not ping_success:
                        print("Can't ping the camera IP!")
                        return None

                else:
                    return None


            # Init the video device
            print("Initializing the video device...")
            print("Device: " + str(self.config.deviceID))
            if self.config.force_v4l2:
                device = cv2.VideoCapture(self.config.deviceID, cv2.CAP_V4L2)
                device.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            
            elif not self.config.force_v4l2 and self.config.force_cv2:
                device = cv2.VideoCapture(self.config.deviceID)

            else:
                print("Initialize GStreamer Device: ")
                # Initialize GStreamer
                Gst.init(None)

                # Create and start a GStreamer pipeline
                device = self.create_gstream_device('BGR')

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
                            ret, frame, _, _ = self.read(device)

                            # If frame is grayscale, stop and restart the pipeline in GRAY8 format
                            if self.is_grayscale(frame):
                                # Stop, Create and restart a GStreamer pipeline
                                self.pipeline.set_state(Gst.State.NULL)
                                device = self.create_gstream_device('GRAY8')

                                self.frame_shape = (height, width)
                        
                        elif video_format == 'GRAY8':
                            self.frame_shape = (height, width)  # Grayscale
                            
                        else:
                            print(f"Unsupported video format: {video_format}.")
                    else:
                        print("Could not determine frame shape.")
                else:
                    print("Could not obtain frame.")

        return device


    def run(self, frame_batch=256):
        """ Capture frames.
        """
        
        # Init the video device
        device = self.initVideoDevice()

        # Create dir to save jpg images
        stationID = str(self.config.stationID)
        date_string = time.strftime("%Y%m%d_%H%M%S", time.gmtime(time.time()))
        dirname = f"JPG_{stationID}_"+ date_string
        dirname = os.path.join(self.config.data_dir, self.config.jpg_dir, dirname)

        # Create the directory
        os.makedirs(dirname, exist_ok=True)

        if device is None:
            print('The video source could not be opened!')
            self.exit.set()
            return False


        # Wait until the device is opened
        device_opened = False
        for i in range(20):
            time.sleep(1)
            if self.device_isOpened(device):
                device_opened = True
                break


        # If the device could not be opened, stop capturing
        if not device_opened:
            print('The video source could not be opened!')
            self.exit.set()
            return False

        else:
            print('Video device opened!')



        # For video devices only (not files), throw away the first 10 frames
        first_skipped_frames = 100
        for i in range(first_skipped_frames):
            self.read(device)


        # Capture a block of frames
        block_frames = frame_batch
        pulse_duration = 0.01
        pulse_on_frame = 5


        print('Grabbing a new block of {:d} frames...'.format(block_frames))

        for i in range(block_frames):

            ret, frame, frame_timestamp, pts_ns = self.read(device)
            
            if ret:
                # Wait a few frames before pulsing the LED
                if i == pulse_on_frame:
                    led_time_0 = time.time()
                    # Turn the LED on
                    self.write_to_file(f"{self.led}/brightness", "1")

                    # Wait for the duration of the LED pulse
                    start_time = time.perf_counter()
        
                    while time.perf_counter() - start_time < pulse_duration:
                        pass  # Busy waiting for the duration of the pulse
                    
                    # Turn the LED off
                    self.write_to_file(f"{self.led}/brightness", "0")
                    self.pulse_timestamp = led_time_0 + pulse_duration / 2

                # Generate the name for the file
                date_string = time.strftime("%Y%m%d_%H%M%S", time.gmtime(frame_timestamp))

                # Calculate miliseconds
                millis = int((frame_timestamp - floor(frame_timestamp))*1000)
                
                # Create the filename
                filename = f"{stationID}_"+ date_string + "_" + str(millis).zfill(3) + "_i:" + i + ".jpg"

                img_path = os.path.join(dirname, filename)

                # Save the image to disk
                try:
                    self.save_image_to_disk(filename, img_path, frame, i)
                except:
                    print("Could not save {:s} to disk!".format(filename))
                
                # Store the iteration data in a dictionary
                record = {
                    'iteration': i,
                    'timestamp': frame_timestamp,
                    'filename': filename,
                    'pts_ns': pts_ns
                }
                
                self.records.append(record)
        
        roi = self.select_roi(img_path)

        self.write_to_file(f"{self.led}/trigger", self.current_trigger)

        print('Releasing video device...')

        # Check if using GStreamer and release resources
        if hasattr(self, 'pipeline') and self.pipeline:
            try:
                self.pipeline.set_state(Gst.State.NULL)
                print('GStreamer Video device released!')
            except Exception as e:
                print(f'Error releasing GStreamer pipeline: {e}')

        # Check if using OpenCV and release resources
        if 'device' in locals() and device:
            try:
                if isinstance(device, cv2.VideoCapture):
                    device.release()
                    print('OpenCV Video device released!')
            except Exception as e:
                print(f'Error releasing OpenCV device: {e}')
        

    def select_roi(self, initial_img_path):
        # Load the image
        img = cv2.imread(initial_img_path)
        if img is None:
            print(f"Could not open or find the image: {initial_img_path}")
            return None

        # Select ROI
        r = cv2.selectROI("Image", img, fromCenter=False, showCrosshair=True)

        # Close the window
        cv2.destroyAllWindows()
        return r


    def is_led_on(self, img, roi, threshold=200):
        # Crop the ROI from the image
        roi_cropped = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        
        # Check if the average brightness in the ROI is above a certain threshold
        return np.mean(roi_cropped) > threshold


    def find_led_on_images(self, dirname, records):
        # Open the first image and select ROI
        initial_img_path = os.path.join(dirname, records[0]['filename'])
        roi = self.select_roi(initial_img_path)
        if roi is None:
            return []

        led_on_images = []
        for record in records:
            img_path = os.path.join(dirname, record['filename'])
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not open or find the image: {img_path}")
                continue
            
            if self.is_led_on(img, roi):
                led_on_images.append(record)
        
        # If only one image has the LED on, retrieve the record for that image
        if len(led_on_images) == 1:
            return led_on_images[0]
        
        return led_on_images
    

    def calculate_1d_position(self, roi, image_width, image_height):
        # Calculate the center of the ROI
        center_x = roi[0] + roi[2] / 2
        center_y = roi[1] + roi[3] / 2
        
        # Calculate the 1D position as if the image is a linear array
        # scanned row by row from left to right
        position_1d = center_y * image_width + center_x
        
        # Calculate the relative position in the 1D array
        total_pixels = image_width * image_height
        rp_1d = position_1d / total_pixels
        
        return rp_1d


# Usage
camera_recorder = CameraRecorder()
dirname = 'path_to_your_images_directory'
records = 'your_records_list'
led_on_image = camera_recorder.find_led_on_images(dirname, records)



if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Starting capture.
        """)

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")


    # Parse the command line arguments
    cml_args = arg_parser.parse_args()
    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('.'))

    # Initialize buffered capture
    bc = BufferedCapture(config)

    # Start buffered capture
    bc.startCapture()
