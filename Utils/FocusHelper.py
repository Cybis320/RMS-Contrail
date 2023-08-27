""" Shows the Focus value of the camera. """

from __future__ import print_function, division, absolute_import

import cv2
import argparse

import RMS.ConfigReader as cr
import numpy as np

def get_device(config):
    """ Get the video device """
    vcap = cv2.VideoCapture(config.deviceID)

    return vcap

if __name__ == "__main__":

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Show live stream from the camera.
        """)

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str,
        help="Path to a config file which will be used instead of the default one.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    # Load the configuration file
    config = cr.loadConfigFromDirectory(cml_args.config, 'notused')

    # Open video device
    vcap = get_device(config)

    ret, frame = vcap.read()
    if not ret:
        print("Failed to grab frame for size detection")
        exit(1)
    
    # Compute frame size once
    height, width = frame.shape[:2]
    top_left_y = height // 4
    top_left_x = width // 4
    bottom_right_y = height - height // 4
    bottom_right_x = width - width // 4

    skip_frames = 10  # Skip every 10 frames
    frame_count = 0

    # Initialize focus window position and size
    focus_w, focus_h = 100, 100  # Initial width and height of the focus window
    focus_x, focus_y = (width // 4) - focus_w, (height // 4) - focus_h
    min_size, max_size = 50, 300  # Min and max size for focus window

    while True:
        ret, frame = vcap.read()
        frame_count += 1
        
        if not ret:
            print("Failed to grab frame")
            break
        
        if frame_count % skip_frames == 0:
            # Crop to central region
            cropped_frame = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            
            # Extract focus window
            focus_window = cropped_frame[focus_y:focus_y+focus_h, focus_x:focus_x+focus_w]
            
            # Convert to grayscale
            gray = cv2.cvtColor(focus_window, cv2.COLOR_BGR2GRAY)

            # Apply Laplacian filter
            laplacian = cv2.Laplacian(gray, cv2.CV_32F)

            # Compute the normalized focus measure (variance of Laplacian)
            num_pixels = focus_w * focus_h
            focus_measure = 100 * np.var(laplacian) / num_pixels

            # Display focus measure, instructions, and focus window on the cropped frame
            cv2.putText(cropped_frame, f"Focus: {focus_measure:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(cropped_frame, "Press 'q' to quit. Use +, -, and arrows to adjust.", (10, cropped_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.rectangle(cropped_frame, (focus_x, focus_y), (focus_x+focus_w, focus_y+focus_h), (0, 255, 0), 2)


            # Zoom 2x for display
            zoomed_frame = cv2.resize(cropped_frame, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            
            # Show the image
            cv2.imshow("Focus Measure", zoomed_frame)

            # Move the focus window based on keyboard input
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == 43:  # '+' key
                focus_w = min(focus_w + 10, max_size)
                focus_h = min(focus_h + 10, max_size)
            elif key == 45:  # '-' key
                focus_w = max(focus_w - 10, min_size)
                focus_h = max(focus_h - 10, min_size)
            elif key == 81:  # Left arrow key
                focus_x = max(focus_x - focus_w // 10, 0)
            elif key == 83:  # Right arrow key
                focus_x = min(focus_x + focus_w // 10, cropped_frame.shape[1] - focus_w)
            elif key == 82:  # Up arrow key
                focus_y = max(focus_y - focus_h // 10, 0)
            elif key == 84:  # Down arrow key
                focus_y = min(focus_y + focus_h // 10, cropped_frame.shape[0] - focus_h)

    vcap.release()
    cv2.destroyAllWindows()
