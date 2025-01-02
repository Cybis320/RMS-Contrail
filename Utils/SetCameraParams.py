#!/usr/bin/env python3
"""
Python script to set up IMX291 cameras from scratch using config files.
Converted from original bash script.
"""

import os
import glob
import re
import time
import subprocess
from typing import List, Optional
import RMS.ConfigReader as cr
from Utils.CameraControl import cameraControl

def findConfigFiles() -> List[str]:
    """Find .config files. If Station configs exist, use those (multicam setup).
    Otherwise check RMS root directory (single cam setup).
    """
    config_files = []
    
    # First check Stations directory for multicam setup
    stations_path = os.path.expanduser("~/source/Stations/")
    if os.path.exists(stations_path):
        station_configs = glob.glob(os.path.join(stations_path, "**/", ".config"), recursive=True)
        config_files = [f for f in station_configs if os.path.isfile(f)]
        
        if config_files:
            print("Found Station config files (multicam setup):")
            for config in config_files:
                print("  {}".format(config))
            return config_files
    
    # Only check RMS root if no Station configs found
    try:
        from RMS.Misc import getRMSRootDir
        rms_root = getRMSRootDir()
        root_config = os.path.join(rms_root, ".config")
        if os.path.isfile(root_config):
            config_files.append(root_config)
            print("Found config in RMS root directory (single cam setup):")
            print("  {}".format(root_config))
    except Exception as e:
        print("Warning: Could not check RMS root directory: {}".format(str(e)))
    
    if not config_files:
        print("No .config files found in either:")
        print("  1. ~/source/Stations/* (multicam setup)")
        print("  2. RMS root directory (single cam setup)")
            
    return config_files

def waitForCamera(ip: str, timeout: int = 120) -> bool:
    """Wait for camera to respond to ping after reboot.
    
    Args:
        ip: Camera IP address
        timeout: Maximum time to wait in seconds
        
    Returns:
        bool: True if camera responds, False if timeout
    """
    start_time = time.time()
    print("Waiting for camera at {} to come back online...".format(ip))
    
    while time.time() - start_time < timeout:
        try:
            # Use ping command with 1 packet and 1 second timeout
            result = subprocess.run(['ping', '-c', '1', '-W', '1', ip],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
            if result.returncode == 0:
                print("Camera is responding!")
                # Add a little extra time for all services to start
                time.sleep(5)
                return True
        except subprocess.SubprocessError:
            pass
            
        # Wait a bit before trying again
        time.sleep(2)
        print(".", end="", flush=True)
        
    print("\nTimeout waiting for camera to respond")
    return False

def checkVideoFormat(camera_ip: str) -> bool:
    """Check if camera is in PAL mode, return True if reboot needed."""
    try:
        # Get current format - getGeneralParams returns (autoreboot_settings, location_settings)
        _, location_settings = cameraControl(camera_ip, 'GetAutoReboot')
        current_format = location_settings['VideoFormat']
        print("Current video format: {}".format(current_format))
        
        if current_format != 'PAL':
            print("Camera is in {} mode, switching to PAL...".format(current_format))
            # Set to PAL
            cameraControl(camera_ip, 'SetParam', ['General', 'VideoFormat', 'PAL'])
            return True
            
        print("Camera already in PAL mode")
        return False
        
    except Exception as e:
        print("Error checking video format: {}".format(str(e)))
        return False

def setupCamera(config_file: str) -> None:
    """Process a single camera with the given config file."""
    print("Processing camera with config file: {}".format(config_file))
    print("")
    print("This script will set your camera to the recommended settings")
    print("for brightness, video style, gain, and so on.")
    print("")
    print("NB: The script requires that your camera is -already- set to the")
    print("right IP address and that this address has been added to the RMS .config file.")
    print("")

    try:
        # Load the config file
        config = cr.loadConfigFromDirectory([config_file], os.path.dirname(config_file))
        if config is None:
            print("Error: Could not parse config: {}".format(config_file))
            return
            
        # Extract IP from config.deviceID
        camera_ip = re.findall(r"[0-9]+(?:\.[0-9]+){3}", config.deviceID)[0]
        print("Camera IP: {}".format(camera_ip))
        
        # Check video format and reboot if needed
        if checkVideoFormat(camera_ip):
            print("Rebooting camera to apply PAL mode...")
            cameraControl(camera_ip, 'reboot')
            
            # Initial wait for camera to start rebooting
            time.sleep(5)
            
            # Wait for camera to respond
            if not waitForCamera(camera_ip):
                print("Warning: Camera not responding after reboot")
                response = input("Do you want to continue anyway? (y/n): ")
                if response.lower() != 'y':
                    return
            
        # Set up miscellaneous settings
        cameraControl(camera_ip, 'SetOSD', ['off'])
        cameraControl(camera_ip, 'SetColor', ['100,50,50,50,0,0'])
        cameraControl(camera_ip, 'SetAutoReboot', ['Sunday,15'])
        cameraControl(camera_ip, 'CameraTime', ['set'])
        
        # Disable phone-home remote connectivity
        cameraControl(camera_ip, 'CloudConnection', ['off'])
        
        # Set the Video Encoder parameters
        encoder_params = [
            ('Encode', 'Video', 'Compression', 'H.264'),
            ('Encode', 'Video', 'Resolution', '1080P'),
            ('Encode', 'Video', 'BitRateControl', 'VBR'),
            ('Encode', 'Video', 'FPS', '25'),
            ('Encode', 'Video', 'Quality', '6'),
            ('Encode', 'AudioEnable', '0'),
            ('Encode', 'VideoEnable', '1'),
            ('Encode', 'SecondStream', '0')
        ]
        
        for param in encoder_params:
            cameraControl(camera_ip, 'SetParam', list(param))
        
        # Set camera parameters
        camera_params = [
            ('Camera', 'ClearFog', 'enable', '0'),
            ('Camera', 'ClearFog', 'level', '50'),
            ('Camera', 'Style', 'type1'),
            ('Camera', 'AeSensitivity', '1'),
            ('Camera', 'ApertureMode', '0'),
            ('Camera', 'BLCMode', '0'),
            ('Camera', 'DayNightColor', '2'),
            ('Camera', 'Day_nfLevel', '0'),
            ('Camera', 'DncThr', '50'),
            ('Camera', 'ElecLevel', '100'),
            ('Camera', 'EsShutter', '0'),
            ('Camera', 'ExposureParam', 'LeastTime', '40000'),
            ('Camera', 'ExposureParam', 'Level', '0'),
            ('Camera', 'ExposureParam', 'MostTime', '40000'),
            ('Camera', 'GainParam', 'AutoGain', '1'),
            ('Camera', 'GainParam', 'Gain', '60'),
            ('Camera', 'BroadTrends', 'AutoGain', '0'),
            ('Camera', 'BroadTrends', 'Gain', '50'),
            ('Camera', 'IRCUTMode', '0'),
            ('Camera', 'IrcutSwap', '0'),
            ('Camera', 'Night_nfLevel', '0'),
            ('Camera', 'RejectFlicker', '0'),
            ('Camera', 'WhiteBalance', '2'),
            ('Camera', 'PictureFlip', '0'),
            ('Camera', 'PictureMirror', '0')
        ]
        
        for param in camera_params:
            cameraControl(camera_ip, 'SetParam', list(param))
        
        # Set network parameters
        cameraControl(camera_ip, 'SetParam', ['Network', 'TransferPlan', 'Fluency'])
        
        print("Finished processing camera with config: {}".format(config_file))
        print("-" * 40)
        print("")
    
    except Exception as e:
        print("Error while configuring camera: {}".format(str(e)))

def rebootCameras(cameras_to_reboot):
    """Reboot a list of cameras and wait for them to come back online.
    
    Args:
        cameras_to_reboot: List of camera IPs to reboot
    """
    print("\nRebooting {} cameras...".format(len(cameras_to_reboot)))
    for camera_ip in cameras_to_reboot:
        print("\nRebooting camera at {}...".format(camera_ip))
        cameraControl(camera_ip, 'reboot')
        time.sleep(5)  # Wait between camera reboots
        
    print("\nWaiting for all cameras to come back online...")
    for camera_ip in cameras_to_reboot:
        if waitForCamera(camera_ip):
            print("Camera at {} is back online.".format(camera_ip))
        else:
            print("Warning: Camera at {} not responding after reboot.".format(camera_ip))

def main():
    """Main function to process all cameras."""
    config_files = findConfigFiles()
    
    if not config_files:
        print("No .config files found in either multicam or single cam locations")
        return 1
    
    total_configs = len(config_files)
    if total_configs > 1:
        print("\nFound {} camera configuration files.".format(total_configs))
    else:
        print("\nFound single camera configuration.")
    print("")
    print("Options:")
    print("  y - Process this camera")
    print("  n - Skip this camera")
    if total_configs > 1:
        print("  a - Process all remaining cameras")
    print("  q - Quit processing")
    print("")
    
    process_all = False
    cameras_to_reboot = []  # List to store camera IPs that were configured
    
    for config_file in config_files:
        if not process_all:
            print("Found config file: {}".format(config_file))
            while True:
                response = input("Process this camera? (y/n/a/q): ").lower()
                if response in ['y', 'n', 'a', 'q']:
                    break
                print("Invalid response. Please enter y, n, a, or q.")
            
            if response == 'q':
                if cameras_to_reboot:
                    print("\n{} cameras have been updated.".format(len(cameras_to_reboot)))
                    while True:
                        reboot = input("Would you like to reboot the updated cameras? (y/n): ").lower()
                        if reboot in ['y', 'n']:
                            break
                    if reboot == 'y':
                        rebootCameras(cameras_to_reboot)
                print("Quitting...")
                return 0
            elif response == 'a':
                process_all = True
                # Get camera IP and add to reboot list if setup successful
                config = cr.loadConfigFromDirectory([config_file], os.path.dirname(config_file))
                if config:
                    camera_ip = re.findall(r"[0-9]+(?:\.[0-9]+){3}", config.deviceID)[0]
                    setupCamera(config_file)
                    cameras_to_reboot.append(camera_ip)
            elif response == 'y':
                # Get camera IP and add to reboot list if setup successful
                config = cr.loadConfigFromDirectory([config_file], os.path.dirname(config_file))
                if config:
                    camera_ip = re.findall(r"[0-9]+(?:\.[0-9]+){3}", config.deviceID)[0]
                    setupCamera(config_file)
                    cameras_to_reboot.append(camera_ip)
            else:  # 'n'
                print("Skipping this camera")
        else:
            # Get camera IP and add to reboot list if setup successful
            config = cr.loadConfigFromDirectory([config_file], os.path.dirname(config_file))
            if config:
                camera_ip = re.findall(r"[0-9]+(?:\.[0-9]+){3}", config.deviceID)[0]
                setupCamera(config_file)
                cameras_to_reboot.append(camera_ip)
    
    if cameras_to_reboot:
        print("\nAll cameras configured. Performing final reboot of all cameras...")
        rebootCameras(cameras_to_reboot)
    
    print("\nAll cameras have been processed and rebooted.")
    return 0

if __name__ == "__main__":
    exit(main())