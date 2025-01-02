#!/bin/bash
#
# bash script to set IMX291 cameras up from scratch using config files
#

# Find all .config files in subdirectories of ~/source/Stations/
config_files=$(find ~/source/Stations/ -name ".config")

# Check if any config files were found
if [ -z "$config_files" ]; then
    echo "No .config files found in ~/source/Stations/ subdirectories"
    exit 1
fi

# Function to process each camera
setup_camera() {
    local config_file="$1"
    echo "Processing camera with config file: $config_file"
    echo ""
    echo "This script will set your camera to the recommended settings"
    echo "for brightness, video style, gain, and so on. "
    echo ""
    echo "NB: The script requires that your camera is -already- set to the "
    echo "right IP address and that this address has been added to the RMS .config file."
    echo ""
    
    # Get camera IP for this config
    currip=$(python -m Utils.CameraControl --config "$config_file" GetIP)
    echo "Camera Address is $currip"
    
    # a few miscellaneous things
    python -m Utils.CameraControl --config "$config_file" SetOSD off
    python -m Utils.CameraControl --config "$config_file" SetColor 100,50,50,50,0,0
    python -m Utils.CameraControl --config "$config_file" SetAutoReboot Everyday,15
    python -m Utils.CameraControl --config "$config_file" CameraTime set
    
    # disable phone-home remote connectivity
    python -m Utils.CameraControl --config "$config_file" CloudConnection off
    
    # set the Video Encoder parameters
    python -m Utils.CameraControl --config "$config_file" SetParam General VideoFormat PAL
    python -m Utils.CameraControl --config "$config_file" SetParam Encode Video Compression H.264
    python -m Utils.CameraControl --config "$config_file" SetParam Encode Video Resolution 1080P
    python -m Utils.CameraControl --config "$config_file" SetParam Encode Video BitRateControl VBR
    python -m Utils.CameraControl --config "$config_file" SetParam Encode Video FPS 25
    python -m Utils.CameraControl --config "$config_file" SetParam Encode Video Quality 6
    python -m Utils.CameraControl --config "$config_file" SetParam Encode AudioEnable 0
    python -m Utils.CameraControl --config "$config_file" SetParam Encode VideoEnable 1
    python -m Utils.CameraControl --config "$config_file" SetParam Encode SecondStream 0
    
    # camera parameters
    python -m Utils.CameraControl --config "$config_file" SetParam Camera ClearFog enable 0
    python -m Utils.CameraControl --config "$config_file" SetParam Camera ClearFog level 50
    python -m Utils.CameraControl --config "$config_file" SetParam Camera Style type1
    python -m Utils.CameraControl --config "$config_file" SetParam Camera AeSensitivity 1
    python -m Utils.CameraControl --config "$config_file" SetParam Camera ApertureMode 0
    python -m Utils.CameraControl --config "$config_file" SetParam Camera BLCMode 0
    python -m Utils.CameraControl --config "$config_file" SetParam Camera DayNightColor 2
    python -m Utils.CameraControl --config "$config_file" SetParam Camera Day_nfLevel 0
    python -m Utils.CameraControl --config "$config_file" SetParam Camera DncThr 50
    python -m Utils.CameraControl --config "$config_file" SetParam Camera ElecLevel 100
    python -m Utils.CameraControl --config "$config_file" SetParam Camera EsShutter 0
    python -m Utils.CameraControl --config "$config_file" SetParam Camera ExposureParam LeastTime 40000
    python -m Utils.CameraControl --config "$config_file" SetParam Camera ExposureParam Level 0
    python -m Utils.CameraControl --config "$config_file" SetParam Camera ExposureParam MostTime 40000
    python -m Utils.CameraControl --config "$config_file" SetParam Camera GainParam AutoGain 1
    python -m Utils.CameraControl --config "$config_file" SetParam Camera GainParam Gain 60
    python -m Utils.CameraControl --config "$config_file" SetParam Camera BroadTrends AutoGain 0
    python -m Utils.CameraControl --config "$config_file" SetParam Camera BroadTrends Gain 50
    python -m Utils.CameraControl --config "$config_file" SetParam Camera IRCUTMode 0
    python -m Utils.CameraControl --config "$config_file" SetParam Camera IrcutSwap 0
    python -m Utils.CameraControl --config "$config_file" SetParam Camera Night_nfLevel 0
    python -m Utils.CameraControl --config "$config_file" SetParam Camera RejectFlicker 0
    python -m Utils.CameraControl --config "$config_file" SetParam Camera WhiteBalance 2
    python -m Utils.CameraControl --config "$config_file" SetParam Camera PictureFlip 0
    python -m Utils.CameraControl --config "$config_file" SetParam Camera PictureMirror 0
    
    # network parameters
    python -m Utils.CameraControl --config "$config_file" SetParam Network TransferPlan Fluency
    
    echo "Finished processing camera with config: $config_file"
    echo "----------------------------------------"
    echo ""
}

# Count total number of config files
total_configs=$(echo "$config_files" | wc -l)
echo "Found $total_configs camera configuration files"
echo ""
echo "Options:"
echo "  y - Process this camera"
echo "  n - Skip this camera"
echo "  a - Process all remaining cameras"
echo "  q - Quit processing"
echo ""

# Process each config file
process_all=false
for config_file in $config_files; do
    if [ "$process_all" = false ]; then
        echo "Found config file: $config_file"
        read -p "Process this camera? (y/n/a/q): " response
        case $response in
            [Aa]* ) process_all=true; setup_camera "$config_file";;
            [Yy]* ) setup_camera "$config_file";;
            [Nn]* ) echo "Skipping this camera";;
            [Qq]* ) echo "Quitting..."; exit 0;;
            * ) echo "Invalid response. Skipping this camera";;
        esac
    else
        setup_camera "$config_file"
    fi
done

echo "All cameras have been processed."