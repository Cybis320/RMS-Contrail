#!/bin/bash

# This script is used for updating the RMS code from GitHub

# WARNING: The update might fail when new dependencies (libraires)
#  are introduced! Further steps might have to be undertaken.


RMSSOURCEDIR=~/source/RMS

RMSBACKUPDIR=~/.rms_backup

# File for indicating that the update is in progress
UPDATEINPROGRESSFILE=$RMSBACKUPDIR/update_in_progress

echo "Updating RMS code..."

# Make the backup directory
mkdir $RMSBACKUPDIR

# Check if the update was interrupted while it was in progress
UPDATEINPROGRESS="0"
if [ -f $UPDATEINPROGRESSFILE ]; then
	echo "Reading update in progress file..."
	UPDATEINPROGRESS=$(cat $UPDATEINPROGRESSFILE)
	echo "Update interuption status: $UPDATEINPROGRESS"
fi

# If an argument (any) is given, then the config and mask won't be backed up
# Also, don't back up the files if the update script was interrupted the last time
if [ $# -eq 0 ] && [ "$UPDATEINPROGRESS" = "0" ]; then
    
    echo "Backing up the config and mask..."

    # Back up the config and the mask
    cp $RMSSOURCEDIR/.config $RMSBACKUPDIR/.
    cp $RMSSOURCEDIR/mask.bmp $RMSBACKUPDIR/.
fi


cd $RMSSOURCEDIR

# Activate the virtual environment
source ~/vRMS/bin/activate

# Remove the build dir
echo "Removing the build directory..."
rm -r build

# Perform cleanup before installations
echo "Running pyclean for thorough cleanup..."
pyclean . -v --debris all

# Set the flag indicating that the RMS dir is reset
echo "1" > $UPDATEINPROGRESSFILE

# Stash the cahnges
git stash

# Pull new code from github
git pull


### Install potentially missing libraries ###

# Function to check if a package is installed
isInstalled() {
    dpkg -s "$1" >/dev/null 2>&1
}

# Function to prompt for sudo password with timeout
sudoWithTimeout() {
    local timeout_duration=30  # Timeout in seconds
    local attempts=3
    local prompt="[sudo] password for $USER: "
    
    for ((i=1; i<=attempts; i++)); do
        # Use read with timeout to get the password securely
        read -s -t "$timeout_duration" -p "$prompt" password
        echo  # Move to a new line after password input

        # Check if password is empty (timeout or Ctrl+D)
        if [[ -z "$password" ]]; then
            echo "Password entry timed out."
            return 1
        fi

        # Validate the password
        if echo "$password" | sudo -S true 2>/dev/null; then
            # Keep sudo token alive in background
            (while true; do sudo -v; sleep 50; done) &
            KEEP_SUDO_PID=$!
            return 0
        else
            if [ $i -lt $attempts ]; then
                echo "Sorry, try again."
            else
                echo "sudo: 3 incorrect password attempts"
                return 1
            fi
        fi
    done
}

# List of packages to check/install
packages=(
    "gobject-introspection"
    "libgirepository1.0-dev"
    "gstreamer1.0-libav"
    "gstreamer1.0-plugins-bad"
)

# Check if any package is missing
missing_packages=()
for package in "${packages[@]}"; do
    if ! isInstalled "$package"; then
        missing_packages+=("$package")
    fi
done

# If all packages are installed, inform and continue
if [ ${#missing_packages[@]} -eq 0 ]; then
    echo "All required packages are already installed."
else
    # Some packages are missing, so we need to update and install
    echo "The following packages need to be installed: ${missing_packages[*]}"

    # Prompt for sudo password with timeout
    if ! sudoWithTimeout; then
        echo "Password entry timed out or was incorrect. Skipping package installation."
    else
        # Password entered successfully, proceed with update and install
        sudo apt-get update

        # Install missing packages
        for package in "${missing_packages[@]}"; do
            echo "Installing $package..."
            sudo apt-get install -y "$package"
        done

        echo "All required packages have been installed."
        
        # Kill the background sudo-keeping process
        kill $KEEP_SUDO_PID 2>/dev/null
    fi
fi

### ###




# make sure the correct requirements are installed
pip install -r requirements.txt

# Run the python setup
python setup.py install

# Create a template file from the source config and copy the user config and mask files back
if [ $# -eq 0 ]; then
    # Rename the existing source .config file to .configTemplate
    mv $RMSSOURCEDIR/.config $RMSSOURCEDIR/.configTemplate

    # Copy the user config and mask files back
    cp $RMSBACKUPDIR/.config $RMSSOURCEDIR/.
    cp $RMSBACKUPDIR/mask.bmp $RMSSOURCEDIR/.
fi

# Set the flag that the update is not in progress
echo "0" > $UPDATEINPROGRESSFILE


echo "Update finished! Update exiting in 5 seconds..."
sleep 5
