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

# Remove the build dir
rm -r build


# Set the flag indicating that the RMS dir is reset
echo "1" > $UPDATEINPROGRESSFILE

# Stash the changes
git stash

########## TODO: get branch/fork from .config file #########

# Check if 'fork' remote exists and add if it doesn't
if ! git remote | grep -q 'fork'; then
  git remote add fork 'https://github.com/Cybis320/RMS-Contrail'
fi

# Pull new code from the forked repository
git pull fork master

# Activate the virtual environment
source ~/vRMS/bin/activate

# make sure the correct requirements are installed
pip install -r requirements.txt

# Run the python setup
python setup.py install

# Copy the config and mask files back
if [ $# -eq 0 ]; then
    
    # Copy the config and the mask back
    cp $RMSBACKUPDIR/.config $RMSSOURCEDIR/.
    cp $RMSBACKUPDIR/mask.bmp $RMSSOURCEDIR/.
fi

# Set the flag that the update is not in progress
echo "0" > $UPDATEINPROGRESSFILE


echo "Update finished! Update exiting in 5 seconds..."
sleep 5