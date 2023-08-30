import os
import time
import logging
from datetime import datetime
import subprocess
from RMS.CaptureDurationDay import captureDuration


def rmsExternal(captured_night_dir, archived_night_dir, config):
    # Initialize the logger
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("logger")
    log.info('External script started')

    # Create lock file to avoid RMS rebooting the system
    lockfile = os.path.join(config.data_dir, config.reboot_lock_file)
    with open(lockfile, 'w') as _:
        pass

    # Create Contrail_data/CapturesFiles directory if not exists
    contrail_data_dir = os.path.expanduser('~/Contrail_data/CapturesFiles/')
    os.makedirs(contrail_data_dir, exist_ok=True)

    # Create directory with specified format
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3] # Removing last 3 digits to get microseconds
    new_dir_name = f"{config.stationID}_{timestamp}"
    new_dir_path = os.path.join(contrail_data_dir, new_dir_name)
    os.makedirs(new_dir_path)
    
    # Set Camera Parameters for daylight operation
    #subprocess.run(["./Scripts/RMS_SetCameraParams_Day.sh"])
    print("External Script Test")
    
    # Remove lock file so RMS is authorized to reboot, if needed
    os.remove(lockfile)

    log.info('External script finished')