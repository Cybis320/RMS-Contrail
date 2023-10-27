import RMS.ConfigReader as cr
from rms_external import rmsExternal  # Import the rmsExternal function from the appropriate module

# Directory path to the configuration
config_path = "/home/pi/source/RMS"
data_path = "/home/pi/RMS_data" # Replace with the actual config file name

# Load the configuration from the directory
config = cr.loadConfigFromDirectory(config_path, data_path)

# Define captured and archived night directories (replace with actual paths)
captured_night_dir = "/home/pi/RMS_data/CapturedFiles"
archived_night_dir = "/home/pi/RMS_data/ArchivedFiles"

# Call the rmsExternal function with the loaded configuration
rmsExternal(captured_night_dir, archived_night_dir, config)
