import time
import re
import ephem
import logging
import Utils.CameraControl as cc
from RMS.Misc import RmsDateTime
from enum import Enum
from datetime import datetime, timedelta

# Get the logger from the main module
log = logging.getLogger("logger")

class CameraMode(Enum):
    DAY = "day"
    TWILIGHT = "twilight"
    NIGHT = "night"


def getSunAltitude(observer, sun):
    """Calculate current solar altitude in degrees."""
    sun.compute(observer)
    return float(sun.alt) * 180 / ephem.pi


def isTwilightDusk(observer, sun):
    """Determine if we're in dusk twilight (true) or dawn twilight (false)."""
    observer.horizon = '-18'
    next_setting = observer.next_setting(sun)
    print(f"next_setting: {next_setting}")
    observer.horizon = '-6'
    next_rising = observer.next_rising(sun)
    print(f"next_rissing: {next_rising}")

    return next_setting < next_rising


def determineCameraMode(sun_altitude):
    """Determine camera mode based on solar altitude."""
    if sun_altitude > -5.8:
        return CameraMode.DAY
    elif sun_altitude > -17.8:
        return CameraMode.TWILIGHT
    else:
        return CameraMode.NIGHT


def getNextTransitionTime(observer, sun, current_altitude):
    """Calculate the time until the next altitude transition with overshoot protection."""
    if current_altitude > -6:  # Currently in daylight
        observer.horizon = '-6'
        next_time = observer.next_setting(sun)
        
    elif current_altitude > -18:  # Currently in twilight
        # Check if we're in dusk or dawn twilight
        if isTwilightDusk(observer, sun):
            observer.horizon = '-18'  # Next transition will be to night
            next_time = observer.next_setting(sun)
            log.info("In dusk twilight - next transition will be to nighttime mode")
        else:
            observer.horizon = '-6'  # Next transition will be to day
            next_time = observer.next_rising(sun)
            log.info("In dawn twilight - next transition will be to daytime mode")
            
    else:  # Currently in night
        observer.horizon = '-18'
        next_time = observer.next_rising(sun)
    
    # Add overshoot protection
    next_time = ephem.Date(next_time + ephem.minute * 2)
    return next_time.datetime()


def switchCameraMode(config, mode, daytime_mode, current_altitude):
    """Switch camera mode with proper error handling and logging.
    
    Args:
        config: Camera configuration
        mode: Target camera mode (DAY, TWILIGHT, or NIGHT)
        daytime_mode: Shared value for day/night status
        current_altitude: Current sun altitude for logging
        
    Returns:
        bool: True if switch successful, False otherwise
    """
    try:
        device_id = re.findall(r"[0-9]+(?:\.[0-9]+){3}", config.deviceID)[0]
        time.sleep(config.postprocess_delay)
        
        if mode == CameraMode.DAY:
            log.info(f'Switching {device_id} to daytime mode (Sun altitude: {current_altitude:.2f}°)')
            daytime_mode.value = True
            cc.cameraControlV2(config, 'SwitchDayTime')
            
        elif mode == CameraMode.TWILIGHT:
            log.info(f'Switching {device_id} to twilight mode (Sun altitude: {current_altitude:.2f}°)')
            daytime_mode.value = False
            cc.cameraControlV2(config, 'SwitchTwilight')
            
        else:  # NIGHT mode
            log.info(f'Switching {device_id} to nighttime mode (Sun altitude: {current_altitude:.2f}°)')
            daytime_mode.value = False
            cc.cameraControlV2(config, 'SwitchNightTime')
            
        log.info(f'Successfully switched {device_id} to {mode.name} mode')
        return True
        
    except cc.DVRIPCommandError as e:
        log.error(f'Command error switching {device_id} to {mode.name} mode: {str(e)}')
        return False
    except cc.DVRIPNotSupportedError as e:
        log.error(f'Command not supported switching {device_id} to {mode.name} mode: {str(e)}')
        return False
    except cc.DVRIPConnectionError as e:
        log.error(f'Connection error switching {device_id} to {mode.name} mode: {str(e)}')
        return False
    except Exception as e:
        log.error(f'Unexpected error switching {device_id} to {mode.name} mode: {str(e)}')
        return False

# Function to switch between day and night modes
def cameraModeSwitcher(config, daytime_mode):
    """ Wait and switch between day, twilight, and night camera modes based on current time.
    
    Arguments:
        config: [Config] config object for determining location and camera
        daytime_mode: [multiprocessing.Value] shared boolean variable to communicate mode switch with other processes
                            True = Day time, False = Night time
    """

    last_mode = None  # Track the last mode to prevent unnecessary switches
    retry_count = 0   # Track consecutive failures for exponential backoff

    while True:
        try:
            # Initialize observer
            o = ephem.Observer()
            o.lat = str(config.latitude)
            o.long = str(config.longitude)
            o.elevation = config.elevation
            
            # Set the current time with microsecond precision
            current_time = RmsDateTime.utcnow()
            o.date = current_time
            
            # Calculate current sun position
            s = ephem.Sun()
            current_altitude = getSunAltitude(o, s)
            
            # Determine the current mode
            current_mode = determineCameraMode(current_altitude)
            
            # Only switch modes if there's been a change
            if current_mode != last_mode:
                log.info(f'Mode change detected: {last_mode.name if last_mode else "None"} -> {current_mode.name}')
                
                # Attempt to switch camera mode
                if switchCameraMode(config, current_mode, daytime_mode, current_altitude):
                    last_mode = current_mode
                    retry_count = 0  # Reset retry count on success
                else:
                    # If switch failed, implement exponential backoff
                    retry_count += 1
                    wait_time = min(300, 30 * (2 ** (retry_count - 1)))  # Max 5 minutes
                    log.warning(f'Switch failed, waiting {wait_time} seconds before retry')
                    time.sleep(wait_time)
                    continue
            
            # Get the next transition time
            next_transition = getNextTransitionTime(o, s, current_altitude)
            
            # Calculate time to wait until next transition
            time_to_wait = (next_transition - current_time).total_seconds()
            log.info(f'Next transition at {next_transition} (in {time_to_wait/3600:.2f} hours)')
            
            # Sleep until next transition, but wake up periodically to check status
            while time_to_wait > 0:
                sleep_interval = min(900, time_to_wait)  # Wake up at least every 15 minutes
                time.sleep(sleep_interval)
                time_to_wait -= sleep_interval
                
                # Recalculate current mode to catch any manual changes or drift
                o.date = RmsDateTime.utcnow()
                current_altitude = getSunAltitude(o, s)
                check_mode = determineCameraMode(current_altitude)
                if check_mode != current_mode:
                    log.warning(f'Mode changed during wait period: {current_mode.name} -> {check_mode.name}')
                    break
            
        except Exception as e:
            log.error(f'Error in camera mode switcher: {str(e)}', exc_info=True)
            time.sleep(60)  # Wait a minute before retrying


    ### For testing ###

    # wait_interval = 5*60
    
    # while True:

    #     if not daytime_mode.value:
    #         log.info(f'Switching to day time mode')
    #         daytime_mode.value = True
    #         cc.cameraControlV2(config, 'SwitchDayTime')

    #     else:
    #         log.info(f'Switching to night time mode')
    #         daytime_mode.value = False
    #         cc.cameraControlV2(config, 'SwitchNightTime')

    #     time.sleep(wait_interval)


if __name__ == "__main__":
    
    import RMS.ConfigReader as cr
    import os

    config = cr.loadConfigFromDirectory('.', os.path.abspath('.'))

    config.latitude = 35.0572167
    config.longitude = -106.6837667
    config.elevation = 1520

    
    
    