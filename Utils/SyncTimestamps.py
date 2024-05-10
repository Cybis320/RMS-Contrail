import os
import time
from datetime import datetime, timedelta
import argparse
import logging
import csv

import RMS.ConfigReader as cr
from RMS.Logger import initLogging


# Define the LED
pwr_led = "/sys/class/leds/PWR"
act_led = "/sys/class/leds/ACT"


class LEDController:
    def __init__(self, pwr_led, act_led):
        self.pwr_led = pwr_led
        self.act_led = act_led
        self.pwr_current_trigger = self.saveCurrentTrigger(pwr_led)
        self.act_current_trigger = self.saveCurrentTrigger(act_led)


    def writeToFile(self, path, value):
        '''
        Control LED by writing to 'file'
        '''
        try:
            with open(path, 'w') as f:
                f.write(value + "\n")
        except OSError as e:
            print(f"Error writing to {path}: {e}")


    def saveCurrentTrigger(self, led):
        '''
        Save the existing LED trigger to be restored when exiting program
        '''

        with open(f"{led}/trigger", 'r') as f:
            trigger_data = f.read()
            # Extract the current trigger name, which is enclosed in square brackets
            start = trigger_data.find('[')
            end = trigger_data.find(']', start)
            if start != -1 and end != -1:
                current_trigger = trigger_data[start + 1:end]  # +1 and end without +1 to exclude the brackets
            else:
                current_trigger = None
                print("No current trigger found.")

            print(current_trigger)
        return current_trigger


    def turnLedOff(self, led):
        self.writeToFile(f"{led}/brightness", "0")


    def turnLedOn(self, led):
        self.writeToFile(f"{led}/brightness", "1")


    def flashRelativeToSystemTime(self, period=1, flash_duration=1, log_to_csv=True):
        # Setup CSV file for logging
        csv_file_path = 'led_log.csv'
        file_exists = os.path.isfile(csv_file_path)

        with open(csv_file_path, 'a', newline='') as csvfile:
            fieldnames = ['timestamp', 'event']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()  # file doesn't exist yet, write a header

            # Initial setup: Get the current real-world time and the corresponding perf_counter value
            base_real_time_0 = datetime.now()
            base_perf_counter = time.perf_counter()
            base_real_time_1 = datetime.now()

            delta_base_time = (base_real_time_1 - base_real_time_0)/2
            print(f"Delta base-time: {delta_base_time*1e6} µs")
            base_real_time = base_real_time_0 + delta_base_time

            # Calculate the initial delay to the next second mark, adjusted for the first signal timing
            next_second = (base_real_time + timedelta(seconds=1)).replace(microsecond=0)
            initial_delay_until_next_second_signal = (next_second - base_real_time).total_seconds()

            # Recalculate if too close to next event
            if initial_delay_until_next_second_signal < 0.1:
                next_second = (base_real_time + timedelta(seconds=2)).replace(microsecond=0)

            while True:
                # Calculate the next LED start and stop signal timings relative to the elapsed time
                next_led_on_signal_time = next_second - timedelta(milliseconds=0)
                next_led_off_signal_time = next_led_on_signal_time + timedelta(milliseconds=flash_duration)

                while ((base_real_time + timedelta(seconds=time.perf_counter() - base_perf_counter))
                        < next_led_on_signal_time):
                    pass
                self.turnLedOn(self.pwr_led)
                led_on_perf_counter = time.perf_counter()
                while ((base_real_time + timedelta(seconds=time.perf_counter() - base_perf_counter))
                        < next_led_off_signal_time):
                    pass
                self.turnLedOff(self.pwr_led)
                led_on_perf_time = base_real_time + timedelta(seconds=led_on_perf_counter - base_perf_counter)

                # Log to CSV if enabled
                if log_to_csv:
                    writer.writerow({'timestamp': led_on_perf_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
                                     'event': 'LED ON'})

                print(f"LED ON Time: {led_on_perf_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")
                next_second += timedelta(seconds=period)


    def flashRelativeToFps(self, relative_period=25, flash_duration=1, config=None, log_to_csv=True):

        if config is not None:
            fps = config.fps
        else:
            fps = 25

        period = relative_period/fps
        self.flashRelativeToSystemTime(period=period, flash_duration=flash_duration, log_to_csv=log_to_csv)


    def run(self, session_duration=True):
        # Turn LED off
        self.turnLedOff(self.pwr_led)
        self.turnLedOff(self.act_led)

        while session_duration:
            pass


    def stop(self):
        # Restore the LED trigger
        self.writeToFile(f"{self.pwr_led}/trigger", pwr_current_trigger)
        self.writeToFile(f"{self.act_led}/trigger", act_current_trigger)




if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""Produces a sequence of LED pulses and log their
                                         timestamps""")

    # Add a mutually exclusive for the parser (the arguments in the group can't be given at the same)
    arg_group = arg_parser.add_mutually_exclusive_group()

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str,
                            help="Path to a config file which will be used instead of the default one.")
    
    arg_parser.add_argument('-m', '--mode', nargs=1, metavar='PULSE_MODE',
                            help="LED Pulse mode.")
    
    # TODO: mode    0:  period absolute (e.g. 1 second)
    #               1:  period relative to config fps (e.g. 25/fps seconds) 
    #               2:  period relative to system time (e.g. top of every minute)
    #               3:  calibration pattern (based on config jpgs_interval)

    arg_group.add_argument('-d', '--duration', metavar='DURATION_HOURS', help="""Specify the duration of the
                           seesion""")

    arg_group.add_argument('-p', '--pulse', metavar='PULSE_DURATION_MILLISECONDS', help="""Specify the
                           duration of each LED flash in milliseconds""")

    arg_group.add_argument('-o', '--output', metavar='FILE_PATH', help="""Path of the output csv file""")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('.'))

    # Initialize the logger
    initLogging(config)

    # Get the logger handle
    log = logging.getLogger("logger")




