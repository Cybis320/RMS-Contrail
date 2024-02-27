import RPi.GPIO as GPIO
import time
from datetime import datetime, timedelta
import argparse

# Display seconds on a Laurel L timer
# Timer should be set for Mode: Stopwatch, Function: A to B
# Display Type: Time in Secs
# Timer precision can be adjusted as follow:
# Decimal Point: 11.1111 Multiplier: 10000 -> ss.####
# Decimal Point: 111.111 Multiplier:  1000 -> ss.###
# Trigger Slope A & B: Positive

# GPIO setup
signal_pin_A = 16
signal_pin_B = 20
pulse_duration = 0.2 / 1000

def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(signal_pin_A, GPIO.OUT)
    GPIO.setup(signal_pin_B, GPIO.OUT)


def send_signal(signal_pin=signal_pin_A, pulse_duration=pulse_duration):
    start_time = time.perf_counter()
    GPIO.output(signal_pin, GPIO.HIGH)
    
    while time.perf_counter() - start_time < pulse_duration:
        pass  # Busy waiting for the duration of the pulse

    GPIO.output(signal_pin, GPIO.LOW)


def send_signal_at_frequency(frequency):
    interval = 1 / frequency  # Calculate the time interval between signals
    next_time = time.perf_counter() + interval
    while True:
        now = time.perf_counter()
        if now >= next_time:
            send_signal()
            next_time += interval
        time.sleep(max(next_time - now, 0))  # Sleep for the remaining time until the next signal

def parse_arguments():
    parser = argparse.ArgumentParser(description='Send signals at a set frequency.')
    parser.add_argument('-c', '--frequency', type=float, help='Set the frequency for sending signals.')
    return parser.parse_args()

def main():
    args = parse_arguments()

    setup_gpio()
    try:
        if args.frequency:
            send_signal_at_frequency(args.frequency)
        else:
            # Initial setup: Get the current real-world time and the corresponding perf_counter value
            base_real_time = datetime.now()
            base_perf_counter = time.perf_counter()

            # Calculate the initial delay to the next minute mark, adjusted for the first signal timing
            next_minute = (base_real_time + timedelta(minutes=1)).replace(second=0, microsecond=0)
            initial_delay_until_next_signal = (next_minute - base_real_time).total_seconds()

            # Recalculate if too close to next minute
            if initial_delay_until_next_signal < 1:
                next_minute = (base_real_time + timedelta(minutes=2)).replace(second=0, microsecond=0)


            while True:
                
                # Calculate the next start and stop signal timings relative to the elapsed time
                next_start_signal_time = next_minute - timedelta(milliseconds=30.1)  # 30.1 ms before the top of the minute
                next_stop_signal_time = next_start_signal_time - timedelta(milliseconds=5) # 5 ms before the start signal
                

                # Wait for the next stop signal time
                print("Waiting to send next stop signal...")

                while (base_real_time + timedelta(seconds=time.perf_counter() - base_perf_counter)) < next_stop_signal_time:
                    pass

                # Send stop signal. Pulse duration should has short as possible
                send_signal(signal_pin=signal_pin_B, pulse_duration=pulse_duration)
                print("\nSignal B!")


                # Wait for the next start signal time
                print("Waiting to send next start signal...")

                while (base_real_time + timedelta(seconds=time.perf_counter() - base_perf_counter)) < next_start_signal_time:
                    pass

                # Send start signal. Pulse duration is not critical 
                send_signal(signal_pin=signal_pin_A, pulse_duration=10*pulse_duration)
                print("\nSignal A!")

                # Update the delay for the next cycle, ensuring it accounts for a full minute past the initial delay
                next_minute += timedelta(minutes=1)
    finally:
        GPIO.cleanup()


if __name__ == "__main__":
    main()
