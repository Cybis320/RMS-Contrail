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
pulse_duration = 0.0001

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
            while True:
                # Get current time
                now = datetime.now()

                # Round to the next minute
                next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
                
                # Calculate next top-of-minute minus 30.1ms for Laurel timer latency  
                next_minute_signal = next_minute - timedelta(milliseconds=30.1)

                # Initialize start signal time
                signal_time = time.perf_counter() + (next_minute_signal - datetime.now()).total_seconds()

                print("Waiting to send next start signal...")

                # Continiously update start signal time
                while time.perf_counter() < signal_time:
                    signal_time = time.perf_counter() + (next_minute_signal - datetime.now()).total_seconds()
                    print(f"\r{signal_time:.6f}", end="", flush=True)

                # Send start signal. Pulse duration is not critical 
                send_signal(signal_pin=signal_pin_A, pulse_duration=10*pulse_duration)
                print("\nSignal A!")

                # Calculate next stop time leaving enough time for pulse duration and timer recycle
                # Timer will not display accurate time during that interval
                next_minute_signal += timedelta(seconds=60 - 100 * pulse_duration)
                
                print("Waiting to send next stop signal...")

                # Initialize stop signal time
                signal_time = time.perf_counter() + (next_minute_signal - datetime.now()).total_seconds()

                # Continiously update stop signal time
                while time.perf_counter() < signal_time:
                    signal_time = time.perf_counter() + (next_minute_signal - datetime.now()).total_seconds()
                    print(f"\r{signal_time:.6f}", end="", flush=True)

                # Send stop signal. Pulse duration should has short as possible
                send_signal(signal_pin=signal_pin_B, pulse_duration=pulse_duration)
                print("\nSignal B!")
    finally:
        GPIO.cleanup()


if __name__ == "__main__":
    main()