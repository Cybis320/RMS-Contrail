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

# Define the LED
pwr_led = "/sys/class/leds/PWR"
act_led = "/sys/class/leds/ACT"
pwr_current_trigger = None
act_current_trigger = None


def write_to_file(path, value):
    try:
        with open(path, 'w') as f:
            f.write(value + "\n")
    except OSError as e:
        print(f"Error writing to {path}: {e}")


def save_current_trigger(led):
    # Save current trigger
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

        print(current_trigger)  # To verify you're getting the correct current trigger
    return current_trigger
    

def turn_led_off(led):
    write_to_file(f"{led}/brightness", "0")


def turn_led_on(led):
    write_to_file(f"{led}/brightness", "1")


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
    pwr_current_trigger = save_current_trigger(pwr_led)
    act_current_trigger = save_current_trigger(act_led)
    
    turn_led_off(pwr_led)
    turn_led_off(act_led)

    try:
        if args.frequency:
            send_signal_at_frequency(args.frequency)
        else:
            # Initial setup: Get the current real-world time and the corresponding perf_counter value
            base_real_time = datetime.now()
            base_perf_counter = time.perf_counter()

            # Calculate the initial delay to the next minute mark, adjusted for the first signal timing
            next_minute = (base_real_time + timedelta(minutes=1)).replace(second=0, microsecond=0)
            initial_delay_until_next_minute_signal = (next_minute - base_real_time).total_seconds()

            # Calculate the initial delay to the next second mark, adjusted for the first signal timing
            next_second = (base_real_time + timedelta(seconds=1)).replace(microsecond=0) - timedelta(microseconds=350)
            initial_delay_until_next_second_signal = (next_second - base_real_time).total_seconds()


            # Recalculate if too close to next minute
            if initial_delay_until_next_second_signal < 0.1:
                next_second = (base_real_time + timedelta(seconds=2)).replace(microsecond=0) - timedelta(microseconds=350)
                if initial_delay_until_next_minute_signal < 0.:
                    next_minute = (base_real_time + timedelta(minutes=2)).replace(second=0, microsecond=0)



            while True:
                
                # Calculate the next timer start and stop signal timings relative to the elapsed time
                next_timer_start_signal_time = next_minute - timedelta(milliseconds=30.1)  # 30.1 ms before the top of the minute
                next_timer_stop_signal_time = next_timer_start_signal_time - timedelta(milliseconds=20) # 5 ms before the start signal
                
                # Calculate the next LED start and stop signal timings relative to the elapsed time
                next_led_on_signal_time = next_second - timedelta(milliseconds=0)
                next_led_off_signal_time = next_led_on_signal_time + timedelta(milliseconds=1)

                if next_timer_stop_signal_time < next_led_on_signal_time:

                    # Wait for the next stop signal time
                    print("Waiting to send next stop signal...")

                    while (base_real_time + timedelta(seconds=time.perf_counter() - base_perf_counter)) < next_timer_stop_signal_time:
                        pass

                    # Send stop signal. Pulse duration should has short as possible
                    send_signal(signal_pin=signal_pin_B, pulse_duration=pulse_duration)
                    print("\nSignal B!")


                    # Wait for the next start signal time
                    print("Waiting to send next start signal...")

                    while (base_real_time + timedelta(seconds=time.perf_counter() - base_perf_counter)) < next_timer_start_signal_time:
                        pass

                    # Send start signal. Pulse duration is not critical 
                    send_signal(signal_pin=signal_pin_A, pulse_duration=10*pulse_duration)
                    print("\nSignal A!")

                    # Update the delay for the next cycle, ensuring it accounts for a full minute past the initial delay
                    next_minute += timedelta(minutes=1)
                else:
                    while (base_real_time + timedelta(seconds=time.perf_counter() - base_perf_counter)) < next_led_on_signal_time:
                        pass
                    turn_led_on(pwr_led)
                    while (base_real_time + timedelta(seconds=time.perf_counter() - base_perf_counter)) < next_led_off_signal_time:
                        pass
                    turn_led_off(pwr_led)
                    next_second += timedelta(seconds=25/24.9823435)
    finally:
        GPIO.cleanup()
        # Restore the LED trigger
        write_to_file(f"{pwr_led}/trigger", pwr_current_trigger)
        write_to_file(f"{act_led}/trigger", act_current_trigger)


if __name__ == "__main__":
    main()
