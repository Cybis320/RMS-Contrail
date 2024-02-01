import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class RollingShutterSensor:
    def __init__(self, sensor_height, fps, exposure_time, readout_time, led_pulses, led_rows):
        self.sensor_height = sensor_height
        self.fps = fps
        self.exposure_time = exposure_time
        self.readout_time = readout_time
        self.led_pulses = pd.read_csv(led_pulses)
        self.led_rows = led_rows

        self.update_read_times()

    def update_read_times(self):
        self.sensor_row_read_times = np.linspace(0, self.readout_time * self.sensor_height, self.sensor_height)
        self.exposure_starts = self.sensor_row_read_times
        self.exposure_ends = self.sensor_row_read_times + self.exposure_time

    def plot_sensor_readout(self, duration=2):
        fig, ax = plt.subplots(figsize=(10, 6))
        frame_time = 1 / self.fps
        num_frames = int(duration / frame_time)
        for index, led_pulse in self.led_pulses.iterrows():
            led_on_time = led_pulse['start_time']
            led_off_time = led_on_time + led_pulse['duration']
            for sensor_row in range(0, self.sensor_height, max(1, self.sensor_height // 32)):
                if sensor_row in self.led_rows:
                    ax.plot([led_on_time, led_off_time], [sensor_row, sensor_row], 'grey', linewidth=2)
        
            # Plot each sensor_row's exposure window
            for frame in range(num_frames):
                frame_offset = frame * frame_time
                for sensor_row in range(0, self.sensor_height, max(1, self.sensor_height // 32)):  # Plotting a subset for visibility
                    start_time = self.exposure_starts[sensor_row] + frame_offset
                    end_time = self.exposure_ends[sensor_row] + frame_offset
                    ax.plot([start_time, end_time], [sensor_row, sensor_row], 'b', linewidth=2)
                    
                    # Plot LED illumination for affected rows
                    if sensor_row in self.led_rows:
                                            
                        # Find overlap between LED illumination and sensor_row exposure window
                        overlap_start = max(led_on_time, start_time)
                        overlap_end = min(led_off_time, end_time)
                        
                        # Plot segment if there's an overlap
                        if overlap_start < overlap_end:
                            ax.plot([overlap_start, overlap_end], [sensor_row, sensor_row], 'r', linewidth=2)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('sensor_row Position (Subset)')
        ax.set_title('Rolling Shutter Sensor Exposure over 1 second with LED Signal')
        plt.grid(True)
        plt.show()

# Example of using the class
led_pulses = '/Users/lucbusquin/Projects/RMS-Contrail/pulses.csv'
led_rows = range(175, 500)  # LED affects rows 100 to 199
sensor = RollingShutterSensor(sensor_height=720, fps=25, exposure_time=1/250, readout_time=.03/720,
                              led_pulses=led_pulses, led_rows=led_rows)
sensor.plot_sensor_readout()
