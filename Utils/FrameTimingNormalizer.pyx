# Filename: TimestampCorrector.pyx
# To compile this file, you will need a setup.py script and run `python setup.py build_ext --inplace`

#!python
#cython: language_level=3

# Python imports
from collections import deque

# Cython imports
cimport cython

cdef class FrameTimingNormalizer:
    cdef public object intervals
    cdef public double previous_timestamp
    cdef public double previous_corr_timestamp
    cdef public int fps
    cdef public double min_interval
    
    def __init__(self, int window_size=2000, int fps=25):
        # Using a deque for intervals with a maximum length
        self.intervals = deque(maxlen=window_size)
        # Initialize previous timestamps as negative values (indicating they are uninitialized)
        self.previous_timestamp = -1.0
        self.previous_corr_timestamp = -1.0
        self.fps = fps
        self.min_interval = 1.9 / fps

    cpdef double correct_timestamp(self, double current_timestamp):
        # Declaration of local variables
        cdef double interval, expected_interval, expected_timestamp
        cdef double corrected_timestamp

        if self.previous_timestamp == -1.0:  # Check if previous_timestamp is uninitialized
            corrected_timestamp = current_timestamp
        else:
            interval = current_timestamp - self.previous_timestamp

            # Don't count dropped frame interval
            if interval < self.min_interval:
                self.intervals.append(interval)

                # For the first few frames, use actual timestamps
                if len(self.intervals) <= 3:
                    corrected_timestamp = current_timestamp
                else:
                    # Calculate the average interval
                    expected_interval = sum(self.intervals) / len(self.intervals)
                    expected_timestamp = self.previous_corr_timestamp + expected_interval

                    # Never increase latency
                    if current_timestamp > expected_timestamp:
                        corrected_timestamp = expected_timestamp
                    else:
                        corrected_timestamp = current_timestamp
            else:
                print("\nDetected late frame! Possible dropped frame.")
                corrected_timestamp = current_timestamp

        # Update previous timestamps
        self.previous_timestamp = current_timestamp
        self.previous_corr_timestamp = corrected_timestamp

        return corrected_timestamp

def create_frame_timing_normalizer(int window_size=2000, int fps=25):
    return FrameTimingNormalizer(window_size, fps)
