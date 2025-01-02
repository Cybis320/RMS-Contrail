import time
import re
import ephem
import logging
import Utils.CameraControl as cc
from RMS.Misc import RmsDateTime
from enum import Enum
from datetime import datetime, timedelta

def test_twilight_detection():
    """Test the twilight detection specifically during twilight periods."""
    o = ephem.Observer()
    o.lat = '35.0572167'  # Albuquerque
    o.long = '-106.6837667'
    o.elevation = 1520
    s = ephem.Sun()
    
    # January times (UTC-7)
    # Local dawn is around 7:00-7:30 AM -> UTC 14:00-14:30
    # Local dusk is around 5:00-5:30 PM -> UTC 00:00-00:30 next day
    jan_twilight_times = [
        ("13:00", "Dawn"),    # 7:00 AM local
        ("13:30", "Dawn"),    # 7:30 AM local
        ("01:00", "Dusk"),    # 5:00 PM local previous day
        ("01:30", "Dusk"),    # 5:30 PM local previous day
    ]
    
    print("\nJanuary Twilight Tests (Times in UTC)")
    print("UTC      Local    Sun Alt    Is Dusk    Mode")
    print("-" * 55)
    
    for time_str, expected in jan_twilight_times:
        o.date = f"2024/01/01 {time_str}:00"
        s.compute(o)
        sun_alt = float(s.alt) * 180/ephem.pi
        
        # Convert UTC to local time (UTC-7 for January)
        utc_time = datetime.strptime(f"2024/01/01 {time_str}:00", "%Y/%m/%d %H:%M:%S")
        local_time = (utc_time - timedelta(hours=7)).strftime("%H:%M")
        
        if -17.8 < sun_alt < -5.8:
            original_horizon = o.horizon
            o.horizon = '-18'
            next_set = ephem.Date(o.next_setting(s))
            next_rise = ephem.Date(o.next_rising(s))
            is_dusk = next_set < next_rise
            o.horizon = original_horizon
            
            mode = determineCameraMode(sun_alt)
            print(f"{time_str:8} {local_time:8} {sun_alt:6.1f}°   {is_dusk!s:8} {mode.name:8}")
        else:
            print(f"{time_str:8} {local_time:8} {sun_alt:6.1f}°   {'N/A':8} {'Not Twilight':8}")

    # July times (UTC-6)
    # Local dawn is around 5:30-6:00 AM -> UTC 11:30-12:00
    # Local dusk is around 8:00-8:30 PM -> UTC 02:00-02:30 next day
    july_twilight_times = [
        ("11:30", "Dawn"),    # 5:30 AM local
        ("12:00", "Dawn"),    # 6:00 AM local
        ("02:00", "Dusk"),    # 8:00 PM local previous day
        ("02:30", "Dusk"),    # 8:30 PM local previous day
    ]
    
    print("\nJuly Twilight Tests (Times in UTC)")
    print("UTC      Local    Sun Alt    Is Dusk    Mode")
    print("-" * 55)
    
    for time_str, expected in july_twilight_times:
        o.date = f"2024/07/01 {time_str}:00"
        s.compute(o)
        sun_alt = float(s.alt) * 180/ephem.pi
        
        # Convert UTC to local time (UTC-6 for July)
        utc_time = datetime.strptime(f"2024/07/01 {time_str}:00", "%Y/%m/%d %H:%M:%S")
        local_time = (utc_time - timedelta(hours=6)).strftime("%H:%M")
        
        if -17.8 < sun_alt < -5.8:
            original_horizon = o.horizon
            o.horizon = '-18'
            next_set = ephem.Date(o.next_setting(s))
            next_rise = ephem.Date(o.next_rising(s))
            is_dusk = next_set < next_rise
            o.horizon = original_horizon
            
            mode = determineCameraMode(sun_alt)
            print(f"{time_str:8} {local_time:8} {sun_alt:6.1f}°   {is_dusk!s:8} {mode.name:8}")
        else:
            print(f"{time_str:8} {local_time:8} {sun_alt:6.1f}°   {'N/A':8} {'Not Twilight':8}")

test_twilight_detection()