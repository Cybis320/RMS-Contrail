'''
# debug code
import sys
sys.path.append('/Users/lucbusquin/Projects/RMS-Contrail')
# end of debug code
'''

import glob
import json
import os
import shutil
import platform
import tarfile
import subprocess
import re
import time
from collections import defaultdict
import collections

from datetime import datetime, timedelta, timezone
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from influxdb import InfluxDBClient

from RMS.Astrometry.ApplyAstrometry import GeoHt2xy
from RMS.Formats.Platepar import Platepar
from Utils.FOVKML import fovKML


# ---- Helper Functions ----

def getBoundingBoxFromKmlFile(file_path):
    """
    Extracts the bounding box from a KML file.

    Args:
        kml_path (str): The KML path

    Returns:
        Optional[Tuple[Tuple[float, float], Tuple[float, float]]]: The bounding box or None if not found
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Search for 'coordinates' element in the KML
    for elem in root.iter():
        if 'coordinates' in elem.tag:
            coords_text = elem.text.strip()
            break
    else:
        return None

    # Parse the coordinates
    coords = [tuple(map(float, point.split(','))) for point in coords_text.split()]

    min_lat = min(coord[1] for coord in coords)
    max_lat = max(coord[1] for coord in coords)
    min_lon = min(coord[0] for coord in coords)
    max_lon = max(coord[0] for coord in coords)

    return ((min_lat, min_lon), (max_lat, max_lon))


def extractTimestampFromName(name):
    """Extracts the timestamp from a file or directory name.

    Args:
        name (str): The name from which to extract the timestamp.

    Returns:
        datetime: The timestamp as a datetime object, or None if extraction fails.
    """

    # Extract the filename or directory name from the path
    name = name.rstrip('/')
    name = os.path.basename(name)

    # Updated regular expression to make fractional seconds part optional
    match = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})(?:_(\d{3,6}))?", name)
    if match:
        groups = match.groups()
        year, month, day, hour, minute, second = map(int, groups[:6])

        # Handle optional fractional seconds
        fractional_seconds = groups[6]
        if fractional_seconds:
            # Adjust microseconds based on the length of the fractional part
            microsecond = (
                int(fractional_seconds) if len(fractional_seconds) == 6
                else int(fractional_seconds)*1000
            )
        else:
            microsecond = 0

        # Create and return a datetime object with UTC timezone
        try:
            return datetime(year, month, day, hour, minute, second, microsecond, tzinfo=timezone.utc)
        except ValueError as e:
            print(f"Error creating datetime: {e}")
            return None

    return None


def queryAircraftPositions(client, begin_time, end_time, bounding_box=None, limit=None,
                           offset=None, retries=3, sleep_time=10):
    """Queries InfluxDB for aircraft positions within a given time range.

    Args:
        client (InfluxDBClient): The InfluxDB client.
        begin_time (datetime): The beginning time of the query.
        end_time (datetime): The ending time of the query.
        bounding_box (tuple, optional): Bounding box coordinates ((min_lat, min_lon),
                                        (max_lat, max_lon)). Defaults to None.
        limit (int, optional): The number of entries to fetch in each query. Defaults to None.
        offset (int, optional): The starting point to fetch entries for each query. Defaults to None.

    Returns:
        list: A list of aircraft positions within the time range.
    """

    # Create time bounds for the query
    start_time = begin_time.isoformat()
    end_time = end_time.isoformat()

    # Formulate the query
    query = f"SELECT * FROM adsb WHERE time >= '{start_time}' AND time <= '{end_time}'"

    # Add bounding box conditions if provided
    if bounding_box:
        (min_lat, min_lon), (max_lat, max_lon) = bounding_box
        bounding_box_conditions = (f" AND lat >= {min_lat}"
                                   f" AND lat <= {max_lat}"
                                   f" AND lon >= {min_lon}"
                                   f" AND lon <= {max_lon}")
        query += bounding_box_conditions

    # Add limit and offset conditions if provided
    if limit:
        query += f" LIMIT {limit}"
    if offset:
        query += f" OFFSET {offset}"

    query += ";"  # Close the query
    print(query)

    # Execute the query and fetch the results
    for _ in range(retries):
        try:
            results = client.query(query)
            points = list(results.get_points())
            return points
        except Exception as e:
            print(f"Query failed with error: {e}. Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
    print(f"Failed to execute query after {retries} attempts.")
    return []


def batchQueryAircraftPositions(client, begin_time, end_time, time_buffer=timedelta(seconds=5),
                                bounding_box=None, input_dir=None, batch_time=timedelta(hours=1)):
    """Queries InfluxDB for aircraft positions within a given time range, using hourly batch intervals.

    Args:
        client (InfluxDBClient): The InfluxDB client.
        begin_time (datetime): The beginning time of the query.
        end_time (datetime): The ending time of the query.
        time_buffer (timedelta, optional): The time buffer to extend the query time range.
                                           Defaults to 5 seconds.
        bounding_box (tuple, optional): Bounding box coordinates ((min_lat, min_lon), (max_lat, max_lon)).
                                        Defaults to None.
        input_dir (str, optional): The directory where the cache file will be stored and checked.
                                   If not provided, caching is skipped. Defaults to None.

    Returns:
        list: A list of aircraft positions within the time range.
    """

    begin_time -= time_buffer
    end_time += time_buffer

    if input_dir:
        # Check if a cache file covers the required time range.
        cache_files_pattern = os.path.join(input_dir, 'query_cache*.json')
        cache_files = glob.glob(cache_files_pattern)

        for cache_file in cache_files:
            timestamp_pattern = (r"(\d{4})(\d{2})(\d{2})"
                                 r"(\d{2})(\d{2})(\d{2})_"
                                 r"(\d{4})(\d{2})(\d{2})"
                                 r"(\d{2})(\d{2})(\d{2})")
            match = re.search(timestamp_pattern, cache_file)
            if match:
                # Extract and convert the date and time components
                date_time_components = map(int, match.groups())
                (year_start, month_start, day_start,
                 hour_start, minute_start, second_start,
                 year_end, month_end, day_end,
                 hour_end, minute_end, second_end) = date_time_components

                cache_begin_time = datetime(year_start, month_start, day_start,
                                            hour_start, minute_start, second_start, tzinfo=timezone.utc)
                cache_end_time = datetime(year_end, month_end, day_end,
                                          hour_end, minute_end, second_end, tzinfo=timezone.utc)

                start_msg = f"Found json with start: {cache_begin_time}, end: {cache_end_time}"
                print(start_msg)
                need_start_msg = (f"Need start: {begin_time.replace(microsecond=0)}, "
                                  f"end: {end_time.replace(microsecond=0)}")
                print(need_start_msg)

                if (cache_begin_time <= begin_time.replace(microsecond=0)
                        and cache_end_time >= end_time.replace(microsecond=0)):
                    print("Using cache file!")
                    with open(cache_file, 'r') as f:
                        return json.load(f)

    # If cache is not found or is not suitable, proceed with batched queries
    all_points = []

    current_begin_time = begin_time
    while current_begin_time < end_time:
        current_end_time = current_begin_time + batch_time
        if current_end_time > end_time:
            current_end_time = end_time

        try:
            points = queryAircraftPositions(client, current_begin_time, current_end_time, bounding_box)
            if points:
                all_points.extend(points)
        except Exception as e:
            print(f"Error querying between {current_begin_time} and {current_end_time}: {e}")

        current_begin_time = current_end_time

    # If an input directory is provided, save results to a cache file
    if input_dir:
        cache_filename = (f"query_cache_{begin_time.strftime('%Y%m%d%H%M%S')}_"
                          f"{end_time.strftime('%Y%m%d%H%M%S')}.json")
        cache_filepath = os.path.join(input_dir, cache_filename)
        try:
            with open(cache_filepath, 'w') as f:
                json.dump(all_points, f)
        except Exception as e:
            print(f"Error saving to cache file {cache_filepath}: {e}")

    return all_points


# TODO: is sorting necessary or influx already sort
def groupAndSortPoints(points):
    """
    Groups the aircraft position points by hex code, sorts them by timestamp,
    and removes any with malformed timestamps.

    Args:
        points (list): List of dictionaries representing aircraft positions.

    Returns:
        dict: Dictionary with hex codes as keys and lists of sorted points as values.
    """

    grouped_points = defaultdict(list)

    for point in points:
        hex_code = point.get('hex', None)

        if hex_code:
            # Convert string timestamps to datetime objects and filter out malformed ones
            try:
                point['time'] = datetime.fromisoformat(point['time'].replace('Z', '+00:00'))
            except ValueError:
                print(f"Malformed timestamp {point['time']} encountered. Skipping this point.")
                continue

            # Validate lat, lon, and alt_geom or alt_baro
            if (point.get('lat') is None
                    or point.get('lon') is None
                    or (point.get('alt_geom') is None and point.get('alt_baro') is None)):
                continue

            grouped_points[hex_code].append(point)

    # Sort points by datetime for each aircraft
    for hex_code, points in grouped_points.items():
        points.sort(key=lambda x: x['time'])

    return grouped_points


def createImageBatches(image_timestamps, batch_size=20):
    """Create batches of images.
    Args:
        image_timestamps (list of tuples): each tuple is a pair of '(img_file, timestamp)'
        batch_size (int): the number of images per batch

    Returns:
        a dictionary (list of tuples): each tuple is a pair of '(img_file, timestamp)'
        """
    number_of_images = len(image_timestamps)
    i = 0
    batches = []
    while i < number_of_images:
        end = min(i + batch_size - 1, number_of_images)
        # print(f"i:end = {i}:{end}")
        batch = image_timestamps[i:end][:]
        batches.append(batch)
        i = end

    return batches


def getPointsForBatch(grouped_points, batch_start_time, batch_end_time, time_buffer=timedelta(seconds=5)):
    batch_start_time -= time_buffer
    batch_end_time += time_buffer

    filtered_points = {}
    for hex_code, points_list in grouped_points.items():
        filtered = [point for point in points_list if batch_start_time <= point['time'] < batch_end_time]
        if filtered:  # Only add to the result if there are points in the time range
            filtered_points[hex_code] = filtered

    return filtered_points


def interpolateAircraftPositions(relevant_points, target_time, time_buffer=timedelta(seconds=5),
                                 adsb_latency=timedelta(milliseconds=0)):
    """Interpolates aircraft positions for a given timestamp.

    Args:
        relevant_points (dict): Grouped and sorted aircraft positions by hex code.
        target_time (datetime): The timestamp for which to interpolate positions.
        time_buffer (timedelta, optional): The time buffer around the target time
                                           in which to include ADS-B position positions.
        adsb_latency (timedelta, optional): the time offset to apply to ADS-B data

    Returns:
        list: A list of dictionaries containing interpolated positions.
    """
    # Dictionary to store the interpolated positions
    interpolated_positions = {}

    for hex_code, points_list in relevant_points.items():
        close_points = [point for point in points_list if abs(point['time'] - adsb_latency - target_time)
                        <= time_buffer]

        if not close_points:
            continue

        before = None
        after = None
        min_diff_before = timedelta.max  # Initialize with maximum timedelta
        min_diff_after = timedelta.max  # Initialize with maximum timedelta

        for point in close_points:
            time_diff = point['time'] - adsb_latency - target_time

            # Check for the closest before point
            if time_diff <= timedelta(0) and abs(time_diff) < min_diff_before:
                before = point
                min_diff_before = abs(time_diff)

            # Check for the closest after point
            elif time_diff > timedelta(0) and time_diff < min_diff_after:
                after = point
                min_diff_after = time_diff

        # Skip if neither before nor after points are available
        if not before and not after:
            continue

        interpolated_point = {}
        if before and after:
            for field in before:
                if isinstance(before[field], (int, float)) and isinstance(after[field], (int, float)):
                    time_difference = after['time'] - before['time']
                    weight = (target_time - before['time'] + adsb_latency)/time_difference
                    interpolated_value = before[field] + weight*(after[field] - before[field])
                    interpolated_point[field] = interpolated_value
                else:
                    time_diff_before = target_time - before['time'] + adsb_latency
                    time_diff_after = after['time'] - adsb_latency - target_time
                    closest_value = before[field] if time_diff_before < time_diff_after else after[field]
                    interpolated_point[field] = closest_value
        else:
            closest_point = before or after
            interpolated_point = closest_point.copy()

        interpolated_point['time'] = target_time
        interpolated_positions[hex_code] = interpolated_point
    # Flatten the dictionary values into a list
    flattened_interpolated_positions = list(interpolated_positions.values())

    return flattened_interpolated_positions


def addPixelCoordinates(interpolated_positions, platepar):
    """Add pixel coordinates to a list of interpolated aircraft positions.

    Args:
        interpolated_positions (list): List of dictionaries containing interpolated positions.
        platepar (object): Platepar object for coordinate conversion.

    Returns:
        list: Updated list of dictionaries with added 'x' and 'y' fields.
    """
    # Create a new list to store positions with pixel coordinates
    pixel_positions = []

    for position in interpolated_positions:
        # Create a copy of the current position
        new_position = position.copy()

        lat = new_position.get('lat', None)
        lon = new_position.get('lon', None)
        alt_geom_ft = new_position.get('alt_geom', None)
        alt_baro_ft = new_position.get('alt_baro', None)

        # Data validation: Skip if lat, lon, or both alt_geom and alt_baro are missing
        if lat is None or lon is None or (alt_geom_ft is None and alt_baro_ft is None):
            print(f"Skipped point: {position}")
            continue

        try:
            # Convert to meters and fallback to alt_baro if alt_geom is None
            h = (alt_geom_ft if alt_geom_ft is not None else alt_baro_ft)*0.3048

            # Convert geo coordinates to pixel coordinates
            x, y = GeoHt2xy(platepar, lat, lon, h)
            new_position['x'] = int(np.round(x[0]))
            new_position['y'] = int(np.round(y[0]))

            # Add the new position with pixel coordinates to the list
            pixel_positions.append(new_position)
        except Exception as e:
            print(f"Error processing position: {position}. Error: {e}")
    return pixel_positions


def centerDistanceTwoRectangles(w1, h1, w2, h2, theta):
    """Helper function to compute distance between the centers of two adjacent rectangles"""
    # Ensure theta is between 0 and 2*pi and convert angle from north up to standard trigo angle + 90
    theta = (-theta)%360
    theta = np.radians(theta)

    max_hor_dist = (w1 + w2)/2
    max_vert_dist = (h1 + h2)/2

    # Right or left side case
    if abs(np.tan(theta)*max_hor_dist) < max_vert_dist:
        return abs(max_hor_dist/np.cos(theta))

    # Top or bottom side case
    else:
        return abs(max_vert_dist/np.sin(theta))


def overlayDataOnImage(image, point, az_center):
    """Helper function to overlay aircraft data on an image."""
    x, y = point['x'], point['y']
    alt_baro = int(round(point.get('alt_baro', None))) if point.get('alt_baro', None) is not None else 'N/A'
    aircraft_type = point['t']
    aircraft_reg = point['r']
    aircraft_track = point['track'] if point['track'] is not None else az_center
    diff_angle = (aircraft_track - az_center)%360

    radius = 10
    outline_color = (0, 0, 0)
    color = (50, 255, 50)
    center = (x, y)

    # Length of the bars inside the circle
    bar_length = radius//2

    # Thickness of the bars
    inner_thickness = 1
    outline_thickness = 2

    # Define common line properties
    line_type = cv2.LINE_AA

    # Draw crosshair with an outline
    cv2.circle(image, center, radius, outline_color, outline_thickness, lineType=line_type)
    cv2.circle(image, center, radius, color, inner_thickness, lineType=line_type)

    # Draw each bar with an outline

    # 0 o'clock
    start_point_0 = (x, y - radius + bar_length)
    end_point_0_outline = (x, y - radius + outline_thickness)
    end_point_0_inner = (x, y - radius)
    cv2.line(image, start_point_0, end_point_0_outline, outline_color, outline_thickness, lineType=line_type)
    cv2.line(image, start_point_0, end_point_0_inner, color, inner_thickness, lineType=line_type)

    # 3 o'clock
    start_point_3 = (x + radius - bar_length, y)
    end_point_3_outline = (x + radius - outline_thickness, y)
    end_point_3_inner = (x + radius, y)
    cv2.line(image, start_point_3, end_point_3_outline, outline_color, outline_thickness, lineType=line_type)
    cv2.line(image, start_point_3, end_point_3_inner, color, inner_thickness, lineType=line_type)

    # 6 o'clock
    start_point_6 = (x, y + radius - bar_length)
    end_point_6_outline = (x, y + radius - outline_thickness)
    end_point_6_inner = (x, y + radius)
    cv2.line(image, start_point_6, end_point_6_outline, outline_color, outline_thickness, lineType=line_type)
    cv2.line(image, start_point_6, end_point_6_inner, color, inner_thickness, lineType=line_type)

    # 9 o'clock
    start_point_9 = (x - radius + outline_thickness, y)
    end_point_9_outline = (x - radius + bar_length, y)
    start_point_9_inner = (x - radius, y)
    end_point_9_inner = (x - radius + bar_length, y)
    cv2.line(image, start_point_9, end_point_9_outline, outline_color, outline_thickness, lineType=line_type)
    cv2.line(image, start_point_9_inner, end_point_9_inner, color, inner_thickness, lineType=line_type)

    # Adjust color as a function of an altitude threshold and set alt to N/A if None
    if isinstance(alt_baro, int):
        text = f"{alt_baro:,} ft\n{aircraft_type}\n{aircraft_reg}"
        if alt_baro >= 27000:
            color = (0, 50, 255)
    else:
        text = f"{alt_baro} ft\n{aircraft_type}\n{aircraft_reg}"  # This will be 'N/A ft'

    # Splits lines of text
    lines = text.split('\n')
    number_of_lines = len(lines)
    line_spacing = 2

    # Compute the size of the text for spacing
    total_text_height = 0
    max_width = 0

    for i, line in enumerate(lines):
        text_width, text_height = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        total_text_height += text_height + line_spacing
        if text_width > max_width:
            max_width = text_width

    # Position text as a function of aircraft track away from potential contrail
    offset = centerDistanceTwoRectangles(2*radius + 5, 2*radius + 5, max_width, total_text_height, diff_angle)
    x_offset = offset*np.cos(np.radians(diff_angle))
    y_offset = offset*np.sin(np.radians(diff_angle))

    x_new = x + x_offset
    y_new = y + y_offset

    # Write multiline text
    for i, line in enumerate(lines):
        text_width, text_height = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_height += line_spacing
        x_line = x_new - text_width/2
        y_line = y_new + text_height*(i+1 - number_of_lines/2)
        cv2.putText(image, line, (int(x_line), int(y_line)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(image, line, (int(x_line), int(y_line)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, lineType=cv2.LINE_AA)


def findClosestEntry(dictionary, target_timestamp):
    closest_timestamp = None
    smallest_difference = None

    for timestamp in dictionary.keys():
        difference = abs(timestamp - target_timestamp)

        if smallest_difference is None or difference < smallest_difference:
            smallest_difference = difference
            closest_timestamp = timestamp

    return dictionary[closest_timestamp]


def cleanup(parent_dir):
    '''Delete 'temp_images' folder within parent_dir and archive working files excluding .mp4 files.'''
    temp_image_folder = os.path.join(parent_dir, 'temp_images')  # Path to the temp_images folder

    # Attempt to delete the temp_images folder
    if os.path.exists(temp_image_folder):
        try:
            shutil.rmtree(temp_image_folder)
            print(f"Deleted temporary directory: {temp_image_folder}")
        except Exception as e:
            print(f"Error deleting temporary directory: {e}")

    # Prepare the archive name and path
    dir_name = os.path.basename(parent_dir).lstrip('JPG_')
    archive_name = f"{dir_name}_adsb_working_files.tar.bz2"
    archive_path = os.path.join(parent_dir, archive_name)

    # Track files added to the archive
    files_added_to_archive = []

    # Creating the archive
    try:
        with tarfile.open(archive_path, "w:bz2") as tar:
            for root, dirs, files in os.walk(parent_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file_path != archive_path and not file.endswith('.mp4'):
                        tar.add(file_path, arcname=os.path.relpath(file_path, parent_dir))
                        files_added_to_archive.append(file_path)
        print(f"Archived non-video files to {archive_path}")

        # Verify the archive exists and has content before deleting files
        if os.path.exists(archive_path) and os.path.getsize(archive_path) > 0 and files_added_to_archive:
            for file_path in files_added_to_archive:
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Could not delete {file_path}: {e}")
    except Exception as e:
        print(f"Error creating archive: {e}")


# ---- Debug Functions ----

def getAllCoordinatesFromKml(file_path, alt=14000):
    '''Debug function to get all the coord from the KML.
       Can be used to overlay on image. All point should be on the edge of the image'''

    tree = ET.parse(file_path)
    root = tree.getroot()

    # Search for 'coordinates' element in the KML
    for elem in root.iter():
        if 'coordinates' in elem.tag:
            coords_text = elem.text.strip()
            break
    else:
        return None

    # Parse the coordinates and convert them into dictionaries
    coords = []
    for point in coords_text.split():
        lon, lat, _ = map(float, point.split(','))
        coords.append({'lat': lat, 'lon': lon, 'alt_geom': alt})

    return coords


def printStructure(data, depth=0, max_depth=3, max_items=3):
    data_type = type(data)
    indent = '  '*depth  # Indentation for visual clarity

    # If it's a list or tuple, inspect its items
    if data_type in [list, tuple]:
        print(f"{indent}List or Tuple with {len(data)} items")
        if depth < max_depth:
            for item in data[:max_items]:
                printStructure(item, depth + 1)

    # If it's a dictionary or defaultdict, inspect its values
    elif data_type in [dict, collections.defaultdict]:
        print(f"{indent}{data_type.__name__} with {len(data)} keys")
        if depth < max_depth:
            for i, (key, value) in enumerate(data.items()):
                if i >= max_items:
                    break
                print(f"{indent}  Key: {key}")
                printStructure(value, depth + 2)

    # Otherwise, just print its type
    else:
        print(f"{indent}{data_type}")


# === ENTRY POINT ===
def runOverlayOnImages(input_path, platepar):
    """Overlay aircraft positions on image files in a directory or a single image.

    Args:
        client (InfluxDBClient): The InfluxDB client.
        input_path (str): The path to a directory containing image files or a single image file.
        platepar (object): Either a Platepar object or a path to a platepar json for coordinate conversion.
    Return:
        Outpur_dir: [path object] the path to the dir containing the images with adsb overlay
    """
    start_total_time = time.time()

    # Initialize the InfluxDB client
    # TODO: consider defining URL in config file
    client = InfluxDBClient(host='contrailcast.local', port=8086)
    client.switch_database('adsb_data')

    time_buffer = timedelta(seconds=30)

    # Determine input type and set appropriate directories
    if os.path.isdir(input_path):
        output_dir = os.path.join(input_path, "temp_images")
        image_files = (glob.glob(os.path.join(input_path, '*.jpg'))
                       + glob.glob(os.path.join(input_path, '*.png')))

        kml_dir = input_path
    elif os.path.isfile(input_path):
        output_dir = os.path.join(os.path.dirname(input_path), "temp_images")
        image_files = [input_path]
        kml_dir = os.path.dirname(input_path)
    else:
        print("Invalid input path.")
        return None

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract, filter, and sort timestamps
    image_timestamps = {img_file: extractTimestampFromName(img_file) for img_file in image_files}
    image_timestamps = {img: ts for img, ts in image_timestamps.items() if ts is not None}
    sorted_timestamps = sorted(image_timestamps.items(), key=lambda item: item[1])

    if sorted_timestamps:
        min_timestamp = sorted_timestamps[0][1]
        max_timestamp = sorted_timestamps[-1][1]
        if max_timestamp - min_timestamp > timedelta(days=1):
            print("Timestamps exceed a 24-hour period.")
            return None
    else:
        print("No valid timestamps found.")
        return None

    find_platepar = False

    if isinstance(platepar, str) and os.path.isfile(platepar):
        with open(platepar) as f:
            try:
                # Load the JSON file with recalibrated platepars
                recalibrated_platepars_dict = json.load(f)

            except json.decoder.JSONDecodeError:
                return None

            # Convert the dictionary of recalibrated platepars to a dictionary of Platepar objects
            recalibrated_platepars = {}

            # Initialize to extreme values
            min_time_pp = datetime.max.replace(tzinfo=timezone.utc)
            max_time_pp = datetime.min.replace(tzinfo=timezone.utc)

            for ff_name in recalibrated_platepars_dict:
                print(f"ff_name: {ff_name}")
                time_pp = extractTimestampFromName(ff_name)

                # Update min and max times
                min_time_pp = min(min_time_pp, time_pp)
                max_time_pp = max(max_time_pp, time_pp)

                pp = Platepar()
                pp.loadFromDict(recalibrated_platepars_dict[ff_name])

                recalibrated_platepars[time_pp] = pp

            if max_timestamp < min_time_pp:
                platepar = recalibrated_platepars[min_time_pp]
            elif min_timestamp > max_time_pp:
                platepar = recalibrated_platepars[max_time_pp]
            else:
                find_platepar = True

    if find_platepar:
        kml_path = fovKML(kml_dir, recalibrated_platepars[max_time_pp], area_ht=18000, plot_station=False)

    else:
        kml_path = fovKML(kml_dir, platepar, area_ht=18000, plot_station=False)

    bounding_box = getBoundingBoxFromKmlFile(kml_path)

    query_result = batchQueryAircraftPositions(client, min_timestamp, max_timestamp,
                                               time_buffer, bounding_box, input_path)
    grouped_points = groupAndSortPoints(query_result)

    # Divide grouped_points into batches
    batch_size = 50  # 75 gave best perf on a macbook
    batches = createImageBatches(sorted_timestamps, batch_size)
    image_count = 0
    total_images = len(sorted_timestamps)

    print("Preparing files for the ADS-B timelapse...")

    for batch in batches:
        batch_start_time = batch[0][1]
        batch_end_time = batch[-1][1]
        # print(f"\nstart time: {batch_start_time}")
        # print(f"end time: {batch_end_time}")

        relevant_points = getPointsForBatch(grouped_points, batch_start_time, batch_end_time, time_buffer)

        # Overlay aircraft positions on images
        for img_file, timestamp in batch:
            interpolated_points = interpolateAircraftPositions(relevant_points, timestamp, time_buffer)

            # Load the closest recalibrated platepar
            if find_platepar:
                platepar = findClosestEntry(recalibrated_platepars, timestamp)

            points_XY = addPixelCoordinates(interpolated_points, platepar)

            # Correct for rolling shutter
            # for point in points_XY:
            #     corr_timestamp = timestamp + 0.03*point[1]/platepar.Y_res
            #     corr_interpolated_points = interpolateAircraftPositions(relevant_points, corr_timestamp, time_buffer)
            #     corr_points_XY = addPixelCoordinates(corr_interpolated_points, platepar)

            image = cv2.imread(img_file)

            # Check if the image was loaded successfully
            if image is None:
                print(f"Failed to load image: {img_file}")
                continue  # Skip processing this image

            for point in points_XY:
                overlayDataOnImage(image, point, platepar.az_centre)

            # overlay timestamp
            img_name = os.path.basename(img_file)
            station_name = img_name.split("_")[0]

            height, _, _ = image.shape

            timestamp = extractTimestampFromName(img_file).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + ' UTC'
            cv2.putText(image, f"{station_name} {timestamp}", (10, height - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"{station_name} {timestamp}", (10, height - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            output_name = f"{img_name.rsplit('.', 1)[0]}_overlay.{img_name.rsplit('.', 1)[1]}"
            output_path = os.path.join(output_dir, output_name)
            cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])

            image_count += 1
            print(f"\r{image_count} / {total_images} Elapsed: {time.time() - start_total_time:.2f}s. "
                  f"(batches of: {batch_size})", end="", flush=True)

    print("\nFinished applying ADS-B overlay to images.")

    return output_dir


# CV2 MP4
# def createVideoFromImages(image_folder, video_name, delete_images=False):
#     images = [img for img in sorted(glob.glob(os.path.join(image_folder, "*_overlay.*")))]
#     if len(images) == 0:
#         print("No overlay images found.")
#         return
    
#     frame = cv2.imread(images[0])

#     # Setting the frame width, height, assuming all images are the same size
#     height, width, layers = frame.shape
#     size = (width, height)

#     # Get the parent directory of the image_folder
#     parent_directory = os.path.dirname(image_folder)
#     video_path = os.path.join(parent_directory, video_name)

#     out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

#     for i in range(len(images)):
#         img_path = images[i]
#         out.write(cv2.imread(img_path))

#         # Delete the image file if the flag is set
#         if delete_images:
#             os.remove(img_path)

#     out.release()

# cfr   size (MB)
#   1   2050
#  15    502
#  20    168
#  23     70


def createVideoFromImages(temp_image_folder, parent_dir, video_path, fps=30, crf=20, delete_images=False):
    """

    """
    images = [img for img in sorted(glob.glob(os.path.join(temp_image_folder, "*_overlay.*")))]
    if len(images) == 0:
        print("No overlay images found.")
        return

    # Create a text file listing all the images
    list_file_path = os.path.join(temp_image_folder, "filelist.txt")
    with open(list_file_path, 'w') as f:
        for img_path in images:
            f.write(f"file '{os.path.basename(img_path)}'\n")

    # Formulate the ffmpeg command
    # base_command = (f"-nostdin -f concat -safe 0 -v quiet -r {fps} -y -i {list_file_path} -c:v libx264 "
    #                 f"-pix_fmt yuv420p -crf {crf} -g 15 -vf \"hqdn3d=4:3:6:4.5,lutyuv=y=gammaval(0.77)\" "
    #                 f"{video_path}"

    base_command = (f"-nostdin -f concat -safe 0 -v quiet -r {fps} -y -i "
                    f"{list_file_path} -c:v libx264 -crf {crf} -g 15 {video_path}")

    if platform.system() in ['Linux', 'Darwin']:  # Darwin is macOS
        software_name = "ffmpeg"
    elif platform.system() == 'Windows':
        software_name = os.path.join(os.path.dirname(__file__), "ffmpeg.exe")
    else:
        print("Unsupported platform.")
        return
    encode_command = f"{software_name} {base_command}"

    # Execute the command
    print("Creating timelapse using ffmpeg...")
    subprocess.call(encode_command.format(fps=fps, list_file_path=list_file_path, crf=crf,
                                          video_path=video_path), shell=True)

    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        print(f"Video created successfully at {video_path}")
        # Optionally, delete the temp images and archive files
        if delete_images:
            cleanup(parent_dir)
    else:
        print("Video creation failed or resulted in an empty file.")


if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""Compute the aircraft X, Y pixel coordinates given
                                         the platepar and jpg images.""",
                                         formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('platepar', metavar='PLATEPAR', type=str,
                            help="Path to the platepar file.")

    arg_parser.add_argument('input_path', metavar='INPUT_PATH', type=str,
                            help="Path to either a single image file (jpg or png) "
                                 "or a directory containing the images.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    # Load the platepar file
    filename = cml_args.platepar

    # Check the file extension
    file_extension = os.path.splitext(filename)[1]

    # If it's a JSON file, pass the path, otherwise pass a platepar object
    if file_extension == '.cal':
        pp = Platepar()
        pp.read(filename)
    elif file_extension == '.json':
        pp = filename
    else:
        raise ValueError("Unsupported calibration file")

    temp_dir = runOverlayOnImages(cml_args.input_path, pp)

    # If the input_path is a directory, run the function to create mp4 video

    if os.path.isdir(cml_args.input_path):
        normalized_path = (os.path.dirname(cml_args.input_path) if cml_args.input_path.endswith('/')
                           else cml_args.input_path)
        dir_name = os.path.basename(normalized_path).lstrip('JPG_')
        timelapse_file_name = dir_name + "_adsb_timelapse.mp4"
        video_path = os.path.join(cml_args.input_path, timelapse_file_name)

        createVideoFromImages(temp_dir, normalized_path, video_path, delete_images=False)

'''
# Debug Test code
# Load the platepar file
pp = Platepar()
#pp.read(cml_args.platepar)
pp.read('/Users/lucbusquin/Projects/test/platepar_cmn2010.cal')

# Initialize the InfluxDB client
client = InfluxDBClient(host='adsbexchange.local', port=8086)
client.switch_database('adsb_data')  # Switch to your specific database

#runOverlayOnImages(client, cml_args.input_path, pp)
runOverlayOnImages(client, '/Users/lucbusquin/Projects/test', pp)

# If the input_path is a directory, run the function to create mp4 video
if os.path.isdir('/Users/lucbusquin/Projects/test'):
    #output_dir = os.path.join(cml_args.input_path, "overlay_images")
    output_dir = os.path.join('/Users/lucbusquin/Projects/test', "overlay_images")
    video_name = os.path.join(output_dir, "overlay_video.mp4")
    createVideoFromImages(output_dir, video_name, False)
'''
