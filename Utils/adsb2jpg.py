'''
# debug code
import sys
sys.path.append('/Users/lucbusquin/Projects/RMS-Contrail')
# end of debug code
'''

import glob
import json
import os
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

def get_bounding_box_from_kml_file(file_path):
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




def extract_timestamp_from_image(image_name):
    """Extracts the timestamp from the image name (JPG or PNG).
    
    Args:
        image_name (str): The name of the image file.
        
    Returns:
        datetime: The timestamp as a datetime object, or None if extraction fails.
    """

    # Use a regular expression to find the date and time components in the image name
    match = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_(\d{3})", image_name)

    if match:
        # Extract and convert the date, time, and milliseconds components to integers
        year, month, day, hour, minute, second, millisecond = map(int, match.groups())

        # Create and return a datetime object with milliseconds
        try:
            microsecond = millisecond * 1000
            return datetime(year, month, day, hour, minute, second, microsecond, tzinfo=timezone.utc)
        except:
            return None

    return None




def query_aircraft_positions(client, begin_time, end_time, bounding_box=None):
    """Queries InfluxDB for aircraft positions within a given time range.
    
    Args:
        client (InfluxDBClient): The InfluxDB client.
        begin_time (datetime): The beginning time of the query.
        end_time (datetime): The ending time of the query.
        bounding_box (tuple, optional): Bounding box coordinates ((min_lat, min_lon), (max_lat, max_lon)). Defaults to None.
        
    Returns:
        list: A list of aircraft positions within the time range.
    """
    
    # Create time bounds for the query
    start_time = (begin_time).isoformat() + 'Z'  # Convert to ISO format and add 'Z' to indicate UTC
    final_end_time = (end_time).isoformat() + 'Z'
    
    # Formulate the query
    query = f"SELECT * FROM adsb WHERE time >= '{start_time}' AND time <= '{final_end_time}'"
    
    # Add bounding box conditions if provided
    if bounding_box:
        (min_lat, min_lon), (max_lat, max_lon) = bounding_box
        bounding_box_conditions = f" AND lat >= {min_lat} AND lat <= {max_lat} AND lon >= {min_lon} AND lon <= {max_lon}"
        query += bounding_box_conditions
    
    query += ";"  # Close the query
    print(query)

    # Execute the query and fetch the results
    results = client.query(query)
    points = list(results.get_points())
    
    return points



def batch_query_aircraft_positions(client, begin_time, end_time, time_buffer=timedelta(seconds=5), bounding_box=None, input_dir=None, batch_duration=timedelta(hours=1)):
    """Queries InfluxDB for aircraft positions within a given time range, using batched time intervals.
    
    Args:
        client (InfluxDBClient): The InfluxDB client.
        begin_time (datetime): The beginning time of the query.
        end_time (datetime): The ending time of the query.
        time_buffer (timedelta, optional): The time buffer to extend the query time range. Defaults to 5 seconds.
        bounding_box (tuple, optional): Bounding box coordinates ((min_lat, min_lon), (max_lat, max_lon)). Defaults to None.
        input_dir (str, optional): The directory where the cache file will be stored and checked. If not provided, caching is skipped. Defaults to None.
        batch_duration (timedelta, optional): The duration of each batch interval. Defaults to 1 hour.
        
    Returns:
        list: A list of aircraft positions within the time range.
    """
    # Pad begin and end time with buffer
    begin_time -= time_buffer
    end_time += time_buffer

    if input_dir:
        # Check if a cache file covers the required time range.
        cache_files = glob.glob(os.path.join(input_dir, '*.json'))
        for cache_file in cache_files:
            match = re.search(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", cache_file)
            if match:
                # Extract and convert the date and time components
                year_start, month_start, day_start, hour_start, minute_start, second_start, year_end, month_end, day_end, hour_end, minute_end, second_end = map(int, match.groups())
                cache_begin_time = datetime(year_start, month_start, day_start, hour_start, minute_start, second_start, tzinfo=timezone.utc)
                cache_end_time = datetime(year_end, month_end, day_end, hour_end, minute_end, second_end, tzinfo=timezone.utc)
                
                if cache_begin_time <= begin_time and cache_end_time >= end_time:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
    
    # If cache is not found or is not suitable, proceed with batched queries
    all_points = []
    current_start_time = begin_time
    while current_start_time < end_time:
        current_end_time = min(current_start_time + batch_duration, end_time)
        try:
            points = query_aircraft_positions(client, current_start_time, current_end_time, bounding_box)
            all_points.extend(points)
        except Exception as e:
            print(f"Error querying for time range {current_start_time} to {current_end_time}: {e}")
        current_start_time = current_end_time
    
    # If an input directory is provided, save results to a cache file
    if input_dir:
        cache_filename = f"query_cache_{begin_time.strftime('%Y%m%d%H%M%S')}_{end_time.strftime('%Y%m%d%H%M%S')}.json"
        cache_filepath = os.path.join(input_dir, cache_filename)
        try:
            with open(cache_filepath, 'w') as f:
                json.dump(all_points, f)
        except Exception as e:
            print(f"Error saving to cache file {cache_filepath}: {e}")

    return all_points



# TODO: create function that detect ADS-B dark windows.
#       Goal is to overlay red flag when no ADS-B data exists



# TODO: is sorting necessary or influx already sort
def group_and_sort_points(points):
    """
    Groups the aircraft position points by hex code, sorts them by timestamp, and removes any with malformed timestamps.
    
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
            if point.get('lat') is None or point.get('lon') is None or (point.get('alt_geom') is None and point.get('alt_baro') is None):
                continue

            grouped_points[hex_code].append(point)
    
    # Sort points by datetime for each aircraft
    for hex_code, points in grouped_points.items():
        points.sort(key=lambda x: x['time'])

    return grouped_points




def create_image_batches(image_timestamps, batch_size=20):
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
        end = min(i + batch_size -1, number_of_images)
        #print(f"i:end = {i}:{end}")
        batch = image_timestamps[i:end][:]
        batches.append(batch)
        i = end
    
    return batches




def get_points_for_batch(grouped_points, batch_start_time, batch_end_time, time_buffer=timedelta(seconds=5)):
    batch_start_time -= time_buffer
    batch_end_time += time_buffer
    
    filtered_points = {}
    for hex_code, points_list in grouped_points.items():
        filtered = [point for point in points_list if batch_start_time <= point['time'] < batch_end_time]
        if filtered:  # Only add to the result if there are points in the time range
            filtered_points[hex_code] = filtered
            
    return filtered_points




def interpolate_aircraft_positions(relevant_points, target_time, time_buffer=timedelta(seconds=5)):
    """Interpolates aircraft positions for a given timestamp.

    Args:
        grouped_points (dict): Grouped and sorted aircraft positions by hex code.
        target_time (datetime): The timestamp for which to interpolate positions.
        max_time_diff (int, optional): The maximum time difference in seconds for which interpolation is allowed.

    Returns:
        list: A list of dictionaries containing interpolated positions.
    """
    # Dictionary to store the interpolated positions
    interpolated_positions = {}

    for hex_code, points_list in relevant_points.items():
        close_points = [point for point in points_list if abs(point['time'] - target_time) <= time_buffer]
        
        if not close_points:
            continue
        
        before = None
        after = None
        min_diff_before = timedelta.max  # Initialize with maximum timedelta
        min_diff_after = timedelta.max  # Initialize with maximum timedelta

        for point in close_points:
            time_diff = point['time'] - target_time
            
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
                    weight = (target_time - before['time']) / (after['time'] - before['time'])
                    interpolated_point[field] = before[field] + weight * (after[field] - before[field])
                else:
                    interpolated_point[field] = before[field] if (target_time - before['time']) < (after['time'] - target_time) else after[field]
        else:
            closest_point = before or after
            interpolated_point = closest_point.copy()

        interpolated_point['time'] = target_time
        interpolated_positions[hex_code] = interpolated_point
    # Flatten the dictionary values into a list
    flattened_interpolated_positions = list(interpolated_positions.values())

    return flattened_interpolated_positions




def add_pixel_coordinates(interpolated_positions, platepar):
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
            h = (alt_geom_ft if alt_geom_ft is not None else alt_baro_ft) * 0.3048
            
            # Convert geo coordinates to pixel coordinates
            x, y = GeoHt2xy(platepar, lat, lon, h)
            new_position['x'] = int(np.round(x[0]))
            new_position['y'] = int(np.round(y[0]))
            
            # Add the new position with pixel coordinates to the list
            pixel_positions.append(new_position)
        except Exception as e:
            print(f"Error processing position: {position}. Error: {e}")
    return pixel_positions




def overlay_data_on_image(image, point, img_file):
    """Helper function to overlay aircraft data on an image."""
    x, y = point['x'], point['y']
    alt_baro = int(round(point.get('alt_baro', None))) if point.get('alt_baro', None) is not None else 'N/A'
    aircraft_type = point['t']
    rectangle_size = 15
    cv2.rectangle(image, (x - rectangle_size, y - rectangle_size), (x + rectangle_size, y + rectangle_size), (0, 255, 0), 1)
    text = f"{alt_baro} ft ({aircraft_type})"
    cv2.putText(image, text, (x - rectangle_size - 5, y - rectangle_size - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    station_name = img_file.split("_")[1]
    timestamp = "_".join(img_file.split("_")[2:6])
    cv2.putText(image, f"Station: {station_name}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, f"Time: {timestamp}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)



# ---- Debug Functions ---- 

def get_all_coordinates_from_kml(file_path, alt=14000):
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




def print_structure(data, depth=0, max_depth=3, max_items=3):
    data_type = type(data)
    indent = '  ' * depth  # Indentation for visual clarity
    
    # If it's a list or tuple, inspect its items
    if data_type in [list, tuple]:
        print(f"{indent}List or Tuple with {len(data)} items")
        if depth < max_depth:
            for item in data[:max_items]:
                print_structure(item, depth + 1)
    
    # If it's a dictionary or defaultdict, inspect its values
    elif data_type in [dict, collections.defaultdict]:
        print(f"{indent}{data_type.__name__} with {len(data)} keys")
        if depth < max_depth:
            for i, (key, value) in enumerate(data.items()):
                if i >= max_items:
                    break
                print(f"{indent}  Key: {key}")
                print_structure(value, depth + 2)

    # Otherwise, just print its type
    else:
        print(f"{indent}{data_type}")




# === ENTRY POINT ===
def run_overlay_on_images(client, input_path, platepar):
    """Overlay aircraft positions on image files in a directory or a single image.
    
    Args:
        client (InfluxDBClient): The InfluxDB client.
        input_path (str): The path to a directory containing image files or a single image file.
        platepar (object): Platepar object for coordinate conversion.
    """
    start_total_time = time.time()
    time_buffer = timedelta(seconds=5)

    # Determine input type and set appropriate directories
    if os.path.isdir(input_path):
        output_dir = os.path.join(input_path, "overlay_images")
        image_files = glob.glob(os.path.join(input_path, '*.jpg')) + glob.glob(os.path.join(input_path, '*.png'))
        kml_dir = input_path
    elif os.path.isfile(input_path):
        output_dir = os.path.join(os.path.dirname(input_path), "overlay_images")
        image_files = [input_path]
        kml_dir = os.path.dirname(input_path)
    else:
        print("Invalid input path.")
        return

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    kml_path = fovKML(kml_dir, platepar, area_ht=14000, plot_station=False)
    bounding_box = get_bounding_box_from_kml_file(kml_path)
    
    # Extract, filter, and sort timestamps
    image_timestamps = {img_file: extract_timestamp_from_image(img_file) for img_file in image_files}
    image_timestamps = {img: ts for img, ts in image_timestamps.items() if ts is not None}
    image_timestamps = sorted(image_timestamps.items(), key=lambda item: item[1])

    # Query aircraft positions
    min_timestamp = image_timestamps[0][1]
    max_timestamp = image_timestamps[-1][1]
    query_result = batch_query_aircraft_positions(client, min_timestamp, max_timestamp, time_buffer, bounding_box, input_path)
    grouped_points = group_and_sort_points(query_result)

    # Divide grouped_points into batches
    batch_size = 75 # Gave best perf on a macbook
    batches = create_image_batches(image_timestamps, batch_size)
    image_count = 0
    total_images = len(image_timestamps)

    for batch in batches:
        batch_start_time = batch[0][1]
        batch_end_time = batch[-1][1]
        print(f"\nstart time: {batch_start_time}")
        print(f"end time: {batch_end_time}")

        relevant_points = get_points_for_batch(grouped_points, batch_start_time, batch_end_time, time_buffer)

        # Overlay aircraft positions on images
        for img_file, timestamp in batch:
            interpolated_points = interpolate_aircraft_positions(relevant_points, timestamp, time_buffer)
            points_XY = add_pixel_coordinates(interpolated_points, platepar)
            
            image = cv2.imread(img_file)
            for point in points_XY:
                overlay_data_on_image(image, point, img_file)

            img_name = os.path.basename(img_file)
            output_name = f"{img_name.rsplit('.', 1)[0]}_overlay.{img_name.rsplit('.', 1)[1]}"
            output_path = os.path.join(output_dir, output_name)
            cv2.imwrite(output_path, image)

            image_count += 1
            print(f"\rSaved {image_count}/{total_images}. {time.time() - start_total_time:.2f}s. batches of: {batch_size}", end="", flush=True)




def create_video_from_images(image_folder, video_name, delete_images=False):
    images = [img for img in sorted(glob.glob(os.path.join(image_folder, "*_overlay.*")))]
    if len(images) == 0:
        print("No overlay images found.")
        return
    
    frame = cv2.imread(images[0])

    # Setting the frame width, height, assuming all images are the same size
    height, width, layers = frame.shape
    size = (width, height)

    # Get the parent directory of the image_folder
    parent_directory = os.path.dirname(image_folder)
    video_path = os.path.join(parent_directory, video_name)

    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

    for i in range(len(images)):
        img_path = images[i]
        out.write(cv2.imread(img_path))

        # Delete the image file if the flag is set
        if delete_images:
            os.remove(img_path)

    out.release()



if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""Compute the aircraft X, Y pixel coordinates given the platepar and jpg images. \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('platepar', metavar='PLATEPAR', type=str, \
                    help="Path to the platepar file.")
    
    arg_parser.add_argument('input_path', metavar='INPUT_PATH', type=str, \
                    help="Path to either a single image file (jpg or png) or a directory containing the images.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################
    
    # Load the platepar file
    filename = cml_args.platepar

    # Check the file extension
    file_extension = os.path.splitext(filename)[1]

    # If it's a JSON file, set the format accordingly, else set it to None
    fmt = 'json' if file_extension == '.json' else None
    pp = Platepar()
    pp.read(filename, fmt=fmt)

    # Initialize the InfluxDB client
    client = InfluxDBClient(host='adsbexchange.local', port=8086)
    client.switch_database('adsb_data')  # Switch to your specific database

    run_overlay_on_images(client, cml_args.input_path, pp)


    # If the input_path is a directory, run the function to create mp4 video
    if os.path.isdir(cml_args.input_path):
        output_dir = os.path.join(cml_args.input_path, "overlay_images")
        video_name = os.path.join(output_dir, "overlay_video.mp4")
        create_video_from_images(output_dir, video_name, True)

'''
# Debug Test code
# Load the platepar file
pp = Platepar()
#pp.read(cml_args.platepar)
pp.read('/Users/lucbusquin/Projects/test/platepar_cmn2010.cal')

# Initialize the InfluxDB client
client = InfluxDBClient(host='adsbexchange.local', port=8086)
client.switch_database('adsb_data')  # Switch to your specific database

#run_overlay_on_images(client, cml_args.input_path, pp)
run_overlay_on_images(client, '/Users/lucbusquin/Projects/test', pp)

# If the input_path is a directory, run the function to create mp4 video
if os.path.isdir('/Users/lucbusquin/Projects/test'):
    #output_dir = os.path.join(cml_args.input_path, "overlay_images")
    output_dir = os.path.join('/Users/lucbusquin/Projects/test', "overlay_images")
    video_name = os.path.join(output_dir, "overlay_video.mp4")
    create_video_from_images(output_dir, video_name, False)
'''
