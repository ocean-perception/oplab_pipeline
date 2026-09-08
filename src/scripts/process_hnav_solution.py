"""
Use SprintNav Hybrid navigation solution as AUV position. 
Accounts for offsets of payloads to SprintNav on the AUV body and interpolates 
navigation solution to image timestamps. 
Output format follows auv_nav process output.

Usage:
process_hnav_solution.py [-h] payload [-b bagfiles] [-d dive_dir] [-s start_time] [-e end_time] [-f]

positional_arguments:
payload         Payload to adjust positional offsets for and save navigation solution

optional_arguments:
-h, --help      Show this help message and exit
-b bagfiles     Bagfile(s) name or partial names containing SprintNav Hybrid navigation (default: None
                to find all bagfiles in nav directory) e.g. sparus2_2026-06-24-*.bag
-d dive_dir     Directory where raw dive data is stored (default:current 
                working directory)
-s start_time   Date and time to start processing dive data for, YYYYMMDD HHMMSS (default: None)
-e end_time     Date and time to end processing dive data for, YYYYMMDD HHMMSS (default: None)
-f, --force     Overwrite output file if exists (default: False)

"""


import pandas as pd
import numpy as np
from pyproj import Transformer
from scipy.spatial.transform import Rotation
import yaml
import rosbag
import rospy
from datetime import datetime,UTC
import calendar
from pathlib import Path
from glob import glob
import os
import argparse

def decode_hnav_status(status):

    flags = {
        "system_error": bool((status >> 0) & 1),
        "hybrid_mode": bool((status >> 1) & 1),
        "heading_valid": not bool((status >> 2) & 1),
        "altitude_valid": not bool((status >> 3) & 1),
        "velocity_valid": not bool((status >> 4) & 1),
        "depth_valid": not bool((status >> 5) & 1),
        "sound_velocity_valid": not bool((status >> 6) & 1),
        "temperature_valid": not bool((status >> 7) & 1),
        "position_valid": not bool((status >> 9) & 1),
        "utc_valid": not bool((status >> 10) & 1),
    }


    if (
        flags["hybrid_mode"] and
        flags["position_valid"] and
        flags["depth_valid"] and
        flags["altitude_valid"] and
        flags["heading_valid"]
    ):
        return True
    else:
        return False
    
# --------------------------------------------------
# WGS84 transformers
# --------------------------------------------------

lla_to_ecef = Transformer.from_crs(
    "EPSG:4979",   # WGS84 3D
    "EPSG:4978",   # ECEF
    always_xy=True
)

ecef_to_lla = Transformer.from_crs(
    "EPSG:4978",
    "EPSG:4979",
    always_xy=True
)


# --------------------------------------------------
# NED -> ECEF rotation
# --------------------------------------------------

def ned_to_ecef_rotation(lat_deg, lon_deg):

    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    sLat = np.sin(lat)
    cLat = np.cos(lat)

    sLon = np.sin(lon)
    cLon = np.cos(lon)

    return np.array([
        [-sLat*cLon, -sLon, -cLat*cLon],
        [-sLat*sLon,  cLon, -cLat*sLon],
        [ cLat,       0.0,  -sLat]
    ])


# --------------------------------------------------
# Transform point
# --------------------------------------------------

def offset_point(
        lat0,
        lon0,
        alt0,
        roll,
        pitch,
        heading,
        surge,
        sway,
        heave
):

    # origin ECEF
    X0, Y0, Z0 = lla_to_ecef.transform(
        lon0,
        lat0,
        alt0
    )

    # body -> NED
    R_body_to_ned = Rotation.from_euler(
        "ZYX",
        [heading, pitch, roll],
        degrees=True
    )

    body_offset = np.array([
        surge,
        sway,
        heave
    ])

    ned_offset = R_body_to_ned.apply(
        body_offset
    )

    # NED -> ECEF
    R_ned_ecef = ned_to_ecef_rotation(
        lat0,
        lon0
    )

    ecef_offset = R_ned_ecef @ ned_offset

    # translated point
    X = X0 + ecef_offset[0]
    Y = Y0 + ecef_offset[1]
    Z = Z0 + ecef_offset[2]

    # back to lat/lon/alt
    lon, lat, alt = ecef_to_lla.transform(
        X,
        Y,
        Z
    )

    return lat, lon, alt

def string_to_epoch(filename, stamp_format='xxxxxxxxxxxxxxxNNNxYYYYxMMxDDxhhmmssxfffuuuxx.xxx'):
    year = ""
    month = ""
    day = ""
    hour = ""
    minute = ""
    second = ""
    msecond = ""
    usecond = ""
    index = ""
    epoch = ""
    for n, f in zip(filename, stamp_format):
        if f == "Y":
            year += n
        if f == "M":
            month += n
        if f == "D":
            day += n
        if f == "h":
            hour += n
        if f == "m":
            minute += n
        if f == "s":
            second += n
        if f == "f":
            msecond += n
        if f == "u":
            usecond += n
        if f == "i":
            index += n
        if f == "e":
            epoch += n
    if not index and epoch == "":
        assert len(year) == 4, "Year in filename should have a length of 4"
        assert (
            len(month) == 2
        ), "Month in filename should have a length of \
            2"
        assert len(day) == 2, "Day in filename should have a length of 2"
        assert len(hour) == 2, "Hour in filename should have a length of 2"
        assert (
            len(minute) <= 2
        ), "Minute in filename should have a length \
            of 2"
        assert (
            len(second) <= 2
        ), "Second in filename should have a length \
            of 2"
        if msecond:
            assert (
                len(msecond) <= 3
            ), "Milliseconds in filename should \
                have a maximum length of 3"
        else:
            msecond = "0"
        if usecond:
            assert (
                len(usecond) <= 3
            ), "Microseconds in filename should \
                have a length of 3"
        else:
            usecond = "0"
        microsecond = int(msecond) * 1000 + int(usecond)

        date = datetime(
            int(year),
            int(month),
            int(day),
            int(hour),
            int(minute),
            int(second),
            microsecond,
        )
        stamp = float(calendar.timegm(date.timetuple()))
        return stamp + microsecond * 1e-6
    elif epoch != "":
        stamp = float(epoch)
        return stamp
    else:
        print('FilenameToDate specified using indexing')
        return None
        # if self.df is None:
        #     print(
        #         "FilenameToDate specified using indexing, but no \
        #         timestamp file has been provided or read."
        #     )
        #     print("Invalid timestamp format")
        # stamp = self.df["epoch_timestamp"][int(index)]
        # return stamp

def process_hnav(payload,bag_filename = None, dive_dir=os.getcwd(),start_time=None,end_time=None,force_overwrite=False):

    mission_file = dive_dir+'/mission.yaml'
    with open(mission_file, 'r') as f:
        mission = yaml.load(f, Loader=yaml.SafeLoader)

    vehicle_file = dive_dir+'/vehicle.yaml'
    with open(vehicle_file, 'r') as f:
        vehicle = yaml.load(f, Loader=yaml.SafeLoader)

    if payload == 'stills':
        camera_file = dive_dir+'/camera.yaml'
        with open(camera_file, 'r') as f:
            camera = yaml.load(f, Loader=yaml.SafeLoader)

    if bag_filename is None:
        nav_bag_paths = glob(dive_dir+'/nav/bags/*')
    else:
        nav_bag_paths = glob(dive_dir+'/nav/bags/'+bag_filename)

    if start_time is None:
        start_t = None
        end_t = None
    else:
        start_t = rospy.Time(datetime.strptime(start_time,'%Y%m%d %H%M%S').timestamp())
        end_t = rospy.Time(datetime.strptime(end_time,'%Y%m%d %H%M%S').timestamp())

    hnav_data = []
    for nav_bag_path in nav_bag_paths:
        bagfile = rosbag.Bag(nav_bag_path)
        print(f"bagfile date and time {datetime.fromtimestamp(bagfile.get_start_time(),UTC)} - {datetime.fromtimestamp(bagfile.get_end_time(),UTC)}")
        for topic,msg,t in bagfile.read_messages(topics=['/sparus2/sonardyne_sprintnav_ins/hnav'],start_time=start_t,end_time=end_t):
            hnav_data.append(msg)
        bagfile.close()

    hnav_df = pd.DataFrame([{**{'timestamp':y.header.stamp.secs + y.header.stamp.nsecs/10**9},
        **{x:getattr(y, x) for x in y.__slots__[1:]}} for y in hnav_data],index=[y.header.seq for y in hnav_data])
    hnav_df.sort_values('timestamp',ascending=True)

    hnav_df['altitude_valid'] = hnav_df['status'].apply(decode_hnav_status)
    hnav_df = hnav_df[hnav_df['altitude_valid']]

    hnav_df[['latitude_corr','longitude_corr','depth_corr']] = [*hnav_df.apply(lambda x: offset_point(x.latitude,x.longitude,x.depth*(-1),x.roll,x.pitch,x.heading,vehicle[payload]['surge_m'],vehicle[payload]['sway_m'],vehicle[payload]['heave_m']),axis=1)]
    hnav_df['depth_corr'] = hnav_df['depth_corr']*(-1)
    hnav_df['altitude_corr'] = hnav_df['altitude'] - (hnav_df['depth_corr'] - hnav_df['depth'])
    hnav_df['northings'] = hnav_df.apply(lambda x: (np.asarray(lla_to_ecef.transform(x.longitude_corr,x.latitude_corr))-np.asarray(lla_to_ecef.transform(mission['origin']['longitude'],mission['origin']['latitude'])))[0]*(-1),axis=1)
    hnav_df['eastings'] = hnav_df.apply(lambda x: (np.asarray(lla_to_ecef.transform(x.longitude_corr,x.latitude_corr))-np.asarray(lla_to_ecef.transform(mission['origin']['longitude'],mission['origin']['latitude'])))[1],axis=1)

    if payload == 'stills':

        p = Path(dive_dir+'/'+camera['cameras'][0]['path'])

        im_paths = glob(str(p)+'/**/*.tif',recursive=True)
        im_paths= sorted(im_paths)

        rel_paths = [str(Path(x).relative_to(dive_dir)) for x in im_paths]

        nav_df = pd.DataFrame({'relative_path':[None],
        'northing [m]':[None],
        'easting [m]':[None],
        'depth [m]':[None],
        'roll [deg]':[None],
        'pitch [deg]':[None],
        'heading [deg]':[None],
        'altitude [m]':[None],
        'timestamp [s]':[None],
        'latitude [deg]':[None],
        'longitude [deg]':[None],
        'x_velocity [m/s]':[None],
        'y_velocity [m/s]':[None],
        'z_velocity [m/s]':[None],},index=range(len(rel_paths)))

        nav_df['relative_path'] = rel_paths
        nav_df['timestamp [s]'] = nav_df['relative_path'].apply(lambda x: string_to_epoch(x.split('/')[-1]))

        if nav_df.iloc[0]['timestamp [s]']<hnav_df.iloc[0]['timestamp']:
            print(f"WARNING: Removing {len(nav_df[nav_df['timestamp [s]'] < hnav_df.iloc[0]['timestamp']])} images which start before nav data ({datetime.strftime(datetime.fromtimestamp(hnav_df.iloc[0]['timestamp'],UTC),'%d/%m/%Y %H:%M:%S')})")
        if nav_df.iloc[-1]['timestamp [s]']>hnav_df.iloc[-1]['timestamp']:
            print(f"WARNING: Removing {len(nav_df[nav_df['timestamp [s]'] > hnav_df.iloc[-1]['timestamp']])} images which end after nav data ({datetime.strftime(datetime.fromtimestamp(hnav_df.iloc[-1]['timestamp'],UTC),'%d/%m/%Y %H:%M:%S')})")
        nav_df = nav_df[(nav_df['timestamp [s]'] > hnav_df.iloc[0]['timestamp'])&(nav_df['timestamp [s]'] < hnav_df.iloc[-1]['timestamp'])]

        nav_df['northing [m]'] = np.interp(nav_df['timestamp [s]'],hnav_df['timestamp'],hnav_df['northings'])
        nav_df['easting [m]'] = np.interp(nav_df['timestamp [s]'],hnav_df['timestamp'],hnav_df['eastings'])
        nav_df['depth [m]'] = np.interp(nav_df['timestamp [s]'],hnav_df['timestamp'],hnav_df['depth_corr'])
        nav_df['roll [deg]'] = np.interp(nav_df['timestamp [s]'],hnav_df['timestamp'],hnav_df['roll'])
        nav_df['pitch [deg]'] = np.interp(nav_df['timestamp [s]'],hnav_df['timestamp'],hnav_df['pitch'])
        nav_df['heading [deg]'] = np.interp(nav_df['timestamp [s]'],hnav_df['timestamp'],hnav_df['heading'])
        nav_df['altitude [m]'] = np.interp(nav_df['timestamp [s]'],hnav_df['timestamp'],hnav_df['altitude_corr'])
        nav_df['latitude [deg]'] = np.interp(nav_df['timestamp [s]'],hnav_df['timestamp'],hnav_df['latitude_corr'])
        nav_df['longitude [deg]'] = np.interp(nav_df['timestamp [s]'],hnav_df['timestamp'],hnav_df['longitude_corr'])
        nav_df['x_velocity [m/s]'] = np.interp(nav_df['timestamp [s]'],hnav_df['timestamp'],hnav_df['forward_velocity'])
        nav_df['y_velocity [m/s]'] = np.interp(nav_df['timestamp [s]'],hnav_df['timestamp'],hnav_df['starboard_velocity'])
        nav_df['z_velocity [m/s]'] = np.interp(nav_df['timestamp [s]'],hnav_df['timestamp'],hnav_df['down_velocity'])

    else:
        nav_df = pd.DataFrame({
        'timestamp':hnav_df['timestamp'],
        'northing [m]':hnav_df['northings'],
        'easting [m]':hnav_df['eastings'],
        'depth [m]':hnav_df['depth_corr'],
        'roll [deg]':hnav_df['roll'],
        'pitch [deg]':hnav_df['pitch'],
        'heading [deg]':hnav_df['heading'],
        'altitude [m]':hnav_df['altitude_corr'],
        'latitude [deg]':hnav_df['latitude_corr'],
        'longitude [deg]':hnav_df['longitude_corr'],
        'x_velocity [m/s]':hnav_df['forward_velocity'],
        'y_velocity [m/s]':hnav_df['starboard_velocity'],
        'z_velocity [m/s]':hnav_df['down_velocity']
        })

    output_dir = f"json_renav_{datetime.strftime(datetime.fromtimestamp(nav_df.iloc[0]['timestamp [s]'],UTC),'%Y%m%d_%H%M%S')}_{datetime.strftime(datetime.fromtimestamp(nav_df.iloc[-1]['timestamp [s]'],UTC),'%Y%m%d_%H%M%S')}"
    output_dir = output_dir+f"/csv/ekf/auv_ekf_centre.csv" if payload == 'dvl' else output_dir+f"/csv/ekf/auv_ekf_{payload}.csv"

    if os.path.exists(dive_dir.replace('raw','processed')+'/'+output_dir):
        if not force_overwrite:
            print(f"ERROR: Navigation solution already exists, to overwrite use tag -f")
            return
        
    os.makedirs('/'.join((dive_dir.replace('raw','processed')+'/'+output_dir).split('/')[:-1]),exist_ok=True)
        
    nav_df.to_csv(dive_dir.replace('raw','processed')+'/'+output_dir,index=False)
    return

  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Use SprintNav Hybrid navigation solution as AUV position." 
            "Accounts for offsets of payloads to SprintNav on the AUV body and interpolates "
            "navigation solution to image timestamps. "
            "Output format follows auv_nav process output."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("payload", help="Payload to adjust positional offsets for and save navigation solution")
    parser.add_argument(
        "-b",
        "--bagfiles",
        help="Bagfile(s) name or partial names containing SprintNav Hybrid navigation (default: None"
                "to find all bagfiles in nav directory) e.g. sparus2_2026-06-24-*.bag",
        default=None,
    )
    parser.add_argument(
        "-d",
        "--dive_dir",
        help="Directory where raw dive data is stored (default:current working directory)",
        default=os.getcwd(),
    )
    parser.add_argument(
        "-s",
        "--start_time",
        help="Date and time to start processing dive data for, YYYYMMDD HHMMSS (default: None)",
        default=None,
    )
    parser.add_argument(
        "-e",
        "--end_time",
        help="Date and time to end processing dive data for, YYYYMMDD HHMMSS (default: None)",
        default=None,
    )
    parser.add_argument(
        "-f", "--force", help="Overwrite output file if it exists", action="store_true"
    )
    a = parser.parse_args()

    process_hnav(a.payload,a.bagfiles, a.dive_dir,a.start_time,a.end_time,a.force)


