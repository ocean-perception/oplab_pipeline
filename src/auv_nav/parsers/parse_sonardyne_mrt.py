from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field
from pathlib import Path

from auv_nav.sensors import Category
from auv_nav.tools.latlon_wgs84 import latlon_to_metres, metres_to_latlon
from auv_nav.tools.time_conversions import read_timezone
from oplab import Console, get_raw_folder


@dataclass
class Time:
    display_time: str
    system_time: str


@dataclass
class MRTBase:  # 9
    obs_uid: str
    source_uid: str
    source_name: str
    parent_uid: str
    parent_name: str
    assoc_uid: str
    time_stamp: str
    filter_uid: str
    fix_number: str
    epoch_timestamp: float = field(init=False, default=None)

    def __post_init__(self):
        try:
            self.epoch_timestamp = datetime.datetime.strptime(
                self.time_stamp, "%Y-%m-%d %H:%M:%S.%f"
            ).timestamp()
        except ValueError:
            pass


@dataclass
class AccelerationObs(MRTBase):
    x_acceleration: float
    y_acceleration: float
    z_acceleration: float
    x_valid: bool
    y_valid: bool
    z_valid: bool


@dataclass
class PitchRollObs(MRTBase):
    pitch: float
    roll: float
    pitch_valid: bool
    roll_valid: bool


@dataclass
class PitchRollHeadingObs(MRTBase):
    pitch: float
    roll: float
    heading: float
    pitch_valid: bool
    roll_valid: bool
    heading_valid: bool


@dataclass
class DisplacementObs(MRTBase):
    surge: float
    heave: float
    sway: float
    surge_valid: bool
    heave_valid: bool
    sway_valid: bool


@dataclass
class DepthObs(MRTBase):
    depth: float
    valid: bool


@dataclass
class HeadingObs(MRTBase):
    heading: float
    valid: bool


@dataclass
class GPSLatLongObs(MRTBase):
    latitude: float
    longitude: float
    altitude: float
    latitude_valid: bool
    longitude_valid: bool
    altitude_valid: bool
    quality: float
    hdop: float
    satellites: int
    utc_time: int
    geoidal_separation: float


@dataclass
class LatLongObs(MRTBase):
    latitude: float
    longitude: float
    altitude: float
    latitude_valid: bool
    longitude_valid: bool
    altitude_valid: bool

    closest_ship_obs: LatLongObs = field(init=False, default=None)
    closest_heading_obs: HeadingObs = field(init=False, default=None)


@dataclass
class GridPositionObs(MRTBase):
    x: float
    y: float
    z: float
    valid_x: bool
    valid_y: bool
    valid_z: bool
    horizontal_position_sd: float
    vertical_position_sd: float
    utm_zone: str


@dataclass
class BeaconVectorObs:
    xc: str
    d_bv: str
    snr: str
    ifl: str


@dataclass
class RangeObs(MRTBase):
    travel_time: float
    travel_time_valid: bool
    node1id: str
    tat1: int
    node2id: str
    tat2: int
    node3id: str
    tat3: int
    node4id: str
    tat4: int
    signal_level: int
    signal_noise_ratio: float
    doppler: int
    pulse_length: int
    range_sd: float
    cross_correlation: float
    decibel_voltage: int
    interference_level: int
    time_of_transmission: str


@dataclass
class USBLVectorObs(MRTBase):  # 9 + 10
    jx: float
    jy: float
    jz: float
    valid_jx: bool
    valid_jy: bool
    valid_jz: bool
    frequency: int
    qj_sum: int
    n_good_samples: int
    direction_sd: float


@dataclass
class USBLRangeObs(MRTBase):
    range_obs: RangeObs
    beacon_vector_obs: BeaconVectorObs
    usbl_vector_obs: USBLVectorObs
    sound_velocity: str
    calibrated: str
    surface_sv: str
    column_sv: str
    usbl_vector_obs2: USBLVectorObs
    internal_pitch: str
    internal_roll: str


@dataclass
class Alarm(MRTBase):
    type_id: str
    severity: str
    state: str
    summary: str
    description: str
    url: str
    origin: str
    category: str


@dataclass
class AlarmStateChange(MRTBase):
    alarm_uid: str
    alarm_type_id: str
    severity: str
    prev_state: str
    new_state: str
    owner: str
    purged: str


@dataclass
class VelocityObs(MRTBase):
    velocity_x: str
    velocity_y: str
    velocity_z: str
    validity_x: bool
    validity_y: bool
    validity_z: bool


@dataclass
class Event(MRTBase):
    type: str
    message: str


@dataclass
class EstimateTrackerMetrics(MRTBase):
    estimate_type: str
    estimate_uid: str
    filter_uid2: str
    weight: str
    predicted: str
    mde: str
    w_test: str


def parse_mrt_csv(filename):
    filename = Path(filename)
    with filename.open("r") as f:
        observations = []
        # Loop through each line
        for i, line in enumerate(f):
            if i > 0 and i < 36:
                continue
            # Split the line into a list of values separated by commas.
            # Allow two commas to be next to each other (empty value)
            # Allow a comma followed by a space - this is not a separator
            raw_values = line.split(",")
            # Look for any entry starting in a space, and join it to the previous entry
            values = []
            for raw_value in raw_values:
                if values is None:
                    values.append(raw_value)
                if raw_value.startswith(" "):
                    values[-1] += "," + raw_value
                else:
                    values.append(raw_value)
            # Remove the newline character from the last value
            values[-1] = values[-1].strip()
            # Check the type of observation
            if values[0] == "AccelerationObs":
                obs = AccelerationObs(*values[1:])
            elif values[0] == "PitchRollObs":
                obs = PitchRollObs(*values[1:])
            elif values[0] == "PitchRollHeadingObs":
                obs = PitchRollHeadingObs(*values[1:])
            elif values[0] == "DisplacementObs":
                obs = DisplacementObs(*values[1:])
            elif values[0] == "DepthObs":
                obs = DepthObs(*values[1:])
            elif values[0] == "HeadingObs":
                obs = HeadingObs(*values[1:])
            elif values[0] == "GPSLatLongObs":
                obs = GPSLatLongObs(*values[1:])
            elif values[0] == "LatLongObs":
                obs = LatLongObs(*values[1:])
            elif values[0] == "GridPositionObs":
                obs = GridPositionObs(*values[1:])
            elif values[0] == "USBLVectorObs":
                obs = USBLVectorObs(*values[1:])
            elif values[0] == "USBLRangeObs":
                range_obs = RangeObs(*values[11:39])
                beacon_vector_obs = BeaconVectorObs(*values[40:44])
                usbl_vector_obs = USBLVectorObs(*values[45:64])
                usbl_vector_obs2 = USBLVectorObs(*values[70:89])
                new_values = (
                    values[1:10]
                    + [range_obs]
                    + [beacon_vector_obs]
                    + [usbl_vector_obs]
                    + values[64:69]
                    + [usbl_vector_obs2]
                    + values[89:91]
                )
                obs = USBLRangeObs(*new_values)
            elif values[0] == "VelocityObs":
                obs = VelocityObs(*values[1:])
            elif values[0] == "Event":
                obs = Event(*values[1:])
            elif values[0] == "EstimateTrackerMetrics":
                obs = EstimateTrackerMetrics(*values[1:])
            elif values[0] == "Time":
                obs = Time(values[2], values[4])
            elif values[0] == "Alarm":
                obs = Alarm(*values[1:])
            elif values[0] == "AlarmStateChange":
                obs = AlarmStateChange(*values[1:])
            elif values[0] == "JobData":
                # Ignore this
                pass
            else:
                print(f"Unknown observation type: {values[0]}")
                continue
            observations.append(obs)
        return observations


def find_closest_observations(observations):
    auv_observations = []
    ship_observations = []
    heading_observations = []

    for obs in observations:
        if type(obs) == LatLongObs:
            if obs.source_name == "AUV 1":
                auv_observations.append(obs)
            elif obs.source_name == "GNSS 1":
                ship_observations.append(obs)
        elif type(obs) == PitchRollHeadingObs:
            if obs.source_name == "MRT-USBL":
                heading_observations.append(obs)

    # Sort the observations by time
    auv_observations.sort(key=lambda x: x.epoch_timestamp)
    ship_observations.sort(key=lambda x: x.epoch_timestamp)
    heading_observations.sort(key=lambda x: x.epoch_timestamp)

    # Find the closest ship observation to each AUV observation

    ship_idx = 0
    hdg_idx = 0

    for auv_obs in auv_observations:
        stamp = auv_obs.epoch_timestamp

        if ship_idx >= len(ship_observations):
            break
        if hdg_idx >= len(heading_observations):
            break

        ship_stamp = ship_observations[ship_idx].epoch_timestamp
        hdg_stamp = heading_observations[hdg_idx].epoch_timestamp

        while ship_stamp < stamp and ship_idx < len(ship_observations) - 1:
            ship_idx += 1
            ship_stamp = ship_observations[ship_idx].epoch_timestamp

        while hdg_stamp < stamp and hdg_idx < len(heading_observations) - 1:
            hdg_idx += 1
            hdg_stamp = heading_observations[hdg_idx].epoch_timestamp

        if ship_idx < len(ship_observations):
            auv_obs.closest_ship_obs = ship_observations[ship_idx]
        if hdg_idx < len(heading_observations):
            auv_obs.closest_heading_obs = heading_observations[hdg_idx]
    return auv_observations


def parse_sonardyne_mrt(mission, vehicle, category, ftype, outpath):
    Console.info("... parsing Sonardyne MRT")

    # parser meta data
    class_string = "measurement"
    sensor_string = "sonardyne_mrt"
    frame_string = "inertial"

    timezone = mission.usbl.timezone
    timeoffset = mission.usbl.timeoffset_s

    timezone_offset_h = read_timezone(timezone)
    timeoffset_s = -timezone_offset_h * 60 * 60 + timeoffset

    filepath = mission.usbl.filepath
    filename = mission.usbl.filename
    # usbl_id = mission.usbl.label
    latitude_reference = mission.origin.latitude
    longitude_reference = mission.origin.longitude

    distance_std_factor = mission.usbl.std_factor
    distance_std_offset = mission.usbl.std_offset

    # parse data
    filename = get_raw_folder(outpath / ".." / filepath / filename)

    # Find all files with the same filename ending in "_1, _2, _3, etc"
    filename = Path(filename)
    if not filename.exists():
        Console.error(f"File {filename} does not exist")
        return None
    base_name = filename.stem
    files = filename.parent.glob(base_name + "_*")
    files = sorted(files)

    all_observations = []
    for filename in files:
        observations = parse_mrt_csv(filename)
        all_observations.extend(observations)
    auv_observations = find_closest_observations(all_observations)

    data_list = []

    # Calculate the distance between the two
    for auv_obs in auv_observations:
        latitude = float(auv_obs.latitude)
        longitude = float(auv_obs.longitude)
        depth = -float(auv_obs.altitude)
        latitude_ship = float(auv_obs.closest_ship_obs.latitude)
        longitude_ship = float(auv_obs.closest_ship_obs.longitude)
        lateral_distance, bearing = latlon_to_metres(
            latitude, longitude, latitude_ship, longitude_ship
        )
        distance = math.sqrt(lateral_distance * lateral_distance + depth * depth)

        # calculate in metres from reference
        lateral_distance_ship, bearing_ship = latlon_to_metres(
            latitude_ship,
            longitude_ship,
            latitude_reference,
            longitude_reference,
        )
        eastings_ship = math.sin(bearing_ship * math.pi / 180.0) * lateral_distance_ship
        northings_ship = (
            math.cos(bearing_ship * math.pi / 180.0) * lateral_distance_ship
        )

        # calculate in metres from reference
        lateral_distance_ship, bearing_ship = latlon_to_metres(
            latitude,
            longitude,
            latitude_reference,
            longitude_reference,
        )
        eastings_target = (
            math.sin(bearing_ship * math.pi / 180.0) * lateral_distance_ship
        )
        northings_target = (
            math.cos(bearing_ship * math.pi / 180.0) * lateral_distance_ship
        )

        distance_std = distance_std_factor * distance + distance_std_offset

        # determine uncertainty in terms of latitude and longitude
        latitude_offset, longitude_offset = metres_to_latlon(
            abs(latitude),
            abs(longitude),
            distance_std,
            distance_std,
        )

        latitude_std = abs(abs(latitude) - latitude_offset)
        longitude_std = abs(abs(longitude) - longitude_offset)

        data = {
            "epoch_timestamp": auv_obs.epoch_timestamp + timeoffset_s,
            "class": class_string,
            "sensor": sensor_string,
            "frame": frame_string,
            "category": Category.USBL,
            "data_ship": [
                {
                    "latitude": float(latitude_ship),
                    "longitude": float(longitude_ship),
                },
                {
                    "northings": float(northings_ship),
                    "eastings": float(eastings_ship),
                },
                {"heading": float(auv_obs.closest_heading_obs.heading)},
            ],
            "data_target": [
                {
                    "latitude": float(latitude),
                    "latitude_std": float(latitude_std),
                },
                {
                    "longitude": float(longitude),
                    "longitude_std": float(longitude_std),
                },
                {
                    "northings": float(northings_target),
                    "northings_std": float(distance_std),
                },
                {
                    "eastings": float(eastings_target),
                    "eastings_std": float(distance_std),
                },
                {
                    "depth": float(depth),
                    "depth_std": float(distance_std),
                },
                {"distance_to_ship": float(distance)},
            ],
        }
        data_list.append(data)
    return data_list
