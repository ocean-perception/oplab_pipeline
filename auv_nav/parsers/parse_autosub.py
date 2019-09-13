import scipy.io as spio
from auv_nav.sensors import BodyVelocity
from auv_nav.sensors import Orientation, Depth, Altitude
from auv_nav.sensors import Category
from auv_nav.tools.folder_structure import get_raw_folder
from auv_nav.tools.console import Console


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def parse_autosub(mission,
                  vehicle,
                  category,
                  ftype,
                  outpath,
                  fileoutname):
    # parser meta data
    class_string = 'measurement'
    sensor_string = 'autosub'
    category = category
    output_format = ftype
    filename = mission.velocity.filename
    filepath = mission.velocity.filepath

    # autosub std models
    depth_std_factor = mission.depth.std_factor
    velocity_std_factor = mission.velocity.std_factor
    velocity_std_offset = mission.velocity.std_offset
    orientation_std_offset = mission.orientation.std_offset
    altitude_std_factor = mission.altitude.std_factor
    headingoffset = vehicle.dvl.yaw

    body_velocity = BodyVelocity(velocity_std_factor,
                                 velocity_std_offset,
                                 headingoffset)
    orientation = Orientation(headingoffset, orientation_std_offset)
    depth = Depth(depth_std_factor)
    altitude = Altitude(altitude_std_factor)

    body_velocity.sensor_string = sensor_string
    orientation.sensor_string = sensor_string
    depth.sensor_string = sensor_string
    altitude.sensor_string = sensor_string

    path = get_raw_folder(outpath / '..' / filepath / filename)

    alr_log = loadmat(str(path))
    alr_acdp = alr_log['missionData']['ADCPbin00']
    alr_ins = alr_log['missionData']['INSAttitude']
    alr_ctd = alr_log['missionData']['CTD']
    alr_acdp_log1 = alr_log['missionData']['ADCPLog_1']

    data_list = []
    if category == Category.VELOCITY:
        Console.info('...... parsing autosub velocity')
        for i in range(len(alr_acdp['eTime'])):
            body_velocity.from_autosub(alr_acdp, i)
            data = body_velocity.export(output_format)
            data_list.append(data)
    if category == Category.ORIENTATION:
        Console.info('...... parsing autosub orientation')
        for i in range(len(alr_ins['eTime'])):
            orientation.from_autosub(alr_ins, i)
            data = orientation.export(output_format)
            data_list.append(data)
    if category == Category.DEPTH:
        Console.info('...... parsing autosub depth')
        for i in range(len(alr_ctd['eTime'])):
            depth.from_autosub(alr_ctd, i)
            data = depth.export(output_format)
            data_list.append(data)
    if category == Category.ALTITUDE:
        Console.info('...... parsing autosub altitude')
        for i in range(len(alr_acdp_log1['eTime'])):
            altitude.from_autosub(alr_acdp_log1, i)
            data = altitude.export(output_format)
            data_list.append(data)
    return data_list
