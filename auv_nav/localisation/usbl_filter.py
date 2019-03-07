
from auv_nav.tools.interpolate import interpolate

# filter usbl data outliers
# interpolate to find the appropriate depth to compute seafloor depth for each
# altitude measurement


# not very robust, need to define max_speed carefully. Maybe use kmeans
# clustering? features = gradient/value compared to nearest few neighbours,
# k = 2, select group with least std/larger value
# first filter by depth as bad USBL data tends to get the depth wrong. This
# should be a function of the range to the target. The usbl and vehicle sensor
# depths should be within 2 std (95% prob) of each other.
# If more than that apart reject

def depth_filter(usbl, depth, depth_std, sigma_factor):
    depth_difference = abs(usbl.depth - depth)
    depth_uncertainty_envelope = abs(usbl.depth_std) + abs(depth_std)
    return depth_difference <= sigma_factor*depth_uncertainty_envelope


# the 2 usbl readings should be within maximum possible distance traveled by
# the vehicle at max speed taking into account the 2 std uncertainty (95%) in
# the USBL lateral measurements
def distance_filter(usbl1, usbl2, sigma_factor, max_auv_speed):
    lateral_distance = (
        (usbl1.northings - usbl2.northings)**2
        + (usbl1.eastings - usbl2.eastings)**2)**0.5
    time_difference = abs(usbl1.epoch_timestamp
                          - usbl2.epoch_timestamp)
    lateral_distance_uncertainty_envelope = ((
        ((usbl1.northings_std)**2
         + (usbl1.eastings_std)**2)**0.5
        + ((usbl2.northings_std)**2
           + (usbl2.eastings_std)**2)**0.5)
        + time_difference * max_auv_speed)

    distance_envelope = sigma_factor*lateral_distance_uncertainty_envelope
    return lateral_distance <= distance_envelope


def usbl_filter(usbl_list, depth_list, sigma_factor, max_auv_speed):
    depth_interpolated = []
    depth_std_interpolated = []
    usbl_temp_list = []

    j = 0
    for i in range(len(usbl_list)):
        while j < len(depth_list)-1 and depth_list[j].epoch_timestamp < usbl_list[i].epoch_timestamp:
            j = j+1
        if j >= 1:
            depth_interpolated.append(interpolate(
                usbl_list[i].epoch_timestamp,
                depth_list[j - 1].epoch_timestamp,
                depth_list[j].epoch_timestamp,
                depth_list[j - 1].depth,
                depth_list[j].depth))
            depth_std_interpolated.append(interpolate(
                usbl_list[i].epoch_timestamp,
                depth_list[j - 1].epoch_timestamp,
                depth_list[j].epoch_timestamp,
                depth_list[j - 1].depth_std,
                depth_list[j].depth_std))

    for i in range(len(depth_interpolated)):
        if depth_filter(usbl_list[i],
                        depth_interpolated[i],
                        depth_std_interpolated[i],
                        sigma_factor):
            usbl_temp_list.append(usbl_list[i])
        i += 1
    usbl_list = usbl_temp_list
    usbl_temp_list = []

    continuity_condition = 2
    # want to check continuity over several readings to get rid of just
    # outliers and not good data around them. Current approach has a lot of
    # redundancy and can be improved when someone has the bandwidth
    i = continuity_condition
    n = -continuity_condition
    while i < len(usbl_list)-continuity_condition:
        if distance_filter(usbl_list[i],
                           usbl_list[i + n],
                           sigma_factor,
                           max_auv_speed):
            n += 1
            if n is 0:
                n += 1
            if n > continuity_condition:
                usbl_temp_list.append(usbl_list[i])
                n = -continuity_condition
                i += 1
        else:
            n = -continuity_condition
            i += 1
    return usbl_temp_list
