
from auv_nav.tools.interpolate import interpolate

# filter usbl data outliers
# interpolate to find the appropriate depth to compute seafloor depth for each altitude measurement


# depth_filter_flag):
def usbl_filter(usbl_list, depth_list, sigma_factor, max_auv_speed, ftype):
    depth_interpolated = []
    depth_std_interpolated = []
    usbl_temp_list = []

    j = 0

    for i in range(len(usbl_list)):
        while j < len(depth_list)-1 and depth_list[j].epoch_timestamp < usbl_list[i].epoch_timestamp:
            j = j+1

        if j >= 1:
            depth_interpolated.append(interpolate(
                usbl_list[i].epoch_timestamp, depth_list[j-1].epoch_timestamp, depth_list[j].epoch_timestamp, depth_list[j-1].depth, depth_list[j].depth))
            depth_std_interpolated.append(interpolate(
                usbl_list[i].epoch_timestamp, depth_list[j-1].epoch_timestamp, depth_list[j].epoch_timestamp, depth_list[j-1].depth_std, depth_list[j].depth_std))

    # not very robust, need to define max_speed carefully. Maybe use kmeans clustering? features = gradient/value compared to nearest few neighbours, k = 2, select group with least std/larger value?

    # first filter by depth as bad USBL data tends to get the depth wrong. This should be a function of the range to the target.

    # the usbl and vehicle sensor depths should be within 2 std (95% prob) of each other. If more than that apart reject
    def depth_filter(i, ftype):
        j = 0

        depth_difference = abs(usbl_list[i].depth-depth_interpolated[i])
        if ftype == 'oplab':
            depth_uncertainty_envelope = abs(
                usbl_list[i].depth_std)+abs(depth_std_interpolated[i])
            if depth_difference <= sigma_factor*depth_uncertainty_envelope:
                return True
            else:
                return False
        elif ftype == 'acfr':
            if depth_difference <= 10:
                return True
            else:
                return False

    def distance_filter(i, n):
        lateral_distance = ((usbl_list[i].northings - usbl_list[i+n].northings)**2 + (
            usbl_list[i].eastings - usbl_list[i+n].eastings)**2)**0.5
        lateral_distance_uncertainty_envelope = (((usbl_list[i].northings_std)**2+(usbl_list[i].eastings_std)**2)**0.5+((usbl_list[i+n].northings_std)**2+(
            usbl_list[i+n].eastings_std)**2)**0.5)+abs(usbl_list[i].epoch_timestamp-usbl_list[i+n].epoch_timestamp)*max_auv_speed

        if lateral_distance <= sigma_factor*lateral_distance_uncertainty_envelope:
            return True
        else:
            return False

    # if depth_filter_flag:
    for i in range(len(depth_interpolated)):
        if depth_filter(i, ftype) is True:
            usbl_temp_list.append(usbl_list[i])
        i += 1

    # print("Length of original usbl list = {}, Length of depth filtered usbl list = {}".format(len(usbl_list), len(usbl_temp_list)))
    usbl_list = usbl_temp_list
    usbl_temp_list = []

    # the 2 usbl readings should be within maximum possible distance travelled by the vehicle at max speed taking into account the 2 std uncertainty (95%) in the USBL lateral measurements

    continuity_condition = 2
    # want to check continuity over several readings to get rid of just outliers and not good data around them. Curent approach has a lot of redundncy and can be improved when someone has the bandwidth
    i = continuity_condition
    n = -continuity_condition
    while i < len(usbl_list)-continuity_condition:

        if distance_filter(i, n) is False:
            n = -continuity_condition
            i += 1
        else:
            n += 1
            if n is 0:
                n += 1
            if n > continuity_condition:
                usbl_temp_list.append(usbl_list[i])
                n = -continuity_condition
                i += 1

    # print("Length of depth filtered usbl list = {}, Length of depth and distance filtered usbl list = {}".format(len(usbl_list), len(usbl_temp_list)))
    # usbl_list = usbl_temp_list

    return usbl_temp_list

    # def speed(ii, n):
    #    value = abs((usbl_list[ii].northings - usbl_list[ii-n].northings)**2 + (usbl_list[ii].eastings - usbl_list[ii-n].eastings)**2/(usbl_list[ii].epoch_timestamp-usbl_list[ii-n].epoch_timestamp))
    #    return value
    # to pick a good starting point. Any thing that says auv is at 5m/s reject
    # i=2
    # while not speed(i, -2)<max_auv_speed and speed(i, -1)<max_auv_speed and speed(i, 1)<max_auv_speed and speed(i, 2)<max_auv_speed:
    #    i+= 1
    # i-=2
    # usbl_temp_list.append(usbl_list[i])
    # while i+2 < len(usbl_list):
    #    if speed(i, -1) < max_auv_speed:
    #        i+=1
    #        usbl_temp_list.append(usbl_list[i])
    #    else:
    #        i+=2
    #print ("Length of original usbl list = {}, Length of filtered usbl list = {}".format(len(usbl_list), len(usbl_temp_list)))
