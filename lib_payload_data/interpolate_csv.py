import pandas as pd
import os, sys
# from lib_calculus import interpolate

# class interpolate_csv:
# 	# take into account vehicle yaml too, with some offset!
# 	# takes in nav_csv file and target csv file, interpolate the timestamps accordingly and add nav data columns to target csv
# 	# only special for chemical, interpolate based on 
# 	# looks at csv output of extract, and interpolate 
# 	def __init__(self, csv_input, csv_nav_centre, file_output):
# 	    # def load_input_df(csv_input):
# 	    #     file_extension = os.path.splitext(csv_input)[1]
# 	    #     if file_extension == '.csv':
# 	    #     	df = pd.read_csv(csv_input)
# 	    #     elif file_extension == '.xlsx':
# 	    #     	df = pd.read_excel(csv_input)
# 	    #     return df

def interpolate(x_query, x_lower, x_upper, y_lower, y_upper):
	if x_upper == x_lower:
		y_query=y_lower
	else:
		y_query=(y_upper-y_lower)/(x_upper-x_lower)*(x_query-x_lower)+y_lower
	return y_query


def interpolate_csv(csv_input, csv_nav_centre, file_output): # (self,
    # read input csv
    input_df = pd.read_csv(csv_input)
    nav_df = pd.read_csv(csv_nav_centre)

    input_timestamps = input_df['epoch_time']
    input_latitudes = []
    input_longitudes = []

    nav_timestamps = nav_df['Timestamp']
    nav_latitudes = nav_df[' Latitude [deg]']
    nav_longitudes = nav_df[' Longitude [deg]']

    # make a function separately for doing this interpolation? Maybe in sensor_class, a general one, and a specifc one for each class that calls on this general one
    nav_timestamps_index_tracker = 0
    input_timestamps_index_tracker = 0
    # make sure nav_timestamps_index_tracker is pointing to the element that is one step behind (or equal)
    if nav_timestamps[nav_timestamps_index_tracker] < input_timestamps[0]:
    	while nav_timestamps[nav_timestamps_index_tracker+1] < input_timestamps[0]:
    		nav_timestamps_index_tracker += 1
    elif nav_timestamps[nav_timestamps_index_tracker] > input_timestamps[0]:
    	while nav_timestamps[nav_timestamps_index_tracker] > input_timestamps[input_timestamps_index_tracker]:
    		input_timestamps_index_tracker += 1
    
    # check if nav data ends earlier than payload data
    input_end_index_tracker = len(input_timestamps) -1
    if nav_timestamps[len(nav_timestamps)-1] < input_timestamps[len(input_timestamps)-1]:
    	while nav_timestamps[len(nav_timestamps)-1] < input_timestamps[input_end_index_tracker]:
    		input_end_index_tracker -= 1

    new_input_df = input_df.iloc[input_timestamps_index_tracker:input_end_index_tracker+1]

    for timestamp in new_input_df['epoch_time']:
    	try:
    		while nav_timestamps[nav_timestamps_index_tracker+1] < timestamp:
    			nav_timestamps_index_tracker += 1
    		input_latitudes.append(interpolate(timestamp, nav_timestamps[nav_timestamps_index_tracker], nav_timestamps[nav_timestamps_index_tracker+1], nav_latitudes[nav_timestamps_index_tracker], nav_latitudes[nav_timestamps_index_tracker+1]))
    		input_longitudes.append(interpolate(timestamp, nav_timestamps[nav_timestamps_index_tracker], nav_timestamps[nav_timestamps_index_tracker+1], nav_longitudes[nav_timestamps_index_tracker], nav_longitudes[nav_timestamps_index_tracker+1]))
    	except IndexError:
    		print("Error! Interpolation incomplete, navigation data insufficient to cover remaining of payload data! Check both csv file timestamps")
    		sys.exit() # if this happens make it output another csv file with just the overlapping data

    new_input_df['Latitude [deg]'] = input_latitudes
    new_input_df['Longitude [deg]'] = input_longitudes

    if file_output != 0:
        csv_output = file_output
    else:
        sub_path = csv_input.split('/') # os.sep)
        outpath = sub_path[0]
        for i in range(1, len(sub_path)):
            if sub_path[i]=='raw':
                sub_path[i] = 'processed'
            outpath = outpath + os.sep + sub_path[i]
            # make the new directories after 'processed' if it doesnt already exist
            if os.path.isdir(outpath) == 0:
                try:
                    os.mkdir(outpath)
                except Exception as e:
                    print("Warning:",e)
        csv_output = outpath #os.path.join(*sub_path)

    new_input_df.to_csv(csv_output, header=True, index=False)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Error: not enough arguments')
        # syntax_error()
    else:
        # read in filepath, start time and finish time from function call
        flag_i = False
        flag_c = False
        flag_o = False
        output_filepath = 0
        for i in range(len(sys.argv)):
            option = sys.argv[i]
            if option == "-i":
                input_filepath = sys.argv[i+1]
                flag_i = True
            if option == "-c":
                nav_filepath = sys.argv[i+1]
                flag_c = True
            if option == "-o":
                output_filepath = sys.argv[i+1]
                flag_o = True
            # elif option == "-h":
            #   print ('usage: mosaic_unsupervised_clustering.py -i <filepath containing csv_config.yaml>')
        if flag_i == flag_c == flag_o == True:
            interpolate_csv(input_filepath, nav_filepath, output_filepath)
        else:
            print ('Error! Not enough input.')

# interpolate_csv('Z:/cruise_data/raw/2018/fk180731/tuna_sand/20180806_210350_ts_un6k/payload/8_6_TS-1-2_Cal.csv', "Z:/cruise_data/processed/2018/fk180731/tuna_sand/20180806_210350_ts_un6k/json_renav_20180806_123000_20180806_173000/csv/dead_reckoning/auv_dr_centre.csv", 0)