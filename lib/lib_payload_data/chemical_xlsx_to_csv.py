import pandas as pd
from datetime import datetime, timedelta
import calendar, time, os, sys

# read in xlsx file, convert date time to epoch, and output csv for processing

# filepath = 'Z:/cruise_data/raw/2018/fk180731/tuna_sand/20180805_215810_ts_un6k/payload/8_5_TS-1-2_Cal.xlsx'

def chemical_xlsx_to_csv(filepath, timezone_offset_to_utc = 0):
	outpath = filepath.split(filepath.split(os.sep)[-1])[0] + filepath.split(os.sep)[-1].split('.xlsx')[0] + '.csv'
	# outpath = filepath.split(os.sep)[:-1] + filepath.split(os.sep)[-1].split('.xlsx')[0] # 'Z:/cruise_data/raw/2018/fk180731/ae2000f/20180803_065749_ae2000f_sx3/payload/8_3_AE_Cal_Extra_Col.csv'

	df = pd.read_excel(filepath)

	epoch_time_list = []
	for i in range(len(df.index)):
		date = df['Date'][i].to_pydatetime()
		time = df['Time'][i]
		datetime_combined = datetime.combine(date, time)
		datetime_combined_utc = datetime_combined - timedelta(hours=timezone_offset_to_utc)
		epoch_time = calendar.timegm(datetime_combined_utc.timetuple())
		epoch_time_list.append(epoch_time)

	df['epoch_time'] = epoch_time_list

	df.to_csv(outpath)

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print('Error: not enough arguments')
		# syntax_error()
	else:
		# read in filepath, start time and finish time from function call
		flag_i = False
		flag_t = False
		for i in range(len(sys.argv)):
			option = sys.argv[i]
			if option == "-i":
				filepath = sys.argv[i+1]
				flag_i = True
			if option == "-t":
				timezone_offset_to_utc = int(sys.argv[i+1])
				flag_t = True
			# elif option == "-h":
			# 	print ('usage: mosaic_unsupervised_clustering.py -i <filepath containing csv_config.yaml>')
		if flag_i:
			if flag_t == True:
				chemical_xlsx_to_csv(filepath, timezone_offset_to_utc=timezone_offset_to_utc)
			else:
				chemical_xlsx_to_csv(filepath)
# chemical_xlsx_to_csv('Z:/cruise_data/raw/2018/fk180731/tuna_sand/20180806_210350_ts_un6k/payload/8_6_TS-1-2_Cal.xlsx', timezone_offset_to_utc=0)