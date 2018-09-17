from datetime import datetime
import time, json, glob, os
import pandas as pd

class parse_chemical: # read xlsx file, calibrate values, convert to epoch timestamps, output to json file
    def __init__(self, filepath, filename, timezone, timeoffset, config_data, ftype, outpath, fileoutname, fileout):
        if ftype == 'oplab':
            def calibrate(x1, x2, y1, y2, x):
                m = (y2-y1)/(x2-x1)
                c = y1 - m*x1
                y = m*x + c
                return y

            class_string = 'measurement'
            sensor_string = 'chemical' # change this to specific XXX one, also change to parse_XXX class
            frame_string = 'body'
            category_string = 'chemical'

            if isinstance(timezone, str):           
                if timezone == 'utc' or  timezone == 'UTC':
                    timezone_offset = 0
                elif timezone == 'jst' or  timezone == 'JST':
                    timezone_offset = 9;
            else:
                try:
                    timezone_offset=float(timezone)
                except ValueError:
                    print('Error: timezone', timezone, 'in mission.cfg not recognised, please enter value from UTC in hours')
                    return
            timeoffset = -timezone_offset*60*60 + timeoffset

            print("...... parsing chemical data")
            data_list=[]
            df = pd.read_excel(filepath + filename, 'Sheet1')
            date_column = df['Date']
            time_column = df['Time']
            if len(time_column) != len(date_column):
                print ('Error! Time and Date columns do not match.')
            else:
                for i in range(len(date_column)):
                    yyyy = date_column[i].year
                    mm = date_column[i].month
                    dd = date_column[i].day

                    hour = time_column[i].hour
                    mins = time_column[i].minute
                    secs = time_column[i].second

                    dt_obj = datetime(yyyy,mm,dd,hour,mins,secs)

                    time_tuple = dt_obj.timetuple()
                    epoch_time = time.mktime(time_tuple)
                    epoch_timestamp = epoch_time+timeoffset
                    data = []
                    for j in config_data:
                        if 'Ch' in j['column_name']:
                            x = df[j['column_name']][i]
                            x1 = j['calibration']['x1']
                            x2 = j['calibration']['x2']
                            y1 = j['calibration']['y1']
                            y2 = j['calibration']['y2']
                            y = calibrate(x1,x2,y1,y2,x)
                            label = j['label']
                            units = j['units']
                            data.append({'label': label, 'value':y, 'units': units})
                    data_packet = {'epoch_timestamp': float(epoch_timestamp), 'class': class_string, 'sensor':sensor_string, 'frame': frame_string, 'category':category_string, 'data': data}
                    data_list.append(data_packet)

            fileout.close()
            for filein in glob.glob(outpath + os.sep + fileoutname):
                try:
                    with open(filein, 'rb') as json_file:                   
                        data_in=json.load(json_file)                        
                        for i in range(len(data_in)):
                            data_list.insert(0,data_in[len(data_in)-i-1])                       
                        
                except ValueError:                  
                    print('Initialising JSON file')

            with open(outpath + os.sep + fileoutname,'w') as fileout:
                json.dump(data_list, fileout)   
                del data_list