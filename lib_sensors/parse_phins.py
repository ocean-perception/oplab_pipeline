# parse_phins

# Scripts to parse ixsea blue phins data

# Author: Blair Thornton
# Date: 25/08/2017

import csv
from datetime import datetime
import codecs, time


class parse_phins:
	def __init__(self, filepath, filename, category, timezone, timeoffset, ftype, fileout):

		# define headers used in phins
		header_start = '$PIXSE'
		header_time = 'TIME__'
		header_atitu = 'ATITUD'
		header_depth = 'DEPIN_'
		header_speed = 'SPEED_'
		header_speed_std = 'STDSPD'
		class_string = 'measurement'
		sensor_string = 'phins'



		# read in date from filename
		yyyy_phins = int(filename[0:4])
		mm_phins = int(filename[5:6])
		dd_phins = int(filename[7:8])

		# read in timezone
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

		# convert to seconds from utc
		timeoffset = -timezone_offset*60*60 + timeoffset 

		# parse phins data
		print('parsing phins standard data from')
		with codecs.open(filepath + filename,'r',encoding='utf-8', errors='ignore') as filein:
			# csvread = csv.reader(filein)            
			# print(filepath + filename)
			# refHead='$HEHDT,';            
			flag_got_time = 0
			string_prev = ''
			for line in filein.readlines():

				line_split = line.strip().split('*')
				line_split_no_checksum = line_split[0].strip().split(',')
				
				# print(line_split_no_checksum)
				# what are the uncertainties
				if line_split_no_checksum[0] == header_start:
					# get time stamp
					if line_split_no_checksum[1]  == header_time:						
						time_string=str(line_split_no_checksum[2])
						hour_phins=int(time_string[0:1])
						mins_phins=int(time_string[2:3])
						secs_phins=int(time_string[4:5])
						msec_phins=int(time_string[7:10])						
						
						dt_obj = datetime(yyyy_phins,mm_phins,dd_phins,hour_phins,mins_phins,secs_phins)
						time_tuple = dt_obj.timetuple()
						epoch_time = time.mktime(time_tuple)
						epoch_timestamp = epoch_time+msec_phins/1000+timeoffset						
						flag_got_time = 1						
					else:
						# get other data only if have a time stamp
						if flag_got_time == 1:
							if category == 'velocity':
								frame_string = 'body'
								if line_split_no_checksum[1]  == header_speed:
									xx_velocity=float(line_split_no_checksum[2])
									yy_velocity=float(line_split_no_checksum[3])
									zz_velocity=float(line_split_no_checksum[4])
									# print('speed',xvel_phins,yvel_phins,zvel_phins)
								if line_split_no_checksum[1]  == header_speed_std:
									xx_velocity_std=float(line_split_no_checksum[2])
									yy_velocity_std=float(line_split_no_checksum[3])
									zz_velocity_std=float(line_split_no_checksum[4])
									# print('speed_std',xvel_std_phins,yvel_std_phins,zvel_std_phins)
									flag_got_time = 0

									# write out in the required format interlace at end
									print(epoch_timestamp,',',class_string,',',sensor_string,',',frame_string,category,',','xx_velocity',',',xx_velocity,',',xx_velocity_std)
									print(epoch_timestamp,',',class_string,',',sensor_string,',',frame_string,category,',','yy_velocity',',',yy_velocity,',',yy_velocity_std)
									print(epoch_timestamp,',',class_string,',',sensor_string,',',frame_string,category,',','zz_velocity',',',zz_velocity,',',zz_velocity_std)
									
									

							# if ftype == 'velocity':
							# 	if line_split_no_checksum[1]  == header_speed:
							# 		xvel_phins=float(line_split_no_checksum[2])
							# 		yvel_phins=float(line_split_no_checksum[3])
							# 		zvel_phins=float(line_split_no_checksum[4])
							# 		# print('speed',xvel_phins,yvel_phins,zvel_phins)
							# 	if line_split_no_checksum[1]  == header_speed_std:
							# 		xvel_std_phins=float(line_split_no_checksum[2])
							# 		yvel_std_phins=float(line_split_no_checksum[3])
							# 		zvel_std_phins=float(line_split_no_checksum[4])
							# 		# print('speed_std',xvel_std_phins,yvel_std_phins,zvel_std_phins)
							# 		flag_got_time = 0							
						else:
							continue

		# velocity_timezone 
                # velocity_timeoffset        
# calendar.timegm()
                # # timeFormat='HHMMSS.FFF'
                


# dateFormat='yyyymmdd'
# timeFormat='HHMMSS.FFF'
# refHead='$HEHDT,';
# refStat='$PIXSE,';
# refTime='TIME__,';
# refAtit='ATITUD,';
# refDept='DEPIN_,';
# refSped='LOGIN_,';

# j=1;
# jj=1;
# date_shift=0;

# k=length(refHead);
# dateTime=NaN;
# yaw=NaN;
# for i=1:n
#     [line]=sscanf(position{i,:},'%c');
        
#     header=line(1,1:k);
    
#     if header==refStat
#         specifier=line(1,k+1:2*k);        
#         switch specifier
#             case refTime                
               
# %                
#                 hh=line(1,2*k+1:2*k+2);      
#                 mm=line(1,2*k+3:2*k+4);
#                 ss=line(1,2*k+5:2*k+6);
#                 msec=line(1,2*k+8:2*k+11);       
                
#                 dateTime=offsetTime(filename,hh,mm,ss,msec,gmtOffset,secondsOffset);                                                                              
# %                 fprintf(fid, 'COMPASS:,%s,%s,',datestr(dateTime,dateFormat),datestr(dateTime,timeFormat));
#             case refAtit
#                 str=strsplit(line(1,:),'*');           
#                 subStr=strsplit(str{1},',');
#                 roll=str2num(subStr{3});
#                 pitch=str2num(subStr{4});
#                 if isnan(dateTime)==0 & isnan(yaw)==0
#                     fprintf(fid, 'COMPASS:,%s,%s,r=%6.6f,p=%6.6f,y=%6.6f\n',datestr(dateTime,dateFormat),datestr(dateTime,timeFormat),roll,pitch,yaw);
#                     dateTime=NaN;% not duplicate 
#                     yaw=NaN;% not duplicate 
#                 end
#             case refDept
#                 str=strsplit(line(1,:),'*');           
#                 subStr=strsplit(str{1},',');
#                 depth=str2num(subStr{3});
#                 time=subStr{4};
#                 hh=time(1,1:2);      
#                 mm=time(1,3:4);      
#                 ss=time(1,5:6);      
#                 msec=time(1,8:10);  
#                 dateTimeDepth=offsetTime(filename,hh,mm,ss,msec,gmtOffset,secondsOffset);
# %                 datestr(dateTimeDepth)
#                 fprintf(fid, 'DEPTH:,%s,%s,d=%6.6f\n',datestr(dateTimeDepth,dateFormat),datestr(dateTimeDepth,timeFormat),depth);                
# %                 pause
                
#             case refSped
#                 str=strsplit(line(1,:),'*');           
#                 subStr=strsplit(str{1},',');                
#                 uvel=str2num(subStr{3});
#                 vvel=str2num(subStr{4});
#                 wvel=str2num(subStr{5});
#                 time=subStr{7};
#                 hh=time(1,1:2);      
#                 mm=time(1,3:4);      
#                 ss=time(1,5:6);      
#                 msec=time(1,8:10);  
#                 dateTimeDVL=offsetTime(filename,hh,mm,ss,msec,gmtOffset,secondsOffset);   
#                 fprintf(fid, 'DVL:,%s,%s,uvel=%6.6f,vvel=%6.6f,wvel=%6.6f\n',datestr(dateTimeDVL,dateFormat),datestr(dateTimeDVL,timeFormat),uvel,vvel,wvel);                                                
#         end
        
#     else if header==refHead            
#             str=strsplit(line(1,:),',');            
#             yaw=str2num(str{2});       
#         end
#     end
   
    
#        if isnan(dateTime)==0
            
# %             fprintf(fid, '%s,%s,%6.6f,%6.6f,%4.2f\n',date_(jj,:),time_(jj,:),lat_(jj),lon_(jj),depth_(jj));
# %             jj=jj+1;
#         end
# end
# close(fid)
# return