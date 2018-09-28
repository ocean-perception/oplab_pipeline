# dead_reckoning

# Scripts to shift deadreackoning data to a single USBL position. This should be improved to match on average as minimum, but really needs to be turned into a kalman/particle filter

# Author: Blair Thornton
# Date: 13/02/2018

import sys, os, math
sys.path.append("..")
from lib_calculus.interpolate import interpolate

class usbl_offset:
	def __init__(self, time_dead_reckoning, northings_dead_reckoning, eastings_dead_reckoning, time_usbl, northings_usbl, eastings_usbl):
		return
				

	def __new__(cls, time_dead_reckoning, northings_dead_reckoning, eastings_dead_reckoning, time_usbl, northings_usbl, eastings_usbl):

# ===============Average Offset===================================================
		start_dead_reckoning=0
		start_usbl=0

		end_dead_reckoning=0
		end_usbl=0

		threshold = 20  # what to consider a big jump in time      
		exit_flag = False

		# Find suitable start points
		if time_usbl[0] < time_dead_reckoning[0]:# if 
			print('usbl starts before dead_reckoning')
			while exit_flag == False:
				#print(start_dead_reckoning,start_usbl,time_dead_reckoning[start_dead_reckoning],time_usbl[start_usbl],time_usbl[start_usbl+1],time_dead_reckoning[start_dead_reckoning] - time_usbl[start_usbl],time_dead_reckoning[start_dead_reckoning] - time_usbl[start_usbl+1])
				if time_dead_reckoning[start_dead_reckoning] - time_usbl[start_usbl+1] > 0:
					start_usbl=start_usbl+1
				else:
					if time_dead_reckoning[start_dead_reckoning] - time_usbl[start_usbl] < threshold and time_usbl[start_usbl+1] - time_usbl[start_usbl] < threshold:
						start_usbl+=1
						exit_flag = True
					else:
						start_dead_reckoning=start_dead_reckoning+1
		else:
			print('usbl starts after dead_reckoning')
			while exit_flag == False:                 
				if time_dead_reckoning[start_dead_reckoning]-time_usbl[start_usbl]<0:                    
					start_dead_reckoning=start_dead_reckoning+1

				else:                                    
					if time_dead_reckoning[start_dead_reckoning]-time_usbl[start_usbl]<threshold and time_usbl[start_usbl+1]-time_usbl[start_usbl]<threshold:
						start_usbl+=1
						exit_flag = True                     
					else:# if the jump is too big, ignore and try another usbl fix
						start_usbl=start_usbl+1

		northings_dead_reckoning_interpolated=[]
		eastings_dead_reckoning_interpolated=[]

		for j_usbl in range(start_usbl, len(time_usbl)):
			# if start_dead_reckoning+1>=len(time_dead_reckoning):
			#     break
			try:
				while time_dead_reckoning[start_dead_reckoning+1]<time_usbl[j_usbl]:
					start_dead_reckoning += 1
			except IndexError:
				break
			northings_dead_reckoning_interpolated.append(interpolate(time_usbl[j_usbl],time_dead_reckoning[start_dead_reckoning],time_dead_reckoning[start_dead_reckoning+1],northings_dead_reckoning[start_dead_reckoning],northings_dead_reckoning[start_dead_reckoning+1]))
			eastings_dead_reckoning_interpolated.append(interpolate(time_usbl[j_usbl],time_dead_reckoning[start_dead_reckoning],time_dead_reckoning[start_dead_reckoning+1],eastings_dead_reckoning[start_dead_reckoning],eastings_dead_reckoning[start_dead_reckoning+1]))

		# print (len(northings_usbl[start_usbl:start_usbl+len(northings_dead_reckoning_interpolated)]),len(eastings_usbl[start_usbl:(start_usbl+len(eastings_dead_reckoning_interpolated))]),len(northings_dead_reckoning_interpolated),len(eastings_dead_reckoning_interpolated))
		northings_offset = sum(northings_usbl[start_usbl:start_usbl+len(northings_dead_reckoning_interpolated)])/len(northings_usbl[start_usbl:(start_usbl+len(northings_dead_reckoning_interpolated))]) - sum(northings_dead_reckoning_interpolated)/len(northings_dead_reckoning_interpolated)
		eastings_offset = sum(eastings_usbl[start_usbl:(start_usbl+len(eastings_dead_reckoning_interpolated))])/len(eastings_usbl[start_usbl:start_usbl+len(eastings_dead_reckoning_interpolated)]) - sum(eastings_dead_reckoning_interpolated)/len(eastings_dead_reckoning_interpolated)

		return northings_offset, eastings_offset

# ===============Initial Offset===================================================

		# start_dead_reckoning=0
		# start_usbl=0
		# threshold = 20  # what to consider a big jump in time      
		# exit_flag = False

		# if time_usbl[0] < time_dead_reckoning[0]:# if 
		#     print('usbl starts before dead_reckoning')
		#     while exit_flag == False:
		#         #print(start_dead_reckoning,start_usbl,time_dead_reckoning[start_dead_reckoning],time_usbl[start_usbl],time_usbl[start_usbl+1],time_dead_reckoning[start_dead_reckoning] - time_usbl[start_usbl],time_dead_reckoning[start_dead_reckoning] - time_usbl[start_usbl+1])
		#         if time_dead_reckoning[start_dead_reckoning] - time_usbl[start_usbl+1] > 0:
		#             start_usbl=start_usbl+1
		#         else:
		#             if time_dead_reckoning[start_dead_reckoning] - time_usbl[start_usbl] < threshold and time_usbl[start_usbl+1] - time_usbl[start_usbl] < threshold:
		#                 exit_flag = True
		#             else:
		#                 start_dead_reckoning=start_dead_reckoning+1

		# else:
		#     print('usbl starts after dead_reckoning')
		#     while exit_flag == False:                 
		#         if time_dead_reckoning[start_dead_reckoning]-time_usbl[start_usbl]<0:                    
		#             start_dead_reckoning=start_dead_reckoning+1

		#         else:                                    
		#             if time_dead_reckoning[start_dead_reckoning]-time_usbl[start_usbl]<threshold and time_usbl[start_usbl+1]-time_usbl[start_usbl]<threshold:
		#                 exit_flag = True                     
		#             else:# if the jump is too big, ignore and try another usbl fix
		#                 start_usbl=start_usbl+1     

		# northings_usbl_interpolated=interpolate(time_dead_reckoning[start_dead_reckoning],time_usbl[start_usbl],time_usbl[start_usbl+1],northings_usbl[start_usbl],northings_usbl[start_usbl+1])
		# eastings_usbl_interpolated=interpolate(time_dead_reckoning[start_dead_reckoning],time_usbl[start_usbl],time_usbl[start_usbl+1],eastings_usbl[start_usbl],eastings_usbl[start_usbl+1])

		# #offset by the deadreackoning position that has been interpolated to 
		# northings_usbl_interpolated=northings_usbl_interpolated-northings_dead_reckoning[start_dead_reckoning]
		# eastings_usbl_interpolated=eastings_usbl_interpolated-eastings_dead_reckoning[start_dead_reckoning]
	   
		# return northings_usbl_interpolated,eastings_usbl_interpolated
