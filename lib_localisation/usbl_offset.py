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
    
        i=0
        j=0
        threshold = 20  # what to consider a big jump in time      
        exit_flag = False

        if time_usbl[0] < time_dead_reckoning[0]:# if 
            print('usbl starts before dead_reckoning')
            while exit_flag == False:
                #print(i,j,time_dead_reckoning[i],time_usbl[j],time_usbl[j+1],time_dead_reckoning[i] - time_usbl[j],time_dead_reckoning[i] - time_usbl[j+1])
                if time_dead_reckoning[i] - time_usbl[j+1] > 0:
                    j=j+1
                else:
                    if time_dead_reckoning[i] - time_usbl[j] < threshold and time_usbl[j+1] - time_usbl[j] < threshold:
                        exit_flag = True
                    else:
                        i=i+1

        else:
            print('usbl starts after dead_reckoning')
            while exit_flag == False:                 
                if time_dead_reckoning[i]-time_usbl[j]<0:                    
                    i=i+1

                else:                                    
                    if time_dead_reckoning[i]-time_usbl[j]<threshold and time_usbl[j+1]-time_usbl[j]<threshold:
                        exit_flag = True                     
                    else:# if the jump is too big, ignore and try another usbl fix
                        j=j+1     

        northings_usbl_interpolated=interpolate(time_dead_reckoning[i],time_usbl[j],time_usbl[j+1],northings_usbl[j],northings_usbl[j+1])
        eastings_usbl_interpolated=interpolate(time_dead_reckoning[i],time_usbl[j],time_usbl[j+1],eastings_usbl[j],eastings_usbl[j+1])

        #offset by the deadreackoning position that has been interpolated to 
        northings_usbl_interpolated=northings_usbl_interpolated-northings_dead_reckoning[i]
        eastings_usbl_interpolated=eastings_usbl_interpolated-eastings_dead_reckoning[i]
       
        return northings_usbl_interpolated,eastings_usbl_interpolated
