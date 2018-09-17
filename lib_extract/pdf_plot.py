# plot data in pdf
        if plot is True:
            print('Plotting data ...')
            plotpath = renavpath + os.sep + 'plots'
            
            if os.path.isdir(plotpath) == 0:
                try:
                    os.mkdir(plotpath)
                except Exception as e:
                    print("Warning:",e)

        # orientation
            print('...plotting orientation_vs_time...')
            
            fig=plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.plot([i.timestamp for i in orientation_list], [i.yaw for i in orientation_list],'r.',label='Yaw')   
            ax.plot([i.timestamp for i in orientation_list], [i.roll for i in orientation_list],'b.',label='Roll')      
            ax.plot([i.timestamp for i in orientation_list], [i.pitch for i in orientation_list],'g.',label='Pitch')                     
            ax.set_xlabel('Epoch time, s')
            ax.set_ylabel('Degrees')
            ax.legend(loc='upper right', bbox_to_anchor=(1, -0.2))
            ax.grid(True)                        
            plt.savefig(plotpath + os.sep + 'orientation_vs_time.pdf', dpi=600, bbox_inches='tight')
            plt.close()

        # velocity_body (north,east,down) compared to velocity_inertial
            print('...plotting velocity_vs_time...')
            
            fig=plt.figure(figsize=(10,7))
            ax1 = fig.add_subplot(321)            
            ax1.plot([i.timestamp for i in dead_reckoning_dvl_list],[i.north_velocity for i in dead_reckoning_dvl_list], 'ro',label='DVL')#time_velocity_body,north_velocity_inertia_dvl, 'ro',label='DVL')
            if len(velocity_inertial_list) > 0:
                ax1.plot([i.timestamp for i in velocity_inertial_list],[i.north_velocity for i in velocity_inertial_list], 'b.',label=velocity_inertial_sensor_name)
            ax1.set_xlabel('Epoch time, s')
            ax1.set_ylabel('Velocity, m/s')
            ax1.legend()
            ax1.grid(True)
            ax1.set_title('north velocity')
            ax2 = fig.add_subplot(323)            
            ax2.plot([i.timestamp for i in dead_reckoning_dvl_list],[i.east_velocity for i in dead_reckoning_dvl_list],'ro',label='DVL')
            if len(velocity_inertial_list) > 0:
                ax2.plot([i.timestamp for i in velocity_inertial_list],[i.east_velocity for i in velocity_inertial_list],'b.',label=velocity_inertial_sensor_name)
            ax2.set_xlabel('Epoch time, s')
            ax2.set_ylabel('Velocity, m/s')
            ax2.legend()
            ax2.grid(True)
            ax2.set_title('east velocity')
            ax3 = fig.add_subplot(325)            
            ax3.plot([i.timestamp for i in dead_reckoning_dvl_list],[i.down_velocity for i in dead_reckoning_dvl_list],'ro',label='DVL')#time_velocity_body,down_velocity_inertia_dvl,'r.',label='DVL')
            if len(velocity_inertial_list) > 0:
                ax3.plot([i.timestamp for i in velocity_inertial_list],[i.down_velocity for i in velocity_inertial_list],'b.',label=velocity_inertial_sensor_name)
            ax3.set_xlabel('Epoch time, s')
            ax3.set_ylabel('Velocity, m/s')
            ax3.legend()
            ax3.grid(True)
            ax3.set_title('down velocity')
            ax4 = fig.add_subplot(322)
            ax4.plot([i.timestamp for i in dead_reckoning_dvl_list],[i.x_velocity for i in dead_reckoning_dvl_list], 'r.',label='Surge') #time_velocity_body,x_velocity, 'r.',label='Surge')
            ax4.set_xlabel('Epoch time, s')
            ax4.set_ylabel('Velocity, m/s')
            ax4.legend()
            ax4.grid(True)
            ax4.set_title('x velocity')
            ax5 = fig.add_subplot(324)
            ax5.plot([i.timestamp for i in dead_reckoning_dvl_list],[i.y_velocity for i in dead_reckoning_dvl_list], 'g.',label='Sway')#time_velocity_body,y_velocity, 'g.',label='Sway')
            ax5.set_xlabel('Epoch time, s')
            ax5.set_ylabel('Velocity, m/s')
            ax5.legend()
            ax5.grid(True)
            ax5.set_title('y velocity')
            ax6 = fig.add_subplot(326)
            ax6.plot([i.timestamp for i in dead_reckoning_dvl_list],[i.z_velocity for i in dead_reckoning_dvl_list], 'b.',label='Heave')#time_velocity_body,z_velocity, 'b.',label='Heave')
            ax6.set_xlabel('Epoch time, s')
            ax6.set_ylabel('Velocity, m/s')
            ax6.legend()
            ax6.grid(True)
            ax6.set_title('z velocity')
            fig.tight_layout()
            plt.savefig(plotpath + os.sep + 'velocity_vs_time.pdf', dpi=600, bbox_inches='tight')
            plt.close()

        # time_dead_reckoning northings eastings depth vs time
            print('...plotting deadreckoning_vs_time...')
            
            fig=plt.figure(figsize=(12,7))
            ax1 = fig.add_subplot(221)
            ax1.plot([i.timestamp for i in dead_reckoning_dvl_list],[i.northings for i in dead_reckoning_dvl_list],'r.',label='DVL')#time_velocity_body,northings_dead_reckoning,'r.',label='DVL')
            if len(velocity_inertial_list) > 0:
                ax1.plot([i.timestamp for i in velocity_inertial_list],[i.northings for i in velocity_inertial_list],'g.',label=velocity_inertial_sensor_name)
            ax1.plot([i.timestamp for i in usbl_list], [i.northings for i in usbl_list],'b.',label='USBL')
            ax1.plot([i.timestamp for i in dead_reckoning_centre_list],[i.northings for i in dead_reckoning_centre_list],'c.',label='Centre')#time_velocity_body,northings_dead_reckoning,'b.')
            ax1.set_xlabel('Epoch time, s')
            ax1.set_ylabel('Northings, m')
            ax1.grid(True)
            ax1.legend()#loc='upper right', bbox_to_anchor=(1, -0.2))
            ax1.set_title('Northings')
            ax2 = fig.add_subplot(222)
            ax2.plot([i.timestamp for i in dead_reckoning_dvl_list],[i.eastings for i in dead_reckoning_dvl_list],'r.',label='DVL')#time_velocity_body,northings_dead_reckoning,'r.',label='DVL')
            if len(velocity_inertial_list) > 0:
                ax2.plot([i.timestamp for i in velocity_inertial_list],[i.eastings for i in velocity_inertial_list],'g.',label=velocity_inertial_sensor_name)
            ax2.plot([i.timestamp for i in usbl_list], [i.eastings for i in usbl_list],'b.',label='USBL')
            ax2.plot([i.timestamp for i in dead_reckoning_centre_list],[i.eastings for i in dead_reckoning_centre_list],'c.',label='Centre')#time_velocity_body,eastings_dead_reckoning,'b.')
            ax2.set_xlabel('Epoch time, s')
            ax2.set_ylabel('Eastings, m')
            ax2.grid(True)
            ax2.legend()
            ax2.set_title('Eastings')
            ax3 = fig.add_subplot(223)
            ax3.plot([i.timestamp for i in usbl_list],[i.depth for i in usbl_list],'b.',label='USBL depth') 
            ax3.plot([i.timestamp for i in depth_list],[i.depth for i in depth_list],'g-',label='Depth Sensor') 
            ax3.plot([i.timestamp for i in altitude_list],[i.seafloor_depth for i in altitude_list],'r-',label='Seafloor') 
            ax3.plot([i.timestamp for i in dead_reckoning_centre_list],[i.depth for i in dead_reckoning_centre_list],'c-',label='Centre')#time_velocity_body,eastings_dead_reckoning,'b.')
            ax3.set_xlabel('Epoch time, s')
            ax3.set_ylabel('Depth, m')
            plt.gca().invert_yaxis()
            ax3.grid(True)
            ax3.legend()
            ax3.set_title('Depth')
            ax4 = fig.add_subplot(224)
            ax4.plot([i.timestamp for i in altitude_list],[i.altitude for i in altitude_list],'r.',label='Altitude')              
            ax4.set_xlabel('Epoch time, s')
            ax4.set_ylabel('Altitude, m')
            ax4.set_xlim(min([i.timestamp for i in depth_list]),max([i.timestamp for i in depth_list]))
            ax4.grid(True)
            ax4.legend()
            fig.tight_layout()
            plt.savefig(plotpath + os.sep + 'deadreckoning_vs_time.pdf', dpi=600, bbox_inches='tight')
            plt.close()

        # usbl latitude longitude
            print('...plotting usbl_LatLong_vs_NorthEast...')

            fig=plt.figure(figsize=(10,5))
            ax1 = fig.add_subplot(121)
            ax1.plot([i.longitude for i in usbl_list],[i.latitude for i in usbl_list],'b.')
            ax1.set_xlabel('Longitude, degrees')
            ax1.set_ylabel('Latitude, degrees')
            ax1.grid(True)
            ax2 = fig.add_subplot(122)
            ax2.plot([i.eastings for i in usbl_list],[i.northings for i in usbl_list],'b.',label='Reference ['+str(latitude_reference)+','+str(longitude_reference)+']')                 
            ax2.set_xlabel('Eastings, m')
            ax2.set_ylabel('Northings, m')
            ax2.grid(True)
            ax2.legend()
            fig.tight_layout()
            plt.savefig(plotpath + os.sep + 'usbl_LatLong_vs_NorthEast.pdf', dpi=600, bbox_inches='tight')
            plt.close()

        # uncertainties plot. 
            # https://plot.ly/python/line-charts/#filled-lines Something like that?
            print('...plotting uncertainties_plot...')

            fig=plt.figure(figsize=(15,7))
            ax1 = fig.add_subplot(231)
            ax1.plot([i.timestamp for i in orientation_list],[i.roll_std for i in orientation_list],'r.',label='roll_std')
            ax1.plot([i.timestamp for i in orientation_list],[i.pitch_std for i in orientation_list],'g.',label='pitch_std')
            ax1.plot([i.timestamp for i in orientation_list],[i.yaw_std for i in orientation_list],'b.',label='yaw_std')
            ax1.set_xlabel('Epoch time, s')
            ax1.set_ylabel('Angle, degrees')
            ax1.legend()
            ax1.grid(True)
            ax2 = fig.add_subplot(234)
            ax2.plot([i.timestamp for i in depth_list],[i.depth_std for i in depth_list],'b.',label='depth_std')
            ax2.set_xlabel('Epoch time, s')
            ax2.set_ylabel('Depth, m')
            ax2.legend()
            ax2.grid(True)
            ax3 = fig.add_subplot(232)
            ax3.plot([i.timestamp for i in velocity_body_list],[i.x_velocity_std for i in velocity_body_list],'r.',label='x_velocity_std')
            ax3.plot([i.timestamp for i in velocity_body_list],[i.y_velocity_std for i in velocity_body_list],'g.',label='y_velocity_std')
            ax3.plot([i.timestamp for i in velocity_body_list],[i.z_velocity_std for i in velocity_body_list],'b.',label='z_velocity_std')
            ax3.set_xlabel('Epoch time, s')
            ax3.set_ylabel('Velocity, m/s')
            ax3.legend()
            ax3.grid(True)
            ax4 = fig.add_subplot(235)
            if len(velocity_inertial_list) > 0:
                ax4.plot([i.timestamp for i in velocity_inertial_list],[i.north_velocity_std for i in velocity_inertial_list],'r.',label='north_velocity_std_inertia')
                ax4.plot([i.timestamp for i in velocity_inertial_list],[i.east_velocity_std for i in velocity_inertial_list],'g.',label='east_velocity_std_inertia')
                ax4.plot([i.timestamp for i in velocity_inertial_list],[i.down_velocity_std for i in velocity_inertial_list],'b.',label='down_velocity_std_inertia')
            ax4.set_xlabel('Epoch time, s')
            ax4.set_ylabel('Velocity, m/s')
            ax4.legend()
            ax4.grid(True)
            ax5 = fig.add_subplot(233)
            ax5.plot([i.timestamp for i in usbl_list],[i.latitude_std for i in usbl_list],'r.',label='latitude_std_usbl')
            ax5.plot([i.timestamp for i in usbl_list],[i.longitude_std for i in usbl_list],'g.',label='longitude_std_usbl')
            ax5.set_xlabel('Epoch time, s')
            ax5.set_ylabel('LatLong, degrees')
            ax5.legend()
            ax5.grid(True)
            ax6 = fig.add_subplot(236)
            ax6.plot([i.timestamp for i in usbl_list],[i.northings_std for i in usbl_list],'r.',label='northings_std_usbl')
            ax6.plot([i.timestamp for i in usbl_list],[i.eastings_std for i in usbl_list],'g.',label='eastings_std_usbl')
            ax6.set_xlabel('Epoch time, s')
            ax6.set_ylabel('NorthEast, m')
            ax6.legend()
            ax6.grid(True)
            fig.tight_layout()                  
            plt.savefig(plotpath + os.sep + 'uncertainties_plot.pdf', dpi=600, bbox_inches='tight')
            plt.close()

        # DR
            print('...plotting camera1_centre_DVL_{}_DR...'.format(velocity_inertial_sensor_name))
            fig=plt.figure()
            ax = fig.add_subplot(111)
            ax.plot([i.eastings for i in camera1_list],[i.northings for i in camera1_list],'y.',label='Camera1')
            ax.plot([i.eastings for i in dead_reckoning_centre_list],[i.northings for i in dead_reckoning_centre_list],'r.',label='Centre')
            ax.plot([i.eastings for i in dead_reckoning_dvl_list],[i.northings for i in dead_reckoning_dvl_list],'g.',label='DVL')
            if len(velocity_inertial_list) > 0:
                ax.plot([i.eastings for i in velocity_inertial_list],[i.northings for i in velocity_inertial_list],'m.',label=velocity_inertial_sensor_name)
            ax.plot([i.eastings for i in usbl_list], [i.northings for i in usbl_list],'c.',label='USBL')
            ax.plot([i.eastings for i in pf_fusion_dvl_list], [i.northings for i in pf_fusion_dvl_list], 'g', label='PF fusion_DVL')
            ax.set_xlabel('Eastings, m')
            ax.set_ylabel('Northings, m')
            ax.legend()#loc='upper right', bbox_to_anchor=(1, -0.2))
            ax.grid(True)   
            plt.savefig(plotpath + os.sep + 'camera1_centre_DVL_{}_DR.pdf'.format(velocity_inertial_sensor_name), dpi=600, bbox_inches='tight')
            plt.close()

            print('Complete plot data: ', plotpath)