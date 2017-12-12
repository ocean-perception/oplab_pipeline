# auv_nav
    """ 
        To install, go to directory you want it to be installed in and in a terminal/command prompt, type

        git clone --recursive https://github.com/ocean-perception/auv_nav.git

        Requires python3.6.2 or later
        Requires matplotlib library which can be downloaded and installed from a terminal

        $ pip3 install matplotlib

        where further instructions can be found here https://matplotlib.org/2.0.2/users/installing.html

        Requires PyYAML which can be downloaded from http://pyyaml.org/wiki/PyYAML

        Go to the folder where the downloaded file (at time of writting)
        
            http://pyyaml.org/download/pyyaml/PyYAML-3.12.tar.gz

        is extracted and from in the extracted folder, execute the following terminal commands 

        $ python3 setup.py install
        $ python3 setup.py test
        
        To push updates, stage changes, commit and push to a branch, usually master

        git add -A
        git commit -m "Some message about the change"
        git push origin master

        Functionality:

        Parses and interleave navigation data for oplab standard and acfr standard formats. 

        inputs are 

        auv_nav.py <options>
            -i <path to mission.yaml>
            -o <output type> 'acfr' or 'oplab'


       Arguments:
            path to the "mission.yaml" file, output format 'acfr' or 'oplab'

                #YAML 1.0
                origin:
                - latitude: 26.674083
                - longitude: 127.868054
                - coordinate_reference_system: wgs84
                - date: 2017/08/17

                velocity:
                - format: phins
                - filepath: nav/phins/
                - filename: 20170817_phins.txt
                - timezone: utc
                - timeoffset: 0.0

                orientation:
                - format: phins
                - filepath: nav/phins/
                - filename: 20170817_phins.txt
                - timezone: utc
                - timeoffset: 0.0

                depth:
                - format: phins
                - filepath: nav/phins/
                - filename: 20170817_phins.txt
                - timezone: utc
                - timeoffset: 0.0

                altitude:
                - format: phins
                - filepath: nav/phins/
                - filename: 20170817_phins.txt
                - timezone: utc
                - timeoffset: 0.0

                usbl:
                - format: gaps
                - filepath: nav/gaps/
                - timezone: utc
                - timeoffset: 0.0

                image:
                - format: acfr_standard
                - filepath: image/r20170817_041459_UG117_sesoko/i20170817_041459/
                - camera1: LC
                - camera2: RC
                - timezone: utc
                - timeoffset: 0.0

        Returns:
            interleaved navigation and imaging data with output options:

                'acfr' - combined.RAW.auv
                    PHINS_COMPASS: 1444452882.644 r: -2.29 p: 17.21 h: 1.75 std_r: 0 std_p: 0 std_h: 0
                    RDI: 1444452882.644 alt:200 r1:0 r2:0 r3:0 r4:0 h:1.75 p:17.21 r:-2.29 vx:0.403 vy:0 vz:0 nx:0 ny:0 nz:0 COG:0 SOG:0 bt_status:32768 h_true:0 p_gimbal:0 sv: 1500
                    PAROSCI: 1444452882.644 298.289
                    VIS: 1444452882.655 [1444452882.655] sx_073311_image0003805_AC.tif exp: 0
                    VIS: 1444452882.655 [1444452882.655] sx_073311_image0003805_FC.tif exp: 0
                    SSBL_FIX: 1444452883 ship_x: 402.988947 ship_y: 140.275056 target_x: 275.337171 target_y: 304.388346 target_z: 299.2 target_hr: 0 target_sr: 364.347071 target_bearing: 127.876747

                'oplab' - nav_standard.json
                    [{"epoch_timestamp": 1501974125.926, "epoch_timestamp_dvl": 1501974125.875, "class": "measurement", "sensor": "phins", "frame": "body", "category": "velocity", "data": [{"x_velocity": -0.075, "x_velocity_std": 0.200075}, {"y_velocity": 0.024, "y_velocity_std": 0.200024}, {"z_velocity": -0.316, "z_velocity_std": 0.20031600000000002}]},
                    {"epoch_timestamp": 1501974002.1, "class": "measurement", "sensor": "phins", "frame": "inertial", "category": "orientation", "data": [{"heading": 243.777, "heading_std": 2.0}, {"roll": 4.595, "roll_std": 0.1}, {"pitch": 0.165, "pitch_std": 0.1}]},
                    {"epoch_timestamp": 1501974125.926, "epoch_timestamp_dvl": 1501974125.875, "class": "measurement", "sensor": "phins", "frame": "body", "category": "altitude", "data": [{"altitude": 31.53, "altitude_std": 0.3153}, {"sound_velocity": 1546.0, "sound_velocity_correction": 0.0}]},
                    {"epoch_timestamp": 1501974002.7, "epoch_timestamp_depth": 1501974002.674, "class": "measurement", "sensor": "phins", "frame": "inertial", "category": "depth", "data": [{"depth": -0.958, "depth_std": -9.58e-05}]},
                    {"epoch_timestamp": 1502840568.204, "class": "measurement", "sensor": "gaps", "frame": "inertial", "category": "usbl", "data_ship": [{"latitude": 26.66935735000014, "longitude": 127.86623359499968}, {"northings": -526.0556603025898, "eastings": -181.08730736724087}, {"heading": 174.0588800058365}], "data_target": [{"latitude": 26.669344833333334, "latitude_std": -1.7801748803947248e-06}, {"longitude": 127.86607166666667, "longitude_std": -1.992112444781924e-06}, {"northings": -527.4487693247576, "northings_std": 0.19816816183128352}, {"eastings": -197.19537408743128, "eastings_std": 0.19816816183128352}, {"depth": 28.8}]},{"epoch_timestamp": 1501983409.56, "class": "measurement", "sensor": "unagi", "frame": "body", "category": "image", "camera1": [{"epoch_timestamp": 1501983409.56, "filename": "PR_20170816_023649_560_LC16.tif"}], "camera2": [{"epoch_timestamp": 1501983409.56, "filename": "PR_20170816_023649_560_RC16.tif"}]}
                    ]

            These are stored in a mirrored file location where the input raw data is stored as follows with the paths to raw data as defined in mission.yaml
            
            e.g. 
                raw     /<YEAR> /<CRUISE>   /<DIVE> /mission.yaml
                                                    /nav/gaps/
                                                    /nav/phins/
                                                    /image/r20170816_023028_UG069_sesoko/i20170816_023028/

            For this example, the outputs would be stored in the follow location, where folders will be automatically generated

            # for oplab
                processed   /<YEAR> /<CRUISE>   /<DIVE> /nav            /nav_standard.json   
            
            # for acfr
                processed   /<YEAR> /<CRUISE>   /<DIVE> /dRAWLOGS_cv    /combined.RAW.auv   
                                                        /mission.cfg

            An example dataset can be downloaded from the following link with the expected folder structure

                https://drive.google.com/drive/folders/0BzYMMCBxpT8BUF9feFpEclBzV0k?usp=sharing
            
            Download, extract and specify the folder location and run as
                
                python3 auv_nav.py -i ~/raw/2017/cruise/dive/ -o acfr
                python3 auv_nav.py -i ~/raw/2017/cruise/dive/ -o oplab

            
    """
