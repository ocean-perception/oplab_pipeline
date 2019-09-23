#!/bin/bash

reprocess () {
    cd $1
    auv_nav parse -F .
    auv_nav process -F -s $2 -e $3 .
    auv_nav convert -s $2 -e $3 .
    cd ..
}


cd /media/oplab-insitu/raw/2019/dy109/autosub6000

reprocess 20190910_071106_as6_sx4_laser_calibration 20190910071106 20190910080940
reprocess 20190910_080940_as6_sx4_mapping 20190910080940 20190910091100
reprocess 20190910_092354_as6_sx4_mapping 20190910091300 20190910093400
reprocess 20190913_090214_as6_sx4_laser_calibration 20190913090000 20190913101500
reprocess 20190913_101337_as6_sx4_mapping 20190912112534 20190913161501
reprocess 20190916_090456_as6_sx4_mapping 20190916054653 20190917054605
reprocess 20190918_105330_as6_sx4_mapping 20190918082243 20190919091800
reprocess 20190922_140621_as6_sx4_mapping 20190921143920 20190922235958