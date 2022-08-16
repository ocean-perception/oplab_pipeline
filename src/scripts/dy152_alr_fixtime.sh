#!/bin/bash

# Copyright (c) 2022, University of Southampton
# All rights reserved.
# Licensed under the BSD 3-Clause License.
# 2022-07-18
# Author: J. Cappelletto

# if no argument is provided, the print basic usage
if [ -z "$1" ]; then 
	echo Usage: \n
	echo dy152_alr_fixtime.sh -s serial_log -e engineering_log -o corrected_log
	echo "************************************************************************************************"
	echo "Example: dy152_alr_fixtime.sh -s 20220713_171541_auv_serial_message.log -e engineering_log_m46.csv -o output_log_m64.csv"
	echo -e '\t' "[DY152 - ALR - Biocam] Adjust the drift between ALR internal clock used in the engineering"
	echo -e '\t' "and the BioCam epoch_timestamp used for auv_nav"
	exit
fi

#######################################################################################################################
# Parsing method extracted from http://wiki.bash-hackers.org/howto/getopts_tutorial
#######################################################################################################################

while getopts "s:e:o:" opt; do
  case $opt in
    s)
	SERIAL_LOG=$OPTARG 
	;;
    e)
	ENG_LOG=$OPTARG 
	;;
    o)
    OUTPUT_FILE=$OPTARG 
    ;;
    \?)
	echo "Invalid option: -$OPTARG" >&2
	exit 1
	;;
    :)
	echo "Option -$OPTARG requires an argument." >&2
	exit 1
	;;
  esac
done

echo -e "Serial log file:\t $SERIAL_LOG" >&2
echo -e "Engineering log file:\t $ENG_LOG" >&2

# Check input files exist
if [ ! -f $SERIAL_LOG ]; then
    echo -e "ERROR: Input file: ["${SERIAL_LOG}"] does not exist"
    exit
fi

if [ ! -f $ENG_LOG ]; then
    echo -e "ERROR: Input file: ["${ENG_LOG}"] does not exist"
    exit
fi

SCRATCH_FILE=$SERIAL_LOG".events"
# Step 1: parse the content of the SERIAL_LOG, extracting the poll_time entries
cat ${SERIAL_LOG} | grep poll_time > ${SCRATCH_FILE}
EVENTS=$(cat ${SCRATCH_FILE} | wc -l)
if [ -z $EVENTS ]; then
    echo "No [poll_time] was detected in the serial log [" ${SERIAL_LOG}]". Exitting..."
    exit
else
    echo "Total [poll_time] events detected: "${EVENTS}
fi

FIRST=$(head -n 1 ${SCRATCH_FILE} | awk -F: '{print $3}' | awk -F, '{print $4","$6}')
LAST=$(tail -n 1 ${SCRATCH_FILE} | awk -F: '{print $3}' | awk -F, '{print $4","$6}')

XINI=$(echo $FIRST | awk -F, '{print $1}')
XEND=$(echo $LAST  | awk -F, '{print $1}')
YINI=$(echo $FIRST | awk -F, '{print $2}')
YEND=$(echo $LAST  | awk -F, '{print $2}')

echo "Biocam times $XINI - $XEND"
echo "ALR    times $YINI - $YEND"

DELTA_X=$(( $XEND - $XINI ))
DELTA_Y=$(( $YEND - $YINI ))
echo "DeltaX: $DELTA_X"
echo "DeltaY: $DELTA_Y"

# We use BC for arithmetic operation (linear adjustment). To improve accuracy in the number representation, we shift all the values by XINI.
# We revert this change at the end when recalculating the actual epoch_timestamp
SLOPE=$(echo "scale=20; $DELTA_Y / $DELTA_X" | bc )
echo "Slope: $SLOPE"

OFFSET=$(echo "scale=20; $YINI - ( $SLOPE * $XINI )" | bc) 
echo "Offset: $OFFSET"

echo "Corrected---------"
YMIN=$(echo "scale=10; ($SLOPE * $XINI)  + $OFFSET" | bc )
echo "Adjusted BioCam to ALR:INI: $YMIN"

YADJ=$(echo "scale=10; ($YINI - $OFFSET)  / $SLOPE" | bc )
echo "Adjusted ALR to BioCam:INI: $YADJ"

YEADJ=$(echo "scale=10; ($YEND - $OFFSET)  / $SLOPE" | bc )
echo "Adjusted ALR to BioCam:END: $YEADJ"

# Phase 2: Read the engineering log and prepend a correct_timestamp column with the adjusted timestamp (ALR to BioCam)
# The output filename is corrected_${input filename}
PREFIX="corrected_"
PREFIX=$(echo $PREFIX | rev)

if [[ -z $OUTPUT_FILE ]]; then
    OUTFILE=$(echo ${ENG_LOG} | rev | sed "s/\//${PREFIX}\//" | rev)
    echo -e "Correcting engineering log timestamp. The column with the adjusted timestamp will be prepended to the existing data..."
    echo $OUTFILE
else
    OUTFILE=$OUTPUT_FILE
fi

if [ -f $OUTFILE ]; then
    echo "Output file ["$OUTFILE"] exists... overwritting"
fi
# Export header
echo "corrected_timestamp,"$(head -n1 ${ENG_LOG}) > $OUTFILE

# Iterate for every single row, and calculate the adjusted timestamp
LIST=$(tail -n+2 ${ENG_LOG})

for KK in $LIST; do
    TIME_ALR=$(echo $KK | awk -F, '{print $1}')
    TIME_ADJ=$(echo "scale=10; (($TIME_ALR * 1000.0 - $OFFSET)  / $SLOPE)/1000" | bc )
    echo ${TIME_ADJ},${KK} >> ${OUTFILE}
    # echo $TIME_ALR   ${TIME_ADJ}
done
