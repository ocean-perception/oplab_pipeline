#!/bin/bash

# Copyright (c) 2023, University of Southampton
# All rights reserved.
# Licensed under the BSD 3-Clause License.
# 2022-07-18
# Authors: J. Cappelletto, A. Bodenmann

HELP_STRING="$0 corrects the timestamps in a file recorded by ALR for ALR's clock drift, by using the BioCam clock as reference.

Usage: 
dy152_alr_fixtime.sh [-f] -s SERIAL_LOG -e INPUT_FILE -o OUTPUT_FILE
Options:
    -f:          Overwrite the output file if it already exists.
    SERIAL_LOG:  (Mandatory) Serial log file from BioCam, usually
                 YYYYMMDD_hhmmss_auv_serial_message.log, where
                 YYYYMMDD_hhmmss is an actual date and time.
    INPUT_FILE:  (Mandatory) csv file with timestamped data, where the
                 timestamp is in epoch format in the first column. E.g. the
                 engineering log from the ALR team.
    OUTPUT_FILE: (Optional) Path of where the output file will be stored. If
                 not provided, the output file will be stored in the same
                 folder as the input file, with the prefix 'corrected_'.
Example: dy152_alr_fixtime.sh -s 20220713_171541_auv_serial_message.log -e engineering_log_m46.csv -o output_log_m46.csv"

# If no argument is provided, or "--help", print help string and exit
if [[ -z "$1" || "$1" == "--help" ]]; then
    echo -e "$HELP_STRING"
	exit
fi

# Parsing method based on explanation at http://wiki.bash-hackers.org/howto/getopts_tutorial
OVERWRITE=false
while getopts "s:e:o:fh" opt; do
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
    f)
    OVERWRITE=true
    ;;
    h)
    echo -e "$HELP_STRING"
	exit 1
    ;;
    \?)
	echo -e "Invalid option: -$OPTARG\n\nHelp:" >&2
    echo -e "$HELP_STRING"
	exit 1
	;;
    :)
	echo -e "Option -$OPTARG requires an argument.\n\nHelp:" >&2
    echo -e "$HELP_STRING"
	exit 1
	;;
  esac
done

if [[ -z "$OUTPUT_FILE" ]]; then
    PREFIX="corrected_"
    PREFIX=$(echo $PREFIX | rev)
    OUTPUT_FILE=$(echo "${ENG_LOG}" | rev | sed "s/\//${PREFIX}\//" | rev)
fi

echo -e "Serial log file path:\t $SERIAL_LOG" >&2
echo -e "Input file path:\t $ENG_LOG" >&2
echo -e "Output file path:\t $OUTPUT_FILE" >&2

# Check input files exist
if [ ! -f "$SERIAL_LOG" ]; then
    echo -e "ERROR: Serial log file: ["${SERIAL_LOG}"] does not exist"
    exit
fi

if [ ! -f "${ENG_LOG}" ]; then
    echo -e "ERROR: Input file: ["${ENG_LOG}"] does not exist"
    exit
fi

# Create a scratch file to store the poll_time entries. Append a random number to avoid overwriting scratch files from other processes when running in parallel.
SCRATCH_FILE="$SERIAL_LOG"".events".$RANDOM
while [ -f "$SCRATCH_FILE" ]; do
    SCRATCH_FILE="$SERIAL_LOG"".events".$RANDOM
done

echo -e "Creating scratch file $SCRATCH_FILE"
# Step 1: parse the content of the SERIAL_LOG, extracting the poll_time entries
cat "${SERIAL_LOG}" | grep poll_time > ""${SCRATCH_FILE}""
EVENTS=$(cat ""${SCRATCH_FILE}"" | wc -l)
if [ -z $EVENTS ]; then
    echo "No [poll_time] was detected in the serial log [${SERIAL_LOG}]."
    echo "Deleting scratch file "$SCRATCH_FILE
    rm -f "${SCRATCH_FILE}"
    echo "Exiting..."
    exit
else
    echo "Total [poll_time] events detected: "${EVENTS}
fi

FIRST=$(head -n 1 ""${SCRATCH_FILE}"" | awk -F: '{print $3}' | awk -F, '{print $4","$6}')
LAST=$(tail -n 1 ""${SCRATCH_FILE}"" | awk -F: '{print $3}' | awk -F, '{print $4","$6}')

echo "Deleting scratch file $SCRATCH_FILE"
rm -f "${SCRATCH_FILE}"

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

# Phase 2: Read the input file and prepend a correct_timestamp column with the adjusted timestamp (ALR to BioCam)
echo "Correcting timestamp. The column with the adjusted timestamp will be prepended to the existing data and written to ${OUTPUT_FILE}."

if [ -f "$OUTPUT_FILE" ]; then
    if [ "$OVERWRITE" = false ]; then
        echo "Output file ["$OUTPUT_FILE"] exists. Exiting. Use -f to overwrite."
        exit
    fi
    echo "Output file ["$OUTPUT_FILE"] exists. Overwriting."
fi
# Export header
echo "corrected_timestamp,"$(head -n1 "${ENG_LOG}") > "$OUTPUT_FILE"

# Load all the lines in the file except the header
LIST=$(tail -n+2 "${ENG_LOG}")

echo "Calculate and prepend corrected timestamp for every line in file..."
TOTAL_LINES="$(echo "$LIST" | wc -l)"
COUNTER=0
for KK in $LIST; do
    TIME_ALR=$(echo $KK | awk -F, '{print $1}')
    TIME_ADJ=$(echo "scale=10; (($TIME_ALR * 1000.0 - $OFFSET)  / $SLOPE)/1000" | bc )
    echo ${TIME_ADJ},${KK} >> "${OUTPUT_FILE}"
    # echo $TIME_ALR   ${TIME_ADJ}
    COUNTER=$(($COUNTER+1))
    if [[ $((COUNTER % 10)) -eq 0 || $COUNTER -eq $TOTAL_LINES ]]; then
        echo -ne "Processed $COUNTER out of $TOTAL_LINES lines\r"
    fi
done
echo -e "\n...done!"
