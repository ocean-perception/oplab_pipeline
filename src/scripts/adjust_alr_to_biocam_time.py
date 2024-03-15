"""
Copyright (c) 2023, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class TimeCorrector:
    def __init__(
            self,
            biocam_serial_log,
            alr_timesamped_file,
            timestamp_col,
            divisions_per_second,
            plot,
            force
        ):
        self.biocam_serial_log = biocam_serial_log
        self.alr_timesamped_file = alr_timesamped_file
        self.timestamp_col = timestamp_col
        self.divisions_per_second = divisions_per_second
        self.plot = plot
        self.force = force
        self.output_files = []


    def populate_output_file_path(self):
        if self.alr_timesamped_file is None:
            return

        for atf in self.alr_timesamped_file:
            if atf[0][-4:] != ".csv":
                print(f"{atf[0]} does not end in .csv. Only .csv files are supported.")
                print("Aborting.")
                quit()
            self.output_files.append(atf[0].replace(".csv", "_corrected.csv"))
        
        output_file_exists = [Path(of).exists() for of in self.output_files]
        if not self.force and any(output_file_exists):
            existing_paths = np.array(self.output_files)[np.array(output_file_exists)]
            print(f"The output file(s) {existing_paths.tolist()} already exist(s).")
            print("Use -f to force overwrite.")
            print("Aborting.")
            quit()

            
    def analyse_time_differences(self):
        timestamps_bc_0_ms = []
        timestamps_alr_ms = []
        timestamps_bc_1_ms = []

        for bsl in self.biocam_serial_log:
            print(f"Reading BioCam and ALR time differences from serial log at {bsl}")

            if not Path(bsl).exists():
                print(f"{bsl} does not exist. Aborting.")
                quit()

            # Open serial log text file
            with open(bsl, "r") as f:
                lines = f.readlines()

            # Extract timestamps
            for line in lines:
                if "poll_time:" in line:
                    timestamps_bc_0_ms.append(int(line.split(",")[1]))
                    timestamps_alr_ms.append(int(line.split(",")[5]))
                    timestamps_bc_1_ms.append(int(line.split(",")[7]))
        
        if len(timestamps_bc_0_ms) == 0:
            print("No time poll lines found in the serial log. Aborting.")
            quit()

        if timestamps_bc_0_ms != sorted(timestamps_bc_0_ms):
            print(
                "Timestamps are not sorted. Please indicate BioCam serial losg in "
                "temporal order. Aborting."
            )
            quit()

        if timestamps_alr_ms != sorted(timestamps_alr_ms):
            print(
                "ALR timestamps are not sorted while BioCam timestamps are. This "
                "shoudld normally not happen. Aborting."
            )
            quit()

        timestamps_bc_mean_ms = [
            (float(a)+float(b))/2 for a,b in zip(timestamps_bc_0_ms, timestamps_bc_1_ms)
        ]
        
        b01 = []
        b0a = []
        ab1 = []
        bma = []
        for i in range(len(timestamps_bc_0_ms)):
            b01.append(timestamps_bc_1_ms[i] - timestamps_bc_0_ms[i])
            b0a.append(timestamps_alr_ms[i] - timestamps_bc_0_ms[i])
            ab1.append(timestamps_bc_1_ms[i] - timestamps_alr_ms[i])
            bma.append(timestamps_alr_ms[i] - timestamps_bc_mean_ms[i])

        if self.plot:
            fig, axs = plt.subplots(2, 1, sharex=True)
            axs[0].plot(bma, label="Time difference ALR-BioCam")
            axs[0].legend()
            axs[0].set_ylabel("Time difference (ms)") 
            axs[1].plot(b01, label="Poll time from ALR to reply received")
            axs[1].legend()
            axs[1].set_ylabel("Duration (ms)")
            plt.xlabel("Sample number")
            plt.show()
        
        # Parametrise conversion from ALR to BioCam time
        # BioCam time = slope * ALR time + offset
        self.offsets_s = [offset/1000. for offset in timestamps_bc_mean_ms]
        self.timestamps_alr_s = [t_alr_ms/1000. for t_alr_ms in timestamps_alr_ms]
        self.slopes = []
        for i in range(len(timestamps_alr_ms)-1):
            delta_t_bc_mean_ms = timestamps_bc_mean_ms[i+1] - timestamps_bc_mean_ms[i]
            delta_t_alr_ms = timestamps_alr_ms[i+1] - timestamps_alr_ms[i]
            self.slopes.append(delta_t_bc_mean_ms/delta_t_alr_ms)
    

    def check_if_inside_modelled_timespan(self, alr_timestamp_start, alr_timestamp_end):
        alr_timestamp_start_s = alr_timestamp_start / self.divisions_per_second
        alr_timestamp_end_s = alr_timestamp_end / self.divisions_per_second
        if alr_timestamp_start_s < self.timestamps_alr_s[0]:
            print(
                "Warning: The first timestamp in the current file is "
                f"{self.timestamps_alr_s[0] - alr_timestamp_start_s:.1f}s before the "
                "range of timestamps used to model the time difference. Extraploating "
                "the model based on the first two points."
            )
        if alr_timestamp_end_s > self.timestamps_alr_s[-1]:
            print(
                "Warning: The last timestamp in the current file is "
                f"{alr_timestamp_end_s - self.timestamps_alr_s[-1]:.1f}s after the "
                "range of timestamps used to model the time difference. Extraploating "
                "the model based on the last two points."
            )


    def correct_timestamp(self, alr_timestamp):
        alr_timestamp_s = alr_timestamp / self.divisions_per_second
        # Find the two closest ALR timestamps
        j = 0
        for i in range(len(self.timestamps_alr_s)-1):
            if alr_timestamp_s >= self.timestamps_alr_s[i]:
                j = i

        t = (alr_timestamp_s-self.timestamps_alr_s[j])*self.slopes[j]+self.offsets_s[j]
        return t

    def correct_timestamps(self):
        if self.alr_timesamped_file is None:
            print("No ALR timestamped file(s) provided.")
            return

        for atf_in, out in zip(self.alr_timesamped_file, self.output_files):
            print(f"Reading ALR timestamped file from {atf_in[0]}")
            df = pd.read_csv(atf_in[0])
            if self.timestamp_col not in df.columns:
                print(
                    f"{atf_in[0]} has no column named {self.timestamp_col}. Aborting."
                )
                quit()
            print(f"Correcting timestamps")
            self.check_if_inside_modelled_timespan(
                df[self.timestamp_col][0], df[self.timestamp_col].iat[-1]
            )
            df.insert(0, "corrected_timestamp", np.nan)
            df["corrected_timestamp"] = df[self.timestamp_col].apply(
                self.correct_timestamp
            )
            print(f"Saving corrected timestamps to {out}")
            df.to_csv(out, index=False)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "This script determines the time difference between BioCam and ALR, and "
            "corrects ALR timestamped files, and/or plots the time difference. "
            "The ALR timestamped file(s) (if provided) must be csv files containing a "
            "column named 'timestamp' containing epoch timestamps. The corrected "
            "timestamps are written to a new column named 'corrected_timestamp' and "
            "the files are saved with the same names as the original files, with "
            "'_corrected' appended before the filename extension."	
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "biocam_serial_log",
        nargs="+", action="append",
        help=(
            "BioCam serial log file(s). If there is more than one serial log file for "
            "an ALR mission, indicate all of them to obtain the best time correction "
            "across the entire dive. Must be indicated in temporal order."
        )
    )
    parser.add_argument(
        "-a", "--alr_timesamped_file",
        nargs="+", action="append",
        help="File(s) timestamped by ALR"
    )
    parser.add_argument(
        "-t", "--timestamp_col", default="timestamp",
        help=(
            "Name of the column containing the timestamps. The same is assumed for all "
            "ALR timestamped files provided."
        )
    )
    parser.add_argument(
        "-d", "--divisions_per_second", type=float, default=1,
        help=(
            "Number of divisions per second in the ALR timestamped file(s), "
            "e.g. 1 for seconds, 1000 for milliseconds, 1000000 for microseconds. "
            "The same is assumed for all ALR timestamped files provided."
        )
    )
    parser.add_argument(
        "-p", "--plot", action="store_true",
        help="Plot the time difference between BioCam and ALR"
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite of existing files"
    )
    args = parser.parse_args()

    tc = TimeCorrector(
        args.biocam_serial_log[0],
        args.alr_timesamped_file,
        args.timestamp_col,
        args.divisions_per_second,
        args.plot,
        args.force
    )
    tc.populate_output_file_path()
    tc.analyse_time_differences()
    tc.correct_timestamps()
    print("Done.")


if __name__ == "__main__":
    main()
