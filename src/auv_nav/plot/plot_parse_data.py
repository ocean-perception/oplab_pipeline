# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import json
import sys
import time

import plotly.graph_objs as go
import plotly.offline as py
from prettytable import ALL, PrettyTable

from auv_nav.tools.time_conversions import (
    epoch_to_localtime,
    epoch_to_utctime,
    get_localtimezone,
)

"""
This looks at nav_standard.json file stored in the format of
[{"<DataName>":<Value>,"<DataName>":<Value>, ...}, ...]
and displays the information such as 'Categories' in it, start date time,
finish date time, etc...

Display infomation of json file
    inputs are
        data_check.py <options>
            -i <path to nav folder containing json file>
example input:
    python data_check.py -i //OPLAB-SURF/processed/2017/SSK17-01/ts_un_006/

"""


# Helper functions
def print_lines(object, time_difference):
    line = ["", ""]
    for i in object:
        if "data" in i:
            line[0] += i + "\n"
            index = 0
            for j in object[i]:
                if index == 0:
                    line[1] += str(j) + "\n"
                    index += 1
                else:
                    line[1] += str(j) + "\n"
                    line[0] += "\n"
        else:
            line[0] += i + "\n"
            line[1] += str(object[i]) + "\n"
    line[0] += "Approximate update rate (Hz)"
    line[1] += str(1 / time_difference)
    if "frame" in object:
        if object["frame"] != "body":
            if object["frame"] != "inertial":
                line[0] += "Warning\n"
                line[1] += "Multiple " "frame" "in this category\n"
    return line


def create_trace(x_list, y_list, text_list, trace_name):
    trace = go.Scattergl(
        x=x_list,
        y=y_list,
        text=text_list,
        name=trace_name,
        mode="markers",
        hoverinfo="text",
    )
    return trace


def data2trace(i, j, t, plotlytable_info_list):
    title = j[0]["category"] + " - " + j[0]["frame"]
    # print the info
    time_difference = j[1]["epoch_timestamp"] - j[0]["epoch_timestamp"]
    n = 0
    # print (time_difference, j[0]['epoch_timestamp'], j[1]['epoch_timestamp'])
    while time_difference == 0 or time_difference < 0.002 and n < len(j):
        n += 1
        time_difference = j[1 + n]["epoch_timestamp"] - j[n]["epoch_timestamp"]
        # print (time_difference, n, len(j))
        if n > 10:
            sys.exit(0)
    line = print_lines(j[0], time_difference)
    t.add_row([j[0]["category"], len(j), line[0], line[1]])

    # plotly table
    plotlytable_info_list[0].append(j[0]["category"])
    plotlytable_info_list[1].append(len(j))
    plotlytable_info_list[2].append(line[0].replace("\n", "<br>"))
    plotlytable_info_list[3].append(line[1].replace("\n", "<br>"))

    # plotly plot
    # format is 'yyyy-mm-dd HH:MM:SS.ssssss'
    x_values = [
        time.strftime("%Y-%m-%d %H:%M:%S", epoch_to_utctime(k["epoch_timestamp"]))
        + ".{}".format(
            ("{:.6f}".format(k["epoch_timestamp"] - int(k["epoch_timestamp"])))[2:9]
        )
        for k in j
    ]
    text_list = [
        time.strftime("%Y-%m-%d %H:%M:%S", epoch_to_utctime(k["epoch_timestamp"]))
        + "(UTC) | "
        + time.strftime("%Y-%m-%d %H:%M:%S", epoch_to_localtime(k["epoch_timestamp"]))
        + "(Local)"
        for k in j
    ]  # add utc time too.
    # x_values = [k['epoch_timestamp'] for k in j]
    y_values = [title] * len(x_values)
    return x_values, y_values, text_list, title


def plot_parse_data(filepath, ftype="oplab"):
    """
    Goes through each data element in json file and first check for different
    types category, and then different types of frame (body or inertial).
    Displays a sample information for each different type of data. In the
    future if element in json file contains more variation other than different
    category/frame, need to expand code to check for variations in them too.
    """
    if ftype == "oplab":
        print("Loading json file")
        # Contains all the 'category' in the json file.
        ct_lst = []  # category_list
        # Contains the same number of elements as <category_list>, each
        # containing additional list of elements that are data from different
        # 'frame'.
        fdt_lst = []  # full_data_list
        start_time = 0
        finish_time = 0

        # Loads and sorts data elements into 'category' and 'frame'.
        fn = filepath / "nav_standard.json"
        with fn.open("r", encoding="utf-8") as json_file:
            data_in = json.load(json_file)
            start_time = data_in[1]["epoch_timestamp"]
            finish_time = data_in[-1]["epoch_timestamp"]
            for i in data_in:
                if i is None:
                    continue
                if i["category"] == "origin":
                    continue
                # to find out how many categories are there
                if i["category"] in ct_lst:
                    # to record all different types of frames
                    if i["frame"] == "body":
                        fdt_lst[ct_lst.index(i["category"])][0].append(i)
                    elif i["frame"] == "inertial":
                        fdt_lst[ct_lst.index(i["category"])][1].append(i)
                    else:
                        if not fdt_lst[ct_lst.index(i["category"])][2]:
                            fdt_lst[ct_lst.index(i["category"])][2].append(i)
                            print(
                                "Warning: %s"
                                "s frame contains something \
                                  different than body or inertial --> %s"
                                % (i["category"], i)
                            )
                        else:
                            flag_same_frame = 0
                            for j in fdt_lst[ct_lst.index(i["category"])][2:]:
                                if j["frame"] == i["frame"]:
                                    flag_same_frame = 1
                                    j.append(i)
                            if flag_same_frame == 0:
                                fdt_lst[ct_lst.index(i["category"])].append([])
                                fdt_lst[ct_lst.index(i["category"])][-1].append(i)
                                print(
                                    "Warning: %s"
                                    "s frame contains more than \
                                      1 different obeject other than body and \
                                      inertial --> %s"
                                    % (i["category"], i)
                                )
                else:
                    ct_lst.append(i["category"])
                    fdt_lst.append([[], [], []])

                    # to record all different types of frames
                    if i["frame"] == "body":
                        fdt_lst[ct_lst.index(i["category"])][0].append(i)
                    elif i["frame"] == "inertial":
                        fdt_lst[ct_lst.index(i["category"])][1].append(i)
                    else:
                        fdt_lst[ct_lst.index(i["category"])][2].append(i)
                        print(
                            "Warning: %s"
                            "s frame contains something \
                              different than body or inertial --> %s"
                            % (i["category"], i)
                        )

        # Create a table of each data 'category' and 'frame' variation, with
        # additional approximation of how frequent the sensors collects data
        # through calculating the difference between the first two
        # epoch_timestamp.
        print("Creating table")
        t = PrettyTable(["Category", "No. of data", "Details", "Sample Value"])
        epoch_timestamp_data_points = []
        titles = []
        trace_list = []
        plotlytable_info_list = [[], [], [], []]
        for i in fdt_lst:
            for j in i:
                if not j:
                    pass
                else:
                    #
                    x, y, text_list, title = data2trace(i, j, t, plotlytable_info_list)
                    titles.append(title)
                    epoch_timestamp_data_points.append(j)
                    trace_list.append(create_trace(x, y, text_list, title))

        table_trace = go.Table(
            columnorder=[1, 2, 3, 4],
            columnwidth=[1, 1, 2, 5],  # [80,400]
            header=dict(
                values=[
                    ["<b>Category</b>"],
                    ["<b>No. of data</b>"],
                    ["<b>Details</b>"],
                    ["<b>Sample Value</b>"],
                ],
                line=dict(color="#506784"),
                fill=dict(color="#119DFF"),
                align=["center", "center", "center", "left"],
                font=dict(color="white", size=12),
                height=40,
            ),
            cells=dict(
                values=plotlytable_info_list,
                line=dict(color="#506784"),
                fill=dict(color=["#25FEFD", "white"]),
                align=["center", "center", "center", "left"],
                font=dict(color="#506784", size=12),
                height=30,
            ),
        )

        layout_table = go.Layout(
            title="Json Data Info Table<br>Start time is: %s (%s), %s (%s), \
                %d (epoch)"
            "<br>Finish time is: %s (%s), %s (%s), %d (epoch)"
            % (
                time.strftime("%Y-%m-%d %H:%M:%S", epoch_to_localtime(start_time)),
                get_localtimezone(),
                time.strftime("%Y-%m-%d %H:%M:%S", epoch_to_utctime(start_time)),
                "UTC",
                start_time,
                time.strftime("%Y-%m-%d %H:%M:%S", epoch_to_localtime(finish_time)),
                get_localtimezone(),
                time.strftime("%Y-%m-%d %H:%M:%S", epoch_to_utctime(finish_time)),
                "UTC",
                finish_time,
            )
        )
        table_fig = go.Figure(data=[table_trace], layout=layout_table)
        py.plot(
            table_fig,
            filename=str(filepath / "json_data_info.html"),
            auto_open=False,
        )

        layout = go.Layout(
            # width=950,
            # height=800,
            title="Timestamp History Plot<br>Start time is: %s (%s), %s (%s), \
                %d (epoch)"
            "<br>Finish time is: %s (%s), %s (%s), %d (epoch)"
            % (
                time.strftime("%Y-%m-%d %H:%M:%S", epoch_to_localtime(start_time)),
                get_localtimezone(),
                time.strftime("%Y-%m-%d %H:%M:%S", epoch_to_utctime(start_time)),
                "UTC",
                start_time,
                time.strftime("%Y-%m-%d %H:%M:%S", epoch_to_localtime(finish_time)),
                get_localtimezone(),
                time.strftime("%Y-%m-%d %H:%M:%S", epoch_to_utctime(finish_time)),
                "UTC",
                finish_time,
            ),
            hovermode="closest",
            xaxis=dict(
                title="Date time (%s)" % ("UTC"),
                rangeselector=dict(
                    buttons=list(
                        [
                            dict(
                                count=5,
                                label="5 secs",
                                step="second",
                                stepmode="backward",
                            ),
                            dict(
                                count=60,
                                label="1 min",
                                step="second",
                                stepmode="backward",
                            ),
                            dict(
                                count=300,
                                label="5 mins",
                                step="second",
                                stepmode="backward",
                            ),
                            dict(
                                count=1200,
                                label="20 mins",
                                step="second",
                                stepmode="backward",
                            ),
                            dict(
                                count=3600,
                                label="1 hour",
                                step="second",
                                stepmode="backward",
                            ),
                            dict(
                                count=7200,
                                label="2 hours",
                                step="second",
                                stepmode="backward",
                            ),
                            dict(step="all"),
                        ]
                    )
                ),
                rangeslider=dict(thickness=0.05),
                type="date",
            ),
            yaxis=dict(
                title="Category-Frame",
            ),
            dragmode="pan",
            margin=go.layout.Margin(l=150),  # noqa
        )
        config = {"scrollZoom": True}
        fig = go.Figure(data=list(reversed(trace_list)), layout=layout)
        py.plot(
            fig,
            config=config,
            filename=str(filepath / "timestamp_history.html"),
            auto_open=False,
        )

        start_end_text = (
            "Start time is: %s (%s), %s (%s), %d (epoch)\nFinish time is: "
            "%s (%s), %s (%s), %d (epoch)\n"
            % (
                time.strftime("%Y-%m-%d %H:%M:%S", epoch_to_localtime(start_time)),
                get_localtimezone(),
                time.strftime("%Y-%m-%d %H:%M:%S", epoch_to_utctime(start_time)),
                "UTC",
                start_time,
                time.strftime("%Y-%m-%d %H:%M:%S", epoch_to_localtime(finish_time)),
                get_localtimezone(),
                time.strftime("%Y-%m-%d %H:%M:%S", epoch_to_utctime(finish_time)),
                "UTC",
                finish_time,
            )
        )  # changed from gmtime to localtime

        t.align["Sample Value"] = "l"
        t.hrules = ALL
        print(start_end_text)
        print(t)
        print("Outputs saved to {}".format(filepath))
    else:
        print("ACFR ftype to be done")
