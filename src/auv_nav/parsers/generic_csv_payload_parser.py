from auv_nav.sensors import Payload
from oplab.console import Console
from oplab.filename_to_date import FilenameToDate

import pandas as pd

def generic_csv_payload_parser(filepath, columns_dict, timeoffset_s=0):
    data_list = []
    # Use FilenameToDate just to get the timestamp
    # TODO: Make a generic timestamp parser out of FilenameToDate
    #f2d = FilenameToDate(None, filepath, columns_dict)
    #df = f2d.df

    df = pd.read_csv(filepath)
    df["epoch_timestamp"] = df["corrected_timestamp"]

    for index, row in df.iterrows():
        data = Payload()
        data.data = {}
        data.epoch_timestamp = row["epoch_timestamp"] + timeoffset_s
        for key in df.columns:
            if key == "epoch_timestamp":
                continue
            data.data[key] = row[key]
        data_list.append(data)
    return data_list
