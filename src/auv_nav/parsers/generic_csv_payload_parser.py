import pandas as pd

from auv_nav.sensors import Payload


def generic_csv_payload_parser(filepath, columns_dict, timeoffset_s=0):
    data_list = []
    # Use FilenameToDate just to get the timestamp
    # TODO: Make a generic timestamp parser out of FilenameToDate
    # f2d = FilenameToDate(None, filepath, columns_dict)
    # df = f2d.df

    df = pd.read_csv(filepath, header=None)
    df.columns = [col["name"] for col in columns_dict]
    
#    df["epoch_timestamp"] = df["corrected_timestamp"]

#    if "epoch_timestamp" in df.columns:
#            df["epoch_timestamp"] = df["epoch_timestamp"] / 1e6

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
