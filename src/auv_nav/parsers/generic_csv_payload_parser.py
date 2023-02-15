from auv_nav.sensors import Other
from oplab.console import Console
from oplab.filename_to_date import FilenameToDate


def generic_csv_payload_parser(filepath, columns_dict, timeoffset_s=0):
    data_list = []
    # Use FilenameToDate just to get the timestamp
    # TODO: Make a generic timestamp parser out of FilenameToDate
    f2d = FilenameToDate(None, filepath, columns_dict)
    df = f2d.df

    # Check if the timestamp is defined in epoch time
    # if the value of the only entry in columns_dict is "e" then the timestamp is in epoch time
    if "e" in list(columns_dict.values())[0]:
        if timeoffset_s > 3600 or timeoffset_s < -3600:
            Console.warning(
                "Time offset larger than 1h are ignored when parsing epoch timestamps"
            )
            # Get the remainder of the division by 3600 (seconds in an hour)
            timeoffset_s = timeoffset_s % 3600

    for _, row in df.iterrows():
        data = Other()
        data.timestamp = row["epoch_timestamp"] + timeoffset_s
        for key, _ in columns_dict.items():
            if key == "epoch_timestamp":
                continue
            data.data[key] = row[key]
        data_list.append(data)
    return data_list
