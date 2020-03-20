from pathlib import Path
import yaml
from datetime import datetime, timedelta
import pandas as pd
import yaml


def string_to_date(filename: str, stamp_format: str):
    year = ''
    month = ''
    day = ''
    hour = ''
    minute = ''
    second = ''
    msecond = ''
    usecond = ''
    index = ''
    for n, f in zip(filename, stamp_format):
        if f is 'Y':
            year += n
        if f is 'M':
            month += n
        if f is 'D':
            day += n
        if f is 'h':
            hour += n
        if f is 'm':
            minute += n
        if f is 's':
            second += n
        if f is 'f':
            msecond += n
        if f is 'u':
            usecond += n
        if f is 'i':
            index += n
    if not index:
        assert len(year) == 4, 'Year in filename should have a length of 4'
        assert len(month) == 2, 'Month in filename should have a length of 2'
        assert len(day) == 2, 'Day in filename should have a length of 2'
        assert len(hour) == 2, 'Hour in filename should have a length of 2'
        assert len(minute) == 2, 'Minute in filename should have a length of 2'
        assert len(second) == 2, 'Second in filename should have a length of 2'
        if msecond:
            assert len(msecond) == 3, 'Milliseconds in filename should have a length of 3'
        else:
            msecond = '0'
        if usecond:
            assert len(usecond) == 3, 'Microseconds in filename should have a length of 3'
        else:
            usecond = '0'
        microsecond = int(msecond)*1000 + int(usecond)
        date = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), microsecond)
        return date
    else:
        return int(index)


def filename_to_date(filename: str, stamp_format: str):
    # Get the name without extension
    filename = Path(filename)
    stamp_format = Path(stamp_format)
    filename = filename.stem
    stamp_format = stamp_format.stem
    return string_to_date(filename, stamp_format)


def read_timestamp_file(filename, columns):
    result = []

    df = pd.read_csv(filename, dtype=str)

    df['combined'] = ''
    df['combined_format'] = ''
    df['datetime'] = ''
    for c in columns:
        name = c['name']
        content = c['content']
        # If it is not index columns, concatenate all columns into one
        if 'i' not in content:
            df['combined'] += df[name].astype(str)
            df['combined_format'] += content

    last_idx = int(df['index'].tail(1))
    print(last_idx)

    for index, row in df.iterrows():
        row['datetime'] = string_to_date(row['combined'], row['combined_format'])

    df = df.drop('combined', axis=1)
    df = df.drop('combined_format', axis=1)
    print(df)


if __name__ == '__main__':

    a = filename_to_date('0000245.raw', 'iiiiiii.xxx')
    print(a)
    a = filename_to_date('image0000246.tif', 'xxxxxiiiiiii.xxx')
    print(a)
    a = filename_to_date('PR_20180811_153729_762_RC16.tif', 'xxxYYYYMMDDxhhmmssxfffxxxxx.xxx')
    print(a)
    a = filename_to_date('20190913_101347_962382_20190913_101346_411014_pcoc.tif', 'YYYYMMDDxhhmmssxfffuuuxxxxxxxxxxxxxxxxxxxxxxxxxxxx.xxx')
    print(a)

    f = open('example.yaml')
    a = yaml.safe_load(f)
    read_timestamp_file(a['timestamp_file'], a['columns'])
