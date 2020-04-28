from pathlib import Path
import yaml
from auv_nav.tools.console import Console
from datetime import datetime
import calendar
import pandas as pd
from auv_nav.tools.folder_structure import get_raw_folder


def resolve(filename, path_to_mission):
    if path_to_mission is None:
        curr_dir = Path.cwd()
    else:
        curr_dir = path_to_mission
    curr_dir = get_raw_folder(curr_dir)
    print(curr_dir)
    print(filename)
    resolved_filename = ''
    for x in curr_dir.glob(filename):
        resolved_filename = x
    if resolved_filename == '':
        Console.error('The file: ', filename, ' could not be found.')
        Console.quit('Invalid timestamp file or format')
    return resolved_filename


class FilenameToDate:
    def __init__(self, stamp_format: str, filename=None, columns=None, path_to_mission=None):
        self.stamp_format = stamp_format
        self.df = None
        if filename is not None and columns is not None:
            self.filename = resolve(filename, path_to_mission)
            self.read_timestamp_file(self.filename, columns)

    # Make the object callable  (e.g. operator() )
    def __call__(self, filename: str):
        # Get the name without extension
        filename = Path(filename)
        stamp_format = Path(self.stamp_format)
        filename = filename.stem
        stamp_format = stamp_format.stem
        return self.string_to_epoch(filename, self.stamp_format)

    def string_to_epoch(self, filename, stamp_format):
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
            if f == 'Y':
                year += n
            if f == 'M':
                month += n
            if f == 'D':
                day += n
            if f == 'h':
                hour += n
            if f == 'm':
                minute += n
            if f == 's':
                second += n
            if f == 'f':
                msecond += n
            if f == 'u':
                usecond += n
            if f == 'i':
                index += n
        if not index:
            assert len(year) == 4, 'Year in filename should have a length of 4'
            assert len(month) == 2, 'Month in filename should have a length of \
                2'
            assert len(day) == 2, 'Day in filename should have a length of 2'
            assert len(hour) == 2, 'Hour in filename should have a length of 2'
            assert len(minute) == 2, 'Minute in filename should have a length \
                of 2'
            assert len(second) == 2, 'Second in filename should have a length \
                of 2'
            if msecond:
                assert len(msecond) == 3, 'Milliseconds in filename should \
                    have a length of 3'
            else:
                msecond = '0'
            if usecond:
                assert len(usecond) == 3, 'Microseconds in filename should \
                    have a length of 3'
            else:
                usecond = '0'
            microsecond = int(msecond)*1000 + int(usecond)
            date = datetime(int(year), int(month), int(day),
                            int(hour), int(minute), int(second), microsecond)
            stamp = float(calendar.timegm(date.timetuple()))
            return stamp + microsecond*1e-6
        else:
            if self.df is None:
                Console.error('FilenameToDate specified using indexing, but no \
                    timestamp file has been provided or read.')
                Console.quit('Invalid timestamp format')
            stamp = self.df['epoch_timestamp'][int(index)]
            return stamp

    def read_timestamp_file(self, filename, columns):
        df = pd.read_csv(filename, dtype=str)
        df['combined'] = ''
        df['combined_format'] = ''
        df['epoch_timestamp'] = ''
        df_index_name = None
        for c in columns:
            name = c['name']
            content = c['content']
            # If it is not index columns, concatenate all columns into one
            if 'i' not in content:
                df['combined'] += df[name].astype(str)
                df['combined_format'] += content
                df.drop(name, axis=1)
            else:
                if df_index_name is None:
                    df_index_name = name
                else:
                    Console.error("There should only be one Index column")
                    Console.quit("Invalid timestamp format")

        last_idx = int(df['index'].tail(1))
        Console.info('Found', last_idx, 'timestamp records in', filename)

        for index, row in df.iterrows():
            row['epoch_timestamp'] = self.string_to_epoch(
                row['combined'], row['combined_format'])

        df = df.drop('combined', axis=1)
        df = df.drop('combined_format', axis=1)
        df[df_index_name] = df[df_index_name].astype(int)
        self.df = df.set_index(df_index_name)
        print(self.df)


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
