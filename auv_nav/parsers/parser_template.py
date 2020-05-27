from oplab import Console
from auv_nav.sensors import Category, Orientation


def parse_template(mission, vehicle, category, output_format, outpath):
    # Get your data from a file using mission paths, for example
    your_data = None

    # Let's say you want a new IMU, instance the measurement to work
    orientation = Orientation()

    data_list = []
    if category == Category.ORIENTATION:
        Console.info('... parsing orientation')
        for i in your_data:
            # Provide a parser in the sensors.py class
            orientation.from_your_data(i)
            data = orientation.export(output_format)
            data_list.append(data)

    return data_list