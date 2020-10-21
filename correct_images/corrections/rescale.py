import imageio
from PIL import Image
from oplab import get_processed_folder
from oplab import Console
from oplab import MonoCamera
from pathlib import Path
from tqdm import trange
import pandas as pd
import os


def get_image_and_info(image_path):
    image = imageio.imread(image_path)
    image_shape = image.shape
    return image, image_shape[1], image_shape[0]


# uses given opening angle of camera and the altitude parameter to determine the pixel size
# give width & horizontal, or height & vertical
def get_pixel_size(altitude, image_size, f):
    image_spatial_size = float(altitude) * image_size / f
    pixel_size = image_spatial_size / image_size
    return pixel_size


def rescale(image_path, interpolate_method, target_pixel_size, altitude, f_x, f_y, maintain_pixels):
    image, image_width, image_height = get_image_and_info(image_path)
    pixel_height = get_pixel_size(altitude, image_height, f_y)
    pixel_width = get_pixel_size(altitude, image_width, f_x)

    vertical_rescale = pixel_height / target_pixel_size
    horizontal_rescale = pixel_width / target_pixel_size

    method = None
    if interpolate_method == "bicubic":
        method = Image.BICUBIC
    elif interpolate_method == "bilinear":
        method = Image.BILINEAR
    elif interpolate_method == "nearest_neighbour":
        method = Image.NEAREST
    elif interpolate_method == "lanczos":
        method = Image.LANCZOS

    if maintain_pixels == "N" or maintain_pixels == "No":
        size = (int(image_width * horizontal_rescale), int(image_height * vertical_rescale))
        image = Image.fromarray(image.astype("uint8"), "RGB")
        image = image.resize(size, resample=method)

    elif maintain_pixels == "Y" or maintain_pixels == "Yes":
        if vertical_rescale < 1 or horizontal_rescale < 1:
            size = (int(image_width * horizontal_rescale), int(image_height * vertical_rescale))
            image = Image.fromarray(image.astype("uint8"), "RGB")
            image = image.resize(size, resample=method)
            size = (image_width, image_height)
            image = image.resize(size, resample=method)
        else:
            crop_width = int(( 1 /horizontal_rescale) * image_width)
            crop_height = int(( 1 /vertical_rescale) * image_height)

            # find crop box dimensions
            box_left = int((image_width - crop_width) / 2)
            box_upper = int((image_height - crop_height) / 2)
            box_right = image_width - box_left
            box_lower = image_height - box_upper

            # crop the image to the center
            box = (box_left, box_upper, box_right, box_lower)
            image = Image.fromarray(image.astype("uint8"), "RGB")
            cropped_image = image.crop(box)

            # resize the cropped image to the size of original image
            size = (image_width, image_height)
            image = image.resize(size, resample=method)

    return image


def rescale_images(imagenames_list,
                   image_directory,
                   interpolate_method,
                   target_pixel_size,
                   dataframe,
                   output_directory,
                   f_x,
                   f_y,
                   maintain_pixels):
    Console.info("Rescaling images...")

    for idx in trange(len(imagenames_list)):
        image_name = imagenames_list[idx]
        source_image_path = Path(image_directory) / image_name
        output_image_path = Path(output_directory) / image_name
        image_path_list = dataframe["relative_path"]
        trimmed_path_list = [path for path in image_path_list if Path(path).stem in image_name]
        trimmed_dataframe = dataframe.loc[dataframe["relative_path"].isin(trimmed_path_list)]
        altitude = trimmed_dataframe["altitude [m]"]
        if len(altitude) > 0:
            rescaled_image = rescale(
                source_image_path, interpolate_method, target_pixel_size, altitude, f_x, f_y, maintain_pixels
            )
            imageio.imwrite(output_image_path, rescaled_image, format="PNG-FI")
        else:
            Console.warn("Did not get distance values for image: " + image_name)


def rescale_camera(path, camera_system, camera):
    name = camera.camera_name
    distance_path = camera.distance_path
    interpolate_method = camera.interpolate_method
    image_path = camera.path
    target_pixel_size = camera.target_pixel_size
    maintain_pixels = camera.maintain_pixels
    output_folder = camera.output_folder

    idx = [
        i
        for i, camera in enumerate(camera_system.cameras)
        if camera.name == name
    ]

    if len(idx) > 0:
        Console.info("Camera found in camera.yaml file...")
    else:
        Console.warn("Camera not found in camera.yaml file. Please provide a relevant camera.yaml file...")
        return False

    # obtain images to be rescaled
    path_processed = get_processed_folder(path)
    image_path = path_processed / image_path

    # obtain path to distance / altitude values
    full_distance_path = path_processed / distance_path
    full_distance_path = full_distance_path / "csv" / "ekf"
    distance_file = "auv_ekf_" + name + ".csv"
    distance_path = full_distance_path / distance_file

    # obtain focal lengths from calibration file
    camera_params_folder = path_processed / "calibration"
    camera_params_filename = "mono_" + name + ".yaml"
    camera_params_file_path = camera_params_folder / camera_params_filename

    if not camera_params_file_path.exists():
        Console.quit("Calibration file not found...")
    else:
        Console.info("Calibration file found...")

    monocam = MonoCamera(camera_params_file_path)
    focal_length_x = monocam.K[0, 0]
    focal_length_y = monocam.K[1, 1]

    # create output path
    output_directory = path_processed / output_folder
    if not output_directory.exists():
        output_directory.mkdir(parents=True)

    # call rescale function
    dataframe = pd.read_csv(Path(distance_path))
    imagenames_list = [
        filename
        for filename in os.listdir(image_path)
        if filename[-4:] == ".jpg" or filename[-4:] == ".png" or filename[-4:] == ".tif"
    ]
    Console.info("Distance values loaded...")
    rescale_images(imagenames_list, image_path, interpolate_method, target_pixel_size, dataframe,
                   output_directory, focal_length_x, focal_length_y, maintain_pixels)
    return True