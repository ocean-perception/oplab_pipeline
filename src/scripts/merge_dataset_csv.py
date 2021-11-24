from pathlib import Path

import pandas as pd

from oplab import Console, get_processed_folder


def find_navigation_csv(
    base, distance_path="json_renav_*", solution="ekf", camera="Cam51707923"
):
    base_processed = get_processed_folder(base)
    json_list = list(base_processed.glob(distance_path))
    if len(json_list) == 0:
        Console.quit("No navigation solution could be found at", base_processed)
    nav_csv_filepath = json_list[0] / ("csv/" + solution)
    nav_csv_filepath = nav_csv_filepath / (
        "auv_" + solution + "_" + camera + ".csv"
    )  # noqa
    if not nav_csv_filepath.exists():
        Console.quit("No navigation solution could be found at", nav_csv_filepath)
    return nav_csv_filepath


def find_image_path(
    base,
    correct_images="attenuation_correction/developed_*",
    correction="altitude_corrected/m30_std10",
):
    correct_images_list = list(base.glob(correct_images))
    if len(correct_images_list) == 0:
        Console.quit("No correct_images solution could be found at", base)

    image_folder = correct_images_list[0] / correction
    if not image_folder.exists():
        Console.quit(
            "No correct_images solution could \
                     be found at",
            image_folder,
        )
    return image_folder


if __name__ == "__main__":

    base = "/media/oplab-source/processed/2019/koyo19-01/ae2000f/"
    dives = [
        "20190903_094907_ae2000f_sx3",
        "20190907_070131_ae2000f_sx3",
        "20190908_100733_ae2000f_sx3",
        "20190909_071627_ae2000f_sx3",
        "20190912_070039_ae2000f_sx3",
    ]

    frames = []
    for d in dives:
        dive_base = Path(base + d)
        nav_csv_filepath = find_navigation_csv(dive_base)

        df = pd.read_csv(nav_csv_filepath)

        df["relative_path"] = df["relative_path"].str.replace(".raw", ".png")  # noqa

        for i in range(len(df["relative_path"])):
            orig_dir = Path(df["relative_path"].values[i])
            orig_dir = dive_base / orig_dir

            print(orig_dir)

            filename = orig_dir.name
            orig_dir = orig_dir.parent

            df["relative_path"][i] = find_image_path(orig_dir) / filename
        frames.append(df)

    merged_df = pd.concat(frames)
    merged_df.to_csv("merged.csv")
