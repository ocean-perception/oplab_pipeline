# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import shutil
from pathlib import Path
from typing import Optional, Union

from .console import Console


def check_exists(p):
    return Path(p).exists()


def valid_dive(p):
    p = Path(p)
    a = check_exists(p)
    if a:
        b = check_exists(p / "mission.yaml")
        c = check_exists(p / "vehicle.yaml")
    return a and b and c


def change_subfolder(path: Path, prior: str, new: str) -> Path:
    # path = path.resolve(strict=False)
    index = path.parts.index(prior)
    parts = list(path.parts)
    parts[index] = new
    new_path = Path(*parts)
    if new_path.is_dir():
        if not new_path.exists():
            dummy_path = Path(*parts[:-1])
            Console.info(
                "The path",
                path,
                "does not exist. I am creating ",
                "it for you.",
            )
            dummy_path.mkdir(exist_ok=True, parents=True)
    elif new_path.is_file():
        # check if parent directories are created
        if not new_path.parent.exists():
            new_path.parent.mkdir(exist_ok=True, parents=True)
    return new_path


def get_folder(path: Union[str, Path], name: str) -> Optional[Path]:
    path = Path(path).resolve(strict=False)
    if name in path.parts:
        return path
    elif "processed" in path.parts:
        return change_subfolder(path, "processed", name)
    elif "raw" in path.parts:
        return change_subfolder(path, "raw", name)
    elif "configuration" in path.parts:
        return change_subfolder(path, "configuration", name)
    else:
        Console.quit(
            "The folder",
            str(path),
            "does not belong to any dataset",
            "folder structure.",
        )


def get_file_list(directory):
    dirpath = Path(directory)
    file_list = []
    for x in dirpath.iterdir():
        if x.is_file():
            file_list.append(x)
        elif x.is_dir():
            file_list.extend(get_file_list(x))
    return file_list


def get_config_folder(path: Union[str, Path]) -> Optional[Path]:
    return get_folder(path, "configuration")


def get_raw_folder(path: Union[str, Path]) -> Optional[Path]:
    return get_folder(path, "raw")


def get_processed_folder(path: Union[str, Path]) -> Optional[Path]:
    return get_folder(path, "processed")


def check_dirs_exist(dirs):
    if isinstance(dirs, list):
        for d in dirs:
            if not d.is_dir():
                return False
    else:
        if not dirs.is_dir():
            return False
    return True


def get_raw_folders(dirs):
    if isinstance(dirs, list):
        dir_list = []
        for d in dirs:
            dir_list.append(get_raw_folder(d))
        return dir_list
    else:
        return get_raw_folder(dirs)


def get_processed_folders(dirs):
    if isinstance(dirs, list):
        dir_list = []
        for d in dirs:
            dir_list.append(get_processed_folder(d))
        return dir_list
    else:
        return get_processed_folder(dirs)


def remove_directory(folder: Path):
    """Remove a specific directory recursively

    Parameters
    -----------
    folder : Path
        folder to be removed
    """
    for p in folder.iterdir():
        if p.is_dir():
            remove_directory(p)
        else:
            p.unlink()
    folder.rmdir()


def _copy(self, target):
    if not target.parent.exists():
        target.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy(str(self), str(target))  # str() only there for Python < (3, 6)


Path.copy = _copy
