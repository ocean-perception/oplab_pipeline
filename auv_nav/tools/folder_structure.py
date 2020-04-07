from auv_nav.console import Console
from pathlib import Path


def check_exists(p):
    return Path(p).exists()


def valid_dive(p):
    p = Path(p)
    a = check_exists(p)
    if a:
        b = check_exists(p / "mission.yaml")
        c = check_exists(p / "vehicle.yaml")
    return a and b and c


def change_subfolder(path, prior, new):
    #path = path.resolve(strict=False)
    index = path.parts.index(prior)
    parts = list(path.parts)
    parts[index] = new
    new_path = Path(*parts)
    if new_path.is_dir():
        if not new_path.exists():
            dummy_path = Path(*parts[:-1])
            Console.info('The path {} does not exist. I am creating it for you.'.format(path))
            dummy_path.mkdir(exist_ok=True, parents=True)
    elif new_path.is_file():
        # check if parent directories are created
        if not new_path.parent.exists():
            new_path.parent.mkdir(exist_ok=True, parents=True)
    return new_path


def get_folder(path, name):
    path = Path(path)  # .resolve(strict=False)
    if name in path.parts:
        return path
    elif 'processed' in path.parts:
        return change_subfolder(path, 'processed', name)
    elif 'raw' in path.parts:
        return change_subfolder(path, 'raw', name)
    elif 'configuration' in path.parts:
        return change_subfolder(path, 'configuration', name)
    else:
        Console.error("The folder {0} does not belong to \
               any dataset folder structure.".format(
                str(path)))


def get_file_list(directory):
    dirpath = Path(directory)
    file_list = []
    for x in dirpath.iterdir():
        if x.is_file():
            file_list.append(x)
        elif x.is_dir():
            file_list.extend(searching_all_files(x))
    return file_list


def get_config_folder(path):
    return get_folder(path, "configuration")


def get_raw_folder(path):
    return get_folder(path, "raw")


def get_processed_folder(path):
    return get_folder(path, "processed")


def _copy(self, target):
    import shutil
    assert self.is_file()
    if not target.parent.exists():
        target.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy(str(self), str(target))  # str() only there for Python < (3, 6)


Path.copy = _copy
