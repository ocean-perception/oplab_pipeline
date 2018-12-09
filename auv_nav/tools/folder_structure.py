from pathlib import Path


def change_subfolder(path, prior, new):
    path = path.absolute()
    index = path.parts.index(prior)
    parts = list(path.parts)
    parts[index] = new
    new_path = Path(*parts)
    if not new_path.exists():
        dummy_path = Path(*parts[:-1])
        print('The path {} does not exist. I am creating it for you.'.format(path))
        dummy_path.mkdir(exist_ok=True, parents=True)
    return new_path


def get_folder(path, name):
    if not isinstance(path, Path):
        path = Path(path)
    if name in path.parts:
        return path
    elif 'processed' in path.parts:
        return change_subfolder(path, 'processed', name)
    elif 'raw' in path.parts:
        return change_subfolder(path, 'raw', name)
    elif 'configuration' in path.parts:
        return change_subfolder(path, 'configuration', name)
    else:
        print("The folder {0} does not belong to \
               any dataset folder structure.".format(
                str(path)))


def get_config_folder(path):
    return get_folder(path, "configuration")


def get_raw_folder(path):
    return get_folder(path, "raw")


def get_processed_folder(path):
    return get_folder(path, "processed")


def _copy(self, target):
    import shutil
    assert self.is_file()
    shutil.copy(str(self), str(target))  # str() only there for Python < (3, 6)


Path.copy = _copy
