from pathlib import Path


def change_subfolder(path, prior, new):
    index = path.parts.index(prior)
    parts = list(path.parts)
    parts[index] = new
    new_path = Path(*parts)
    if not new_path.exists():
        new_path.mkdir(exist_ok=True, parents=True)
    return new_path


def get_folder(path, name):
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
    p = Path(path)
    return get_folder(p, "configuration")


def get_raw_folder(path):
    p = Path(path)
    return get_folder(p, "raw")


def get_processed_folder(path):
    p = Path(path)
    return get_folder(p, "processed")


def _copy(self, target):
    import shutil
    assert self.is_file()
    shutil.copy(str(self), str(target))  # str() only there for Python < (3, 6)


Path.copy = _copy
