

def is_subfolder_of(path, folder_name):
    path = path.replace('\\', '/')
    sub_path = path.split('/')
    for i in range(len(sub_path)):
        if sub_path[i] == folder_name:
            return True
    return False


def get_folder(path, name):
    if is_subfolder_of(path, name):
        return path
    elif is_subfolder_of(path, "processed"):
        return path.replace("processed", name)
    elif is_subfolder_of(path, "raw"):
        return path.replace("raw", name)
    elif is_subfolder_of(path, "configuration"):
        return path.replace("configuration", name)
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
