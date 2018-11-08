

def is_subfolder_of(path, folder_name):
    path = path.replace('\\', '/')
    sub_path = path.split('/')
    for i in range(len(sub_path)):
        if sub_path[i] == folder_name:
            return True
    return False


def get_raw_folder(path):
    if is_subfolder_of(path, "raw"):
        return path
    else:
        return path.replace("processed", "raw")


def get_processed_folder(path):
    if is_subfolder_of(path, "processed"):
        return path
    else:
        return path.replace("raw", "processed")
