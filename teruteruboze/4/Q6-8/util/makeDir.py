import os

def do(log_path, log_file_path, csv_files_path, checkpoint_path, figs_path, default_path='./exports/'):
    # log file delete
    try:
        os.makedirs(default_path + log_path)
    except FileExistsError:
        pass

    try:
        os.remove(default_path + log_path + log_file_path)
    except OSError as e:
        pass

    # CSV
    try:
        os.makedirs(default_path + csv_files_path)
    except FileExistsError:
        pass

    # net.pth CHECKPOINT
    try:
        os.makedirs(default_path + checkpoint_path)
    except FileExistsError:
        pass

    # figures
    try:
        os.makedirs(default_path + figs_path)
    except FileExistsError:
        pass