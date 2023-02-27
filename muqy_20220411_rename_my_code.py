import glob
import os
import time


def read_and_prefix_py_file_name(dir_path):
    """Read all the files in the folder and prefix their filenames with the date"""
    for file in glob.glob(dir_path + "*.py"):
        dir_name = os.path.dirname(file)
        file_name = os.path.basename(file)
        year = time.localtime(os.path.getmtime(file)).tm_year
        mon = time.localtime(os.path.getmtime(file)).tm_mon
        day = time.localtime(os.path.getmtime(file)).tm_mday
        new_file_name = (
            str("muqy_")
            + str(year)
            + str(mon).zfill(2)
            + str(day).zfill(2)
            + "_"
            + file_name
        )
        print(dir_name + new_file_name)
        print("\n")
        # rename the file
        os.rename(file, dir_name + "//" + new_file_name)


read_and_prefix_py_file_name("/RAID01/data/muqy/PYTHON_CODE/")


# def separate_py_file_dir_name():
#     """Separate the file directory from the file name in the string"""
#     for file in glob.glob("E://pytest//*.py"):
#         print(file)
#         print(os.path.dirname(file))
#         print(os.path.basename(file))
#         print("\n")


# separate_py_file_dir_name()
