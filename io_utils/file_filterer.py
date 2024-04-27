import os
import shutil
import time
from feature_vectors.constants import FILE_PATH_RES, FILE_PATH_DATA, FILE_PATH_FILTERED_DATA, FILE_PATH_FILTERED_RES
from io_utils.file_reader import get_result_file_name


def filter_files_by_type(folder_path):
    r_files = []
    e_files = []
    h_files = []
    res_files = []

    # Iterate over files in the folder
    for file_name in os.listdir(folder_path):
        if file_name[9] == 'R':
            r_files.append(file_name[:9] + file_name[10:])
        elif file_name[9] == 'E':
            e_files.append(file_name[:9] + file_name[10:])
        elif file_name[9] == 'H':
            h_files.append(file_name[:9] + file_name[10:])

    for element in r_files[:]:
        if element not in e_files or element not in h_files:
            r_files.remove(element)

    for element in e_files[:]:
        if element not in r_files or element not in h_files:
            e_files.remove(element)

    for element in h_files[:]:
        if element not in r_files or element not in e_files:
            h_files.remove(element)

    # generate result file name, remove element if not exist, else add to array
    for element in r_files[:]:
        temp_r_file_name = element[:9] + 'R' + element[9:]
        res_file_name = get_result_file_name(FILE_PATH_RES, temp_r_file_name)
        if res_file_name is not None:
            res_files.append(res_file_name)
        else:
            r_files.remove(element)
            e_files.remove(element)
            h_files.remove(element)

    for i in range(len(r_files)):
        r_files[i] = r_files[i][:9] + 'R' + r_files[i][9:]
        e_files[i] = e_files[i][:9] + 'E' + e_files[i][9:]
        h_files[i] = h_files[i][:9] + 'H' + h_files[i][9:]

    return r_files, e_files, h_files, res_files


def copy_files_to_folder(files, source, destination):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination):
        os.makedirs(destination)

    # Copy each file to the destination folder
    for file in files:
        source_file_path = os.path.join(source, file)
        destination_file_path = os.path.join(destination, file)
        shutil.copyfile(source_file_path, destination_file_path)


# Time Usage: 12.89 - 20.89 in seconds
def filter_files_and_copy():
    start = time.time()
    print("Start filtering files...")
    r_files, e_files, h_files, res_files = filter_files_by_type(FILE_PATH_DATA)
    print("r files: " + str(len(r_files)))
    print("e files: " + str(len(e_files)))
    print("h files: " + str(len(h_files)))
    print("result files: " + str(len(h_files)))

    print("Finished filtering files!\n")

    print("Copy r files...")
    copy_files_to_folder(r_files, FILE_PATH_DATA, FILE_PATH_FILTERED_DATA)
    print("All r files copied!\n")

    print("Copy e files...")
    copy_files_to_folder(e_files, FILE_PATH_DATA, FILE_PATH_FILTERED_DATA)
    print("All e files copied!\n")

    print("Copy h files...")
    copy_files_to_folder(h_files, FILE_PATH_DATA, FILE_PATH_FILTERED_DATA)
    print("All h files copied!\n")

    print("Copy result files...")
    copy_files_to_folder(res_files, FILE_PATH_RES, FILE_PATH_FILTERED_RES)
    print("All result files copied!\n")

    end = time.time()
    print("Time Usage: " + str(round((end - start), 2)) + " in seconds\n")
