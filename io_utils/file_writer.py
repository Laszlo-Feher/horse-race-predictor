import os
import datetime


def create_text_file(formatted_time, folder, file_name):
    current_time = datetime.datetime.now()

    reports_folder = "reports"
    if not os.path.exists(reports_folder):
        os.makedirs(reports_folder)

    time_folder = os.path.join(reports_folder, formatted_time)
    if not os.path.exists(time_folder):
        os.makedirs(time_folder)

    method_folder = os.path.join(time_folder, folder)
    if not os.path.exists(method_folder):
        os.makedirs(method_folder)

    file_path = os.path.join(method_folder, f"{file_name}.txt")

    with open(file_path, "a") as file:
        file.write(current_time.strftime("%Y-%m-%d - %H:%M:%S") + "\n")

    return file_path


def write_to_file(array_content, file_name):
    try:
        with open(file_name, "a") as file:
            # Write each element of the array to a new line in the file
            for element in array_content:
                file.write(element + "\n")
        print("Content successfully written to the file:", file_name)
    except Exception as e:
        print("Error occurred while writing to file:", e)


def write_dataframe_to_file(dataframe, directory_path):
    """
    Write a Pandas DataFrame to a file with a specific name within a directory.

    Parameters:
    dataframe (DataFrame): The DataFrame to be written to the file.
    directory_path (str): The path to the directory where the DataFrame file will be written.

    Returns:
    None
    """
    file_name = "exported_feature_vectors.csv"
    file_path = os.path.join(directory_path, file_name)

    try:
        dataframe.to_csv(file_path, index=False)
        print("DataFrame has been successfully written to", file_path)
    except Exception as e:
        print("Error occurred while writing the DataFrame to", file_path)
        print("Error:", e)
