import os
import datetime


def create_text_file():
    current_time = datetime.datetime.now()

    formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")

    if not os.path.exists("reports"):
        os.makedirs("reports")

    file_path = os.path.join("reports", f"{formatted_time}.txt")

    with open(file_path, "w") as file:
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
