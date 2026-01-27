import gdown
import os
import configparser
from colorama import Fore
from pathlib import Path
from ftplib import FTP

# Models
model_weights = {"PPE_200.pt": "https://drive.google.com/file/d/19WYWQOscHPSbbuyoulR0ULPD3mCPi-a0/view?usp=drive_link",
                 "yolov5s.pt": "https://drive.google.com/file/d/1EpbhGRXU8PA0ReeAU89fOWFqZYinxYCq/view?usp=drive_link",
                 "yolov5n.pt": "https://drive.google.com/file/d/1EpbhGRXU8PA0ReeAU89fOWFqZYinxYCq/view?usp=drive_link",
                 "yolov8s.pt": "https://drive.google.com/file/d/17z52g4E0hQ1VKVCNRxj3mzuay9oXyNZ5/view?usp=drive_link"}


def download_model_from_ftp_server(local_model_path, model_name, remote_model_path):
    directory_path = Path(local_model_path)

    # creating folder structure
    directory_path.mkdir(parents=True, exist_ok=True)

    # complete file path
    local_model_path = os.path.join(local_model_path, model_name)
    remote_model_path = remote_model_path + model_name

    if not os.path.exists(local_model_path):
        # server key
        # config_data = read_config()

        server_address = "ftp.deutics.com"
        username = "abdul.wahab.malik@deutics.com"
        password = "KeaW_GL=$W)r"

        with FTP(server_address) as ftp:
            ftp.login(user=username, passwd=password)

            with open(local_model_path, 'wb') as local_file:
                def callback(data):
                    local_file.write(data)

                ftp.retrbinary(f'RETR {remote_model_path}', callback,
                               blocksize=8192)  # Increased block size to 8192 bytes


def get_indexes(model_classes, expected_classes):
    """*************************************
    Functionality: find the indexes of exp_classes(list of string) form model_classes(list of string)
    Parameters: model_classes(list of yolo model class), exp_classes(list of our expected classes)
    Returns: List of indexes
    ****************************************"""
    if isinstance(model_classes, dict):
        model_classes = list(model_classes.values())

    indexes = []
    for i, label in enumerate(expected_classes):
        if label in model_classes:
            indexes.append(model_classes.index(label))

    return indexes


def download_model_from_drive(model_path, model_name):
    directory_path = Path(model_path)
    directory_path.mkdir(parents=True, exist_ok=True)

    model_directory = os.path.join(model_path, model_name)
    if not os.path.exists(model_directory):
        print(Fore.RED + 'Model Not Found\nDownloading the model' + Fore.RESET)

        link = model_weights[model_name]
        file_id = link.split('/')[-2]
        prefix = "https://drive.google.com/uc?/exports=download&id="
        link = prefix + file_id

        # download_weights_from_drive(url=link, output_path=model_directory)
        gdown.download(link, quiet=False, output=model_directory, use_cookies=False)


def read_config():
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read('config.cfg')

    # Access values from the configuration file
    host_name = config.get('FTP_SERVER', 'HOSTNAME')
    user_name = config.get('FTP_SERVER', 'USERNAME')
    password = config.get('FTP_SERVER', 'PASSWORD')

    # Return a dictionary with the retrieved values
    config_values = {
        'USERNAME': user_name,
        'PASSWORD': password,
        'HOSTNAME': host_name
    }

    return config_values
