import gdown
import os
from colorama import Fore
from pathlib import Path

# Models
model_weights = {"yolov8s.pt": "https://drive.google.com/file/d/17z52g4E0hQ1VKVCNRxj3mzuay9oXyNZ5/view?usp=drive_link"}


def download_weights_from_drive(url, output_path):
    gdown.download(url, quiet=False, output=output_path, use_cookies=False)


def download_model_from_gdrive(model_path, model_name):
    directory_path = Path(model_path)
    directory_path.mkdir(parents=True, exist_ok=True)

    model_directory = os.path.join(model_path, model_name)
    if not os.path.exists(model_directory):
        print(Fore.RED + 'Model Not Found\nDownloading the model' + Fore.RESET)
        # converting link
        link = model_weights[model_name]
        file_id = link.split('/')[-2]
        prefix = "https://drive.google.com/uc?/exports=download&id="
        link = prefix+file_id
        #
        download_weights_from_drive(url=link, output_path=model_directory)

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
