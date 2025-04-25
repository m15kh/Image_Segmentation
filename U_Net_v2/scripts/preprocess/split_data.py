import os
import shutil

def split_data(label_folder, image_folder, json_folder):
    # Ensure the output folders exist
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    if not os.path.exists(json_folder):
        os.makedirs(json_folder)

    # Iterate through files in the label folder
    for file_name in os.listdir(label_folder):
        file_path = os.path.join(label_folder, file_name)
        if os.path.isfile(file_path):
            # Check file extension and move to the respective folder
            if file_name.endswith('.png'):
                shutil.move(file_path, os.path.join(image_folder, file_name))
            elif file_name.endswith('.json'):
                shutil.move(file_path, os.path.join(json_folder, file_name))

# Define paths
label_folder = "/home/rteam2/m15kh/Auto_Encoder/U_Net/data/label"
image_folder = "/home/rteam2/m15kh/Auto_Encoder/U_Net/data/images"
json_folder = "/home/rteam2/m15kh/Auto_Encoder/U_Net/data/json"

# Split the data
split_data(label_folder, image_folder, json_folder)
