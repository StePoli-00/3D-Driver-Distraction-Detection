from Utils import ImgUtils as img_op
import pandas as pd
import numpy as np
import os
import json
import random

dataset = "state-farm-distracted-driver-detection"
driver_list_path = "../Data/" + dataset + "/driver_imgs_list.csv"
img_path = "../Data/" + dataset + "/imgs/train/"  # original dataset path
subj_train = "../Data/" + dataset + "/imgs/subj/train/"  # path where we divided our samples by subject [train]
subj_val = "../Data/" + dataset + "/imgs/subj/val/"  # path where we divided our samples by subject [test]

val_drivers_path = "./Data/json/val_drivers.json"
img_classes_path = "./Data/json/img_classes.json"

with open(img_classes_path, "r") as json_file:
    img_classes = json.load(json_file)

classes_dict = {}

for key in img_classes:
    classes_dict[key] = img_classes[key]

print(f"img_classes: {classes_dict}")

# Loading drivers data from CSV
drivers = pd.read_csv(driver_list_path)
unique_names = drivers['subject'].unique()

if os.path.exists(val_drivers_path):
    with open(val_drivers_path, "r") as json_file:
        val_names = json.load(json_file)

        val_names = val_names['val_names']
else:
    val_num = 3  # 3 drivers will be selected as part of the Validation set
    val_names = random.sample(list(unique_names), val_num)

    with open(val_drivers_path, "w") as json_file:
        json.dump({"val_names": val_names}, json_file)

train_names = [element for element in list(unique_names) if element not in val_names]

print(f"train names: {train_names}")
print(f"val names: {val_names}")

classes = np.array(range(0, 10))
samples, _, sample_names, num_dict = img_op.load_images("img", img_path, classes)


print("\n > Samples dictionary ")
for key in img_classes:
    sel_classes = img_classes[key]
    acc = 0

    if sel_classes == 'c0':
        print(f" key '{key}' - samples: {len(num_dict['c0'])}")
    else:
        for c in sel_classes:
            if c in num_dict:  # Check if the key exists in num_dict
                curr_len = len(num_dict[c])
                acc += curr_len
            else:
                print(f"Key '{c}' not found in num_dict")

        print(f" key '{key}' - samples: {acc}")


user_choice = str(input("\nDo you want to perform Data Augmentation on any of the classes? [Y/N]: "))

while user_choice.lower() not in ["y", "n"]:
    print("Invalid input. Please select Y or N ")
    user_choice = str(input(" Your input: "))

if user_choice == "Y":
    sel_classes = []
    while True:
        user_input = input("Enter the class id [0, 1, 2, 3, 4] (or 'done' to finish): ")

        if user_input.lower() == 'done':
            break
        try:
            # Convert the user input to an integer and add it to the list
            value = int(user_input)
            if value not in range(0, 5):
                print("Invalid input. Integer must be in range [0, 4]")
            else:
                sel_classes.append(value)
        except ValueError:
            print("Invalid input. Please enter an integer or 'done'.")
    print(" ")
else:
    sel_classes = []

img_data = {'sample': samples,
            'file_name': sample_names}

img_df = pd.DataFrame(img_data)

print(sel_classes)

img_op.img_processor(train_names, drivers, img_df, classes_dict, sel_classes, subj_train)
img_op.img_processor(val_names, drivers, img_df, classes_dict, sel_classes, subj_val)
