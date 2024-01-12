import csv
import os
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose


# Used to extract MediaPipe Landmarks data
def get_landmark_data(out_lm):
    landmarks = []
    for lm in mp_pose.PoseLandmark:
        landmarks.append(out_lm[lm.value].x)
        landmarks.append(out_lm[lm.value].y)
        landmarks.append(out_lm[lm.value].z)

    return landmarks


# Used to Check the existence of the file within the "path" path
def check_file(path, fields):
    if os.path.isfile(path) and os.access(path, os.R_OK):
        pass
    else:
        print(" *** creating new file at path: " + str(path))
        with open(path, 'w') as new_file:
            writer = csv.DictWriter(new_file, delimiter=',', fieldnames=fields)
            writer.writeheader()
            pass
    return


# Used to save float data to file
def to_file(file_path, sample_name, fields, sample_data, label):
    check_file(file_path, fields)
    with open(file_path, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writerow({'sample': sample_name, 'landmarks': sample_data, 'label': label})
        csv_file.close()


# Used to read csv files from disk
def load_csv_data(path):
    train_data = []
    train_labels = []
    for file in os.listdir(path):
        if file == ".DS_Store":
            continue
        else:
            with open(os.path.join(path, file), 'r') as csv_file:
                csvreader = csv.reader(csv_file)

                next(csvreader)

                for row in csvreader:
                    curr_label = int(row[2])
                    data_str = row[1].replace('[', '').replace(']', '')
                    data_values = data_str.split()

                    # Convert the values to floats
                    data_float = np.array(data_values, dtype=float)

                    train_data.append(data_float)
                    train_labels.append(curr_label)

    return np.array(train_data), np.array(train_labels)
