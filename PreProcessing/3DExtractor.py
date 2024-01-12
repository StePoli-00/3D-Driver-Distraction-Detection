"""
    This script is used to extract MediaPipe landmarks and YOLO data from the pre-processed images;
    such data is then written to disk in form of csv files, refer to the "csv_path" param;
"""
import json
import cv2
import numpy as np
import mediapipe as mp
import os
import ultralytics
from ultralytics import YOLO
from Utils import GeneralUtils as op
from Utils import ImgUtils as img_op

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

dataset = "state-farm-distracted-driver-detection"
subj_train = "../Data/" + dataset + "/imgs/subj/train/"
subj_val = "../Data/" + dataset + "/imgs/subj/val/"

csv_train = "./Data/csv/train/"
csv_val = "./Data/csv/val/"

yolo_path = "./Model/best.pt"

img_classes_path = "./Data/json/img_classes.json"

with open(img_classes_path, "r") as json_file:
    img_classes = json.load(json_file)

classes_dict = {}

for key in img_classes:
    classes_dict[key] = img_classes[key]

print(f"img_classes: {classes_dict}")


def get_unique_values(path):
    unique = []
    curr_path = path + "c0/"

    for subj in os.listdir(curr_path):
        if subj == ".DS_Store":
            continue
        else:
            unique.append(subj)
    return unique


train_subjects = get_unique_values(subj_train)
val_subjects = get_unique_values(subj_val)

print(f" train_subjects: {str(train_subjects)}")
print(f" val_subjects: {str(val_subjects)}")

fields = ['sample', 'landmarks', 'label']

ultralytics.checks()
model = YOLO(yolo_path)  # pretrained YOLOv8n model


def check_label(sample_label):
    sample_label = "c" + str(sample_label)

    if sample_label in classes_dict['1']:
        return 1
    elif sample_label in classes_dict['2']:
        return 2
    elif sample_label in classes_dict['3']:
        return 3
    elif sample_label in classes_dict['4']:
        return 4
    else:
        return 0


def from_feed(data, labels, samples, csv_path):
    with mp_pose.Pose(static_image_mode=True,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        for idx in range(0, len(data)):
            img = data[idx]
            img_name = samples[idx]
            sample_lab = labels[idx]

            if img is None:
                print("Error: Unable to load image.")
                continue

            class_label = check_label(sample_lab)

            if class_label in [0, 2]:
                bbox_coords = [0.0] * 4
                detection = -1.0
            else:
                yolo_out = model.predict(img)
                bbox = yolo_out[0].boxes

                if bbox.xyxyn.shape[0] == 0:
                    bbox_coords = [0.0] * 4
                    detection = -1.0
                else:
                    bbox_coords = bbox.xyxyn.tolist()[0]
                    # detection = 0.0
                    detection = bbox.cls[0].item()
                    print(f" detection: {detection}")
                    if detection == 1 and class_label == 1:
                        print(f" detection '1' [phone] for class label 1 - ERROR ")
                        detection = 0.0
                    elif detection == 0 and class_label != 1:
                        print(f" detetection '0' [drink] for class label != 1 - ERROR")
                        detection = 1.0

            results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            try:
                out_lm = op.get_landmark_data(results.pose_landmarks.landmark)
                img_data = np.concatenate((out_lm, bbox_coords, [detection]), axis=0)
                sample_path = csv_path + "train" + str(class_label) + ".csv"
                op.to_file(sample_path, img_name, fields, img_data, class_label)

            except Exception as e:
                print("Exception:", e)
                continue


classes = [1, 3]

_, train_data, train_labels, train_path = img_op.load_images("train", subj_train, classes)
_, val_data, val_labels, val_path = img_op.load_images("train", subj_val, classes)

print(f" len(train_data): {str(len(train_data))}")
print(f" len(val_data): {str(len(val_data))}")

from_feed(train_data, train_labels, train_path, csv_train)
print(f" train_data written to csv")
from_feed(val_data, val_labels, val_path, csv_val)
print(f" val_data written to csv")
