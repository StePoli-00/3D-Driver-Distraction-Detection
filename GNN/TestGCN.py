import os
from Utils import GeneralUtils as op
from Utils import ImgUtils as img_op
from Utils import GraphUtils as graph_op
import torch
import numpy as np
import mediapipe as mp
import cv2
import json
from ultralytics import YOLO
from PIL import Image
from GCNModel import GATNet

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

dataset = "Dataset-Balanced"
#img_path = "./Data/" + dataset + "/test"
img_path="Data/Query"
yolo_path = "./Model/best.pt"
gcn_path = "./Model/GCN.pth"

classes = "./Data/json/classes.json"

gcn_params = "./Data/json/gcn_params.json"

# Read the classes.json file
with open(classes, "r") as json_file:
    classes = json.load(json_file)

print(f" classes: {classes}")

# Read the JSON file
with open(gcn_params, "r") as json_file:
    gcn_parameters = json.load(json_file)

# Access the values
input_dim = gcn_parameters["input_dim"]
hidden_dim = gcn_parameters["hidden_dim"]
output_dim = gcn_parameters["output_dim"]
num_heads = gcn_parameters["num_heads"]

print(f" gcn_params: {gcn_parameters}")

gcn_model = GATNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_heads=num_heads)

gcn_model.load_state_dict(torch.load(gcn_path))
gcn_model.eval()

yolo_model = YOLO(yolo_path)

total_preds = 0
correct_preds = 0


def from_feed(images):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for idx in range(0, len(images)):
            img = images[idx]

            if img is None:
                print("Error: Unable to load image.")
                continue

            pp_img = img_op.img_pre_process(img)

            yolo_out = yolo_model.predict(pp_img)[0]

            if len(yolo_out) == 0:
                detection = -1
            else:
                y = yolo_out
                bbox = y.boxes
                detection = int(bbox.data[0, 5])

                # Visualize the YOLO detection (if found)
                im_array = y.plot()
                im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(im_rgb)

                # Convert the PIL image back to a NumPy array in BGR format
                im_bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
                pp_img = im_bgr  # im_bgr is already pre-processed

            # MediaPipe part
            mp_image = cv2.cvtColor(pp_img, cv2.COLOR_BGR2RGB)
            mp_image.flags.writeable = False

            # Make detection, results variable holds what we get
            results = pose.process(mp_image)

            # Recolor back to BGR
            mp_image.flags.writeable = True
            mp_image = cv2.cvtColor(mp_image, cv2.COLOR_RGB2BGR)

            try:
                img_data = op.get_landmark_data(results.pose_landmarks.landmark)
                img_data.append(detection)

            except Exception as e:
                print("Exception:", e)
                continue

            if img_data:
                g_view, g_data = graph_op.data_to_graph([img_data])

                # Perform inference
                with torch.no_grad():
                    tensor_data= graph_op.graph_to_tensor(g_data)
                    gcn_out,_ = gcn_model(tensor_data[0])
                    pred = torch.argmax(gcn_out).item()
                    print(" *** GCN Prediction: " + str(classes[str(pred)]))

                # MediaPipe Visualization
                mp_drawing.draw_landmarks(mp_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Graph Visualization
                graph_snapshot_bytes = graph_op.get_graph_snapshot(g_view[0])

                # Convert the image bytes to a numpy array
                graph_snapshot_array = np.frombuffer(graph_snapshot_bytes, np.uint8)

                # Decode the image using OpenCV
                graph_snapshot = cv2.imdecode(graph_snapshot_array, cv2.IMREAD_COLOR)

                graph_img = cv2.resize(graph_snapshot, (mp_image.shape[1], mp_image.shape[0]))

                comp_img = np.hstack((mp_image, graph_img))

                # Display the MediaPipe Feed
                cv2.imshow("3D Detection", comp_img)

                cv2.waitKey(0)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()


test_images, test_samples = img_op.load_images("test", img_path, [])
from_feed(test_images)
