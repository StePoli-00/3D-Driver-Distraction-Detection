import numpy as np
import os
import torch
from Utils import ImgUtils as iu
from Utils import GraphUtils as graph_ut
from Utils import GeneralUtils as general_ut
from ultralytics import YOLO
import cv2
import mediapipe as mp
from GNN.GNN_NET import GCN_NET


yolo_path_trained = "Yolo/Yolo parameters/trained_yolo.pt"
yolo_path_pretrained="Yolo/Yolo parameters/pretrained_yolo.pt"
gcn_model = GCN_NET().get_GCN()
mp_pose = mp.solutions.pose

def create_query(query_path: str,dimension:int,option, n_queries=-1):
    yolo_model =[]
    if option=="pre_trained":
        yolo_model=YOLO(yolo_path_pretrained)
    else:
        yolo_model = YOLO(yolo_path_trained)
    rejected=[]
    query_embedding=[]
    query_names=[]

    if n_queries == -1:
         images, image_name = iu.load_images("test", query_path,shuffle=True)
    else:
         images, image_name = iu.load_images("test", query_path, n_images=n_queries,shuffle=True)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
         for idx in range(0, len(images)):
             img = images[idx]

             if img is None:
                 print("Error: Unable to load image.")
                 continue

             pp_img = iu.img_pre_process(img)
             yolo_out = yolo_model.predict(img)[0]

             if len(yolo_out) == 0:
                 detection = -1
             else:
                 y = yolo_out
                 bbox = y.boxes
                 detection = int(bbox.data[0, 5])
             # MediaPipe part
             mp_image = cv2.cvtColor(pp_img, cv2.COLOR_BGR2RGB)
             mp_image.flags.writeable = False

             # Make detection, results variable holds what we get
             results = pose.process(mp_image)

             # Recolor back to BGR
             mp_image.flags.writeable = True
             mp_image = cv2.cvtColor(mp_image, cv2.COLOR_RGB2BGR)
             # print(f" working on image: {image_name[idx]}")
             try:
                 out_lm = general_ut.get_landmark_data(results.pose_landmarks.landmark)
                 img_data = general_ut.get_landmark_data(results.pose_landmarks.landmark)
                 img_data.append(detection)

             except Exception as e:
                 print("Exception:", e)
                 rejected.append(image_name[idx])
                 continue

             if img_data:
                 g_view, g_data = graph_ut.data_to_graph([img_data])
                 # Perform inference
                 with torch.no_grad():
                     tensor_data = graph_ut.graph_to_tensor(g_data)
                     gcn_out, embedding = gcn_model(tensor_data[0])
                     query_embedding.append(embedding)
                     query_names.append(image_name[idx])

    print(f"Embedding created {len(query_embedding)}")
    print(f"Image rejected: {len(rejected)}")
    print(rejected)
    return save_query(query_embedding, dimension), query_names





def save_query(query_list,dimension):
    xq = np.array([]).reshape(0, dimension)

    for q in query_list:
        query_np = q.numpy()
        #query_np= query_np/np.linalg.norm(query_np)
        xq = np.vstack((xq, query_np))

    return xq