import numpy as np
import os
import json
import torch
from Utils import AssociationTableUtils as at
from Utils import ImgUtils as iu
from Utils import GraphUtils as graph_ut
from Utils import GeneralUtils as general_ut
from ultralytics import YOLO
import cv2
from PIL import Image
from GNN.GNN_NET import  GCN_NET
import mediapipe as mp


yolo_path ="Yolo/Yolo parameters/trained_yolo.pt"
yolo_model = YOLO(yolo_path)
mp_pose = mp.solutions.pose
mapping_label={39:0,67:1}

#DICT for prediction
pred_dict = { 0: "ZERO",1: "LOW",2: "MEDIUM",3: "HIGH",4: "VERY_HIGH"}
gcn_model = GCN_NET().get_GCN()



def create_embedding(database_retrieval_path: str,embedding_path, n_embedding=-1):
    ID=0
    images=[]
    image_name=[]

    already_created=len(os.listdir(embedding_path))
    if n_embedding == -1:
        images, image_name = iu.load_images("test", database_retrieval_path)
        n_embedding=len(images)
    else:
        images, image_name = iu.load_images("test", database_retrieval_path, n_images=n_embedding)

    if already_created!=0:

        print(f"Warning {embedding_path} folder not empty: {already_created} found")
        print(f"Warning: there are already {already_created} embedding in {embedding_path} folder")
        print(f"Do you want to add more? Y/N")
        ans=input()
        if ans.upper()=="N":
            return

        files=os.listdir(embedding_path)
        print(files)
        last=files[len(files)-1]
        with open(os.path.join(embedding_path,last),"r") as  f:
            ID=int(f.readline())+1

    emb_list=[]
    rejected_emb=[]
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for idx in range(0, len(images)):
            img = images[idx]

            if img is None:
                print("Error: Unable to load image.")
                continue


            pp_img = iu.img_pre_process(img)
            print(f"Image processed: {image_name[idx]}")
            yolo_out = yolo_model.predict(pp_img)[0]
            box_cord = []
            font_color = (255, 0, 0)
            detection = -1
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
            # print(f" working on image: {image_name[idx]}")
            try:
                out_lm = general_ut.get_landmark_data(results.pose_landmarks.landmark)
                img_data = general_ut.get_landmark_data(results.pose_landmarks.landmark)
                img_data.append(detection)

            except Exception as e:
                print("Exception:", e)
                rejected_emb.append(image_name[idx])
                continue

            if img_data:
                g_view, g_data = graph_ut.data_to_graph([img_data])
                # Perform inference
                with torch.no_grad():
                    tensor_data = graph_ut.graph_to_tensor(g_data)
                    gcn_out,embedding= gcn_model(tensor_data[0])
                    pred = torch.argmax(gcn_out).item()
                    label = pred_dict[pred]

                    basename_file = image_name[idx].split(".jpg")
                    basename_file = basename_file[0] + ".txt"
                    file = os.path.join(embedding_path, basename_file)
                    if not os.path.exists(file):
                        emb_list.append(embedding)
                        at.association_table[ID]=[basename_file, label]
                        save_embedding(embedding, os.path.join(embedding_path, basename_file), ID, label)
                        ID += 1

    print(f"Embedding created {len(emb_list)}")
    print(f"Image rejected: {len(rejected_emb)}")
    print(rejected_emb)
    at.write_association_table()
    return


def save_embedding(embedding, emd_name, index,label):

    vector = embedding.numpy()

    with open(emd_name, 'w') as file:
        file.write((str(index) + "\n"))
        file.write(label+"\n")
        np.savetxt(emd_name, vector)



