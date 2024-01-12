import os
import torch
import cv2
import random
import json
from Retrieval import PlotResult as pr
import numpy as np
from Utils import ImgUtils as iu
from Utils import GraphUtils as graph_ut
from Utils import GeneralUtils as general_ut
from Utils import AssociationTableUtils as at
from ultralytics import YOLO
from GNN.GNN_NET import GCN_NET
import mediapipe as mp
from Utils import MetricsUtils as mu

from collections import Counter



#query_path="Data/Retrieval/Good-Query"
query_path="Data/Retrieval/Query"
metric_save_path="Data/Retrieval/Plot-Results/files"
database_path="Data/Retrieval/Database"
embedding_path="Data/Retrieval/Embeddings-Small"
result_path="Data/Retrieval/Results"
association_table_path="Data/json/association_table.json"
yolo_path_trained = "Yolo/Yolo parameters/trained_yolo.pt"
yolo_path_pretrained="Yolo/Yolo parameters/pretrained_yolo.pt"
mapping_label={39:0,67:1}
detection_option=""
queries=[] #serve per salvare il nome delle query
mp_pose = mp.solutions.pose
color = (0, 0, 0)  # Colore del testo (in formato BGR)
font = cv2.FONT_HERSHEY_COMPLEX
font_scale = 2
font_thickness = 2
gcn_model = GCN_NET().get_GCN()

pred_dict = {
    0: "ZERO",
    1: "LOW",
    2: "MEDIUM",
    3: "HIGH",
    4: "VERY_HIGH"
}

color_classes={
    "ZERO":(0,255,0),
    "LOW":(0,255,255),
    "MEDIUM":(0,127,255),
    "HIGH":(0,0,255),
    "VERY_HIGH":(0,0,127)}



def load_emb(emb_path:str, dimension: int):
    xb = np.array([]).reshape(0, dimension)

    embeddings=os.listdir(emb_path)
    random.shuffle(embeddings)
    for f in embeddings:
            arr = np.loadtxt(os.path.join(emb_path, f), dtype=np.float32, skiprows=2)
            xb = np.vstack((xb, arr))  # concatena verticalmente gli array

    print(f"{xb.shape[0]} Embedding loaded")
    return xb


def get_value(value: str)->int:
    for k,v in pred_dict.items():
        if v==value:
            return k


def load_association_table():

    if os.path.exists(association_table_path):
        pairs_database=at.load_association_table_from_json()
    else:
        pairs_database=at.load_association_table_from_embeddings()

    return pairs_database

def normalize_weight(D):

    min = np.min(D)
    max = np.max(D)
    new_min = 0
    new_max = 1
    range_data = max - min

    normalized_data = [(x - min) / range_data * (new_max - new_min) + new_min for x in D]
    #normalized_data  = 1 / D ** 2

    return normalized_data


def get_k_images(I, D):

    pairs=load_association_table()
    retrieved_images={}
    pairs_database={}
    for k in pairs.keys():
        pairs_database[int(k)]=pairs[k]

    for i, d in zip(I, D):
        name,label = pairs_database[i]
        name,_=name.split(".txt")
        name+=".jpg"
        retrieved_images[name] = (label, d)

    return retrieved_images


def make_detection(img):

    detection=-1
    if detection_option=="pre_trained":
        yolo_path=yolo_path_pretrained
        yolo_model = YOLO(yolo_path)
        yolo_out = yolo_model.predict(img)[0]
        if len(yolo_out) != 0:
            y = yolo_out
            bbox = y.boxes.data

            for b in bbox:
                label = int(b[5])
                if label == 39:
                    detection = mapping_label[39]
                elif label == 67:
                    detection = mapping_label[67]

    else:
        yolo_path = yolo_path_trained
        yolo_model = YOLO(yolo_path)
        yolo_out = yolo_model.predict(img)[0]
        if len(yolo_out) != 0:
            y = yolo_out
            bbox = y.boxes
            detection = int(bbox.data[0, 5])
    return detection




def make_prediction(query_img):


    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        img = query_img
        pp_img = iu.img_pre_process(img)
        # YOLO section
        detection=make_detection(pp_img)

        # MediaPipe part
        mp_image = cv2.cvtColor(pp_img, cv2.COLOR_BGR2RGB)
        mp_image.flags.writeable = False

        # Make detection, results variable holds what we get
        results = pose.process(mp_image)

        try:
            img_data = general_ut.get_landmark_data(results.pose_landmarks.landmark)
            img_data.append(detection)

            if img_data:
                g_view, g_data = graph_ut.data_to_graph([img_data])
                # Perform inference
                with torch.no_grad():
                    tensor_data = graph_ut.graph_to_tensor(g_data)
                    gcn_out, embedding = gcn_model(tensor_data[0])
                    # Find the class with the highest confidence score
                    pred = torch.argmax(gcn_out).item()
                    return pred
            else:
                print("No valid landmarks found in the image. Skipping...")

        except Exception as e:
            print("Exception:", e)

        except Exception as e:
            print("Exception:", e)






# def update_arrays_for_metrics(retrieved_results:list):
#
#
#     class_counter = Counter()
#     # query_img = cv2.imread(os.path.join(query_path,query))
#     # count = {"ZERO": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0, "VERY_HIGH": 0}
#     for r, weight in retrieved_results:
#         class_counter[r] += weight
#
#
#     most_common = class_counter.most_common(1)
#     pred_label = most_common[0][0]
#
#     print(f"Query {query}")
#     print(f"*** PREDICTED LABEL FROM KNN: {pred_label} ***")
#
#     if pred_label is not None:
#         return pred_label



def find_knn_prediction(retrieved_results:list):

    class_counter = Counter()
    for r, weight in retrieved_results:
        class_counter[r] += weight

    most_common = class_counter.most_common(1)
    pred_label = most_common[0][0]

    return pred_label


def update_arrays_for_metrics(query:str, retrieved_results:list):

    label=[]
    class_counter = Counter()
    query_img = cv2.imread(os.path.join(query_path,query))
    count = {"ZERO": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0, "VERY_HIGH": 0}
    for r, weight in retrieved_results:
        class_counter[r] += weight


    most_common = class_counter.most_common(1)
    pred_label = most_common[0][0]

    print(f"Query {query}")
    print(f"*** PREDICTED LABEL FROM KNN: {pred_label} ***")
    true_label = make_prediction(query_img)



    if true_label is not None and pred_label is not None:
        true_label = int(true_label)
        pred_label = get_value(pred_label)
        # if pred_label == (true_label + 1) and true_label != 4:
        #     pred_label = true_label

        return (true_label,pred_label)

def show_and_save_result(index, xq, query_names:list, list_k: list,y_true):



    results=[]
    #y_true=[]
    y_pred=[]


    for k in list_k:
        print(f"*** Search queries with  {k} neighbours***")
        D, I = index.search(xq, k)  # actual search

        print("Index matrix")
        print(I)
        print("Distance matrix")
        print(D)
        D = normalize_weight(D)
        for i in range (len(query_names)):
            retrieved = get_k_images(I[i][:], D[i][:])
            frame=create_frame2(query_names[i],list(retrieved.keys()),k)
            #result=update_arrays_for_metrics(query_names[i], list(retrieved.values()))

            try:
                    #results.append((frame,result[1]))
                    #y_true.append(result[0])
                    #y_pred.append(result[1])
                    y=find_knn_prediction(list(retrieved.values()))
                    results.append((frame,y))
                    y_pred.append(get_value(y))
            except:
                pass



        # for _,v in predictions.items():
        #     yt,yp=v[0],v[1]
        #     y_true.append(yt)
        #     y_pred.append(yp)

        mu.calculate_metrics(y_true,y_pred,k)

        print("*** Showing results ***")

        if k%2==0:
            i=0
            for im,label in results:

                cv2.namedWindow('Retrieval results', cv2.WINDOW_NORMAL)
                cv2.putText(im, label, (20, 50), font, font_scale, color_classes[label], font_thickness)
                cv2.imshow('Retrieval results',im)
                cv2.waitKey(0)
                # cv2.imwrite(f'Retrieval results{i}.jpg', im)
                # i+=1

                #cv2.imwrite(os.path.join(result_path,output_name+"_"+str(k)+".jpg"),final)
                #cv2.destroyAllWindows()


        results.clear()
        y_pred.clear()
        #y_true.clear()

    mu.save_metrics()
    return








def find_dim(num):
    for i in range(2, num):
        if num % i == 0:
            first = i
            second = num // i
            return first,second
    return None, None



def create_frame2(query, image_list,k):

    retr_dim=(300,300)
    dim=len(image_list)
    additional=False
    output_dim=[]
    output_name,_=query.split(".jpg")
    query = cv2.imread(os.path.join(query_path, query))
    final=[]
    if dim==1:
        img=cv2.imread(os.path.join(database_path,image_list[0]))
        img=cv2.resize(img,(400,200))
        output_dim=(400,200)
        final=img
        # query=cv2.resize(query,(400,200))
        # final=cv2.hconcat([img,query])

    else:

        if(dim%2!=0):
            dim+=1
            additional=True

        rows,cols=find_dim(dim)
        if(rows==None and cols==None):
            rows=1
            cols=dim
        output_dim=(rows * retr_dim[0], cols * retr_dim[1])
        img=[cv2.imread(os.path.join(database_path,im)) for im in image_list]
        img=[cv2.resize(im, retr_dim) for im in img]

        if additional==True:
            img.append(np.ones((retr_dim[1], retr_dim[0], 3), dtype=np.uint8) * 255)
        grid=[]

        for r in range(rows):
            frame=[]
            for c in range(cols):
                if c==0:
                    frame = img[r*cols+c]
                else:
                    frame=cv2.vconcat([frame,img[r*cols+c]])
            grid.append(frame)


        final=[]
        for i in range(len(grid)):

            if i==0:
                final=grid[i]
            else:
                final=cv2.hconcat([final,grid[i]])

    #print(f"before{query.shape}")


      # mostro a schermo i risultati
    query=cv2.resize(query,dsize=output_dim,interpolation=cv2.INTER_AREA)
    #cercare una funzione che ridimensioni un immagine e metta un padding
    #print(f"after{query.shape}")
    final=cv2.hconcat([query,final])
   # cv2.namedWindow('Retrieval results', cv2.WINDOW_NORMAL)
    # cv2.imshow('Retrieval results', final)
    # #cv2.imshow('Retrieval results', final)  # mostro a schermo i risultati
    # cv2.waitKey(0)
    #  #cv2.imwrite(os.path.join(result_path,output_name+"_"+str(k)+".jpg"),final)
    # cv2.destroyAllWindows()
    return final


def create_frame(query, image_list,k):

    retr_dim=(300,300)
    dim=len(image_list)
    additional=False
    output_dim=[]
    output_name,_=query.split(".jpg")
    query = cv2.imread(os.path.join(query_path, query))
    final=[]
    if dim==1:
        img=cv2.imread(os.path.join(database_path,image_list[0]))
        img=cv2.resize(img,(400,200))
        output_dim=(400,200)
        final=img
        # query=cv2.resize(query,(400,200))
        # final=cv2.hconcat([img,query])

    else:

        if(dim%2!=0):
            dim+=1
            additional=True

        rows,cols=find_dim(dim)
        if(rows==None and cols==None):
            rows=1
            cols=dim
        output_dim=(rows * retr_dim[0], cols * retr_dim[1])
        img=[cv2.imread(os.path.join(database_path,im)) for im in image_list]
        img=[cv2.resize(im, retr_dim) for im in img]

        if additional==True:
            img.append(np.ones((retr_dim[1], retr_dim[0], 3), dtype=np.uint8) * 255)
        grid=[]

        for r in range(rows):
            frame=[]
            for c in range(cols):
                if c==0:
                    frame = img[r*cols+c]
                else:
                    frame=cv2.vconcat([frame,img[r*cols+c]])
            grid.append(frame)


        final=[]
        for i in range(len(grid)):

            if i==0:
                final=grid[i]
            else:
                final=cv2.hconcat([final,grid[i]])

    print(f"before{query.shape}")


      # mostro a schermo i risultati
    query=cv2.resize(query,dsize=output_dim,interpolation=cv2.INTER_AREA)
    #cercare una funzione che ridimensioni un immagine e metta un padding
    print(f"after{query.shape}")
    final=cv2.hconcat([query,final])
    cv2.namedWindow('Retrieval results', cv2.WINDOW_NORMAL)
    cv2.imshow('Retrieval results', final)
    #cv2.imshow('Retrieval results', final)  # mostro a schermo i risultati
    cv2.waitKey(0)
     #cv2.imwrite(os.path.join(result_path,output_name+"_"+str(k)+".jpg"),final)
    cv2.destroyAllWindows()
    return



def plot_results():
    pr.plot_results()


