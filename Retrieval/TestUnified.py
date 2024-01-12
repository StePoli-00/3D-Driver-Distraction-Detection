import copy
import torch
import cv2
import faiss
import numpy as np
import mediapipe as mp
import json
from ultralytics import YOLO
from PIL import Image

from Utils import GeneralUtils as op
from Utils import ImgUtils as img_op
from Utils import GraphUtils as graph_op
from Utils import EmbeddingUtils as eu
from Utils import QueryUtils as qu
from Retrieval import FaissRetrieval as fs
from GNN.GCNModel import GATNet




mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

dataset = "Dataset-Balanced"

img_path = "Data/Retrieval/Query"
#img_path="Data/Retrieval/Good-Query"

yolo_path_trained="Yolo/Yolo parameters/trained_yolo.pt"
yolo_path_pretrained="Yolo/Yolo parameters/pretrained_yolo.pt"
gcn_path = "./Model/GCN.pth"
classes = "./Data/json/classes.json"
gcn_params = "./Data/json/gcn_params.json"
database_retrieval_path="Data/Retrieval/Database-for-Retrieval"
embedding_path="Data/Retrieval/Embeddings"
index_fname="Retrieval/Indexes/Retrieval_small_dataset.index"
# Read the classes.json file

with open(classes, "r") as json_file:
    classes = json.load(json_file)

# Read the JSON file
with open(gcn_params, "r") as json_file:
    gcn_parameters = json.load(json_file)

# Access the values
input_dim = gcn_parameters["input_dim"]
hidden_dim = gcn_parameters["hidden_dim"]
output_dim = gcn_parameters["output_dim"]
num_heads = gcn_parameters["num_heads"]

#print(f" gcn_params: {gcn_parameters}")

gcn_model = GATNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_heads=num_heads)

gcn_model.load_state_dict(torch.load(gcn_path))
gcn_model.eval()


#39 Bottle-> Drink 67 Cell phone->Phone
mapping_label={39:0,67:1}
total_preds = 0
correct_preds = 0
label_text = ""
font_color=-1

name_im=""

color_classes={
    "ZERO":(0,255,0),
    "LOW":(0,255,255),
    "MEDIUM":(0,127,255),
    "HIGH":(0,0,255),
    "VERY_HIGH":(0,0,127)}
#
def make_detection(img,yolo_option):

    detection=-1
    box_cord=[]
    im_array=[]
    if yolo_option=="pre_trained":
        yolo_path=yolo_path_pretrained
        yolo_model = YOLO(yolo_path)
        yolo_out = yolo_model.predict(img)[0]
        global label_text, font_color
        if len(yolo_out) != 0:
            y = yolo_out
            bbox = y.boxes.data

            for b in bbox:
                label = int(b[5])
                if label == 39:
                    box_cord = b[0:4]
                    box_cord = box_cord.tolist()
                    detection = mapping_label[39]
                    label_text = "Bottle"
                    font_color = (0, 0, 255)  # Colore del testo (in formato BGR)
                elif label == 67:
                    box_cord = b[0:4]
                    box_cord = box_cord.tolist()
                    detection = mapping_label[67]
                    label_text = "Phone"
                    font_color = (255, 0, 0)
            im_array=img
    else:
        yolo_path = yolo_path_trained
        yolo_model = YOLO(yolo_path)
        yolo_out = yolo_model.predict(img)[0]
        if len(yolo_out) == 0:
            detection = -1
        else:
            y = yolo_out
            bbox = y.boxes
            detection = int(bbox.data[0, 5])
            im_array = y.plot()

    return detection,box_cord,im_array



def show_prediction(option_detection,box_cord,mp_image,graph_img,pred):
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 2
    font_thickness = 2
    if option_detection == "pre_trained" and len(box_cord) != 0:
        cv2.rectangle(mp_image, (round(box_cord[0]), round(box_cord[1])), (round(box_cord[2]), round(box_cord[3])),
                      font_color, 2)
        cv2.putText(mp_image, label_text, (round(box_cord[0]), round(box_cord[1] - 10)), font, font_scale, font_color,
                    font_thickness)


    comp_img = np.hstack((mp_image, graph_img))
    cv2.putText(comp_img, str(classes[str(pred)]), (20, 50), font, font_scale, color_classes[classes[str(pred)]],
                font_thickness)
    #cv2.imwrite(f"{name_im}_GNNPrediction.jpg", comp_img)
    cv2.imshow("GNN Prediction", comp_img)
    cv2.waitKey(0)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    # return


def from_feed(images, img_names, option_detection):
    xq=[]
    query_names=[]
    y_true=[]
    option_detection=option_detection.lower()
    if option_detection!="trained" and option_detection!="pre_trained" :
        print(f"ERROR:Invalid Detection Option [trained,pre_trained]")
        exit(1)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for idx in range(0, len(images)):
            img = images[idx]
            global name_im
            name=img_names[idx]
            name_im,_=name.split(".jpg")



            if img is None:
                print("Error: Unable to load image.")
                continue

            pp_img = img_op.img_pre_process(img)

            detection,box_cord,im_array=make_detection(pp_img,option_detection)




            # Visualize the YOLO detection (if found)
            if detection!=-1:
                cv2.imwrite(f"{name_im}_detection.jpg", im_array)
                im_rgb = cv2.cvtColor(np.array(im_array), cv2.COLOR_BGR2RGB)
                im = Image.fromarray(im_rgb)
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
                    gcn_out,data = gcn_model(tensor_data[0])

                    xq.append(data)
                    query_names.append(name)
                    pred = torch.argmax(gcn_out).item()
                    print(f"detection {name}")
                    print(f" *** GCN Prediction: {str(classes[str(pred)])} ***""")
                    y_true.append(pred)

                # MediaPipe Visualization
                mp_drawing.draw_landmarks(mp_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                #cv2.imwrite(f"{name_im}_mediapipe.jpg", mp_image)

                # Graph Visualization
                graph_snapshot_bytes = graph_op.get_graph_snapshot(g_view[0])

                # Convert the image bytes to a numpy array
                graph_snapshot_array = np.frombuffer(graph_snapshot_bytes, np.uint8)

                # Decode the image using OpenCV
                graph_snapshot = cv2.imdecode(graph_snapshot_array, cv2.IMREAD_COLOR)

                graph_img = cv2.resize(graph_snapshot, (mp_image.shape[1], mp_image.shape[0]))

                show_prediction(option_detection,box_cord,mp_image,graph_img,pred)


    return xq,query_names,y_true


if __name__=="__main__":

    d = 1536
    list_k = [4]#list(range(2,10))

    option_yolo="trained"
    test_images, sample_names = img_op.load_images("test", img_path, [])
    xq, query_names,y_true = from_feed(test_images, sample_names,option_yolo)

    if len(xq)==0:
        print("Retrieval part cannot be perfomed: 0 queries accepted")
        exit(1)

    print(f"Query Accepted:{query_names}")

    #eu.create_embedding(database_retrieval_path, n_embedding=300)
    #xb = fs.load_emb(embedding_path,d)

    index = faiss.IndexFlatL2(d)
    xq=qu.save_query(xq,d)

    #index.add(xb)
    #faiss.write_index(index, "Retrieval/Indexes/Retrieval.index") #per salvare gli indici in un file
    index=faiss.read_index(index_fname) #per leggere gli indici da un file
    print(f"*** Loaded {index.ntotal} index for Retrieval ***")
    fs.show_and_save_result(index,xq,query_names,list_k,y_true)



