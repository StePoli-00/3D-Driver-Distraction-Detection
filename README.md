# Driver Distraction Detection

This project, developed in collaboration with [Francesco Zampirollo](https://github.com/zampifre) and [Vincenzo Macellaro](https://github.com/vincenzomacellaro), was undertaken as part of the Computer Vision & Cognitive System exam for our Master's degree. The primary objective of the system is to classify the driver's state into five distinct risk categories, ranging from zero to very high.

## Features: 
1. **Mediapipe**: Used for keypoint detection to analyze the driver's state.
2. **YOLO (You Only Look Once)**: Employed for detecting potential distraction objects such as phones.
3. **Graph Neural Network (GNN)**: Developed by us, that combine Mediapipe's output with the YOLO bounding box's coordinates, for the classification of the driver's state.
4. **Faiss**: Library used to Retrieval part. Retrieval system returns the embedding images most similar a specified query. The process involves comparing the ground-truth classification (GNN) with the classification obtained through K-nearest neighbors (KNN) during the retrieval process with the K-embeddings.

### Overview: 
<img width="1379" alt="Screenshot 2024-01-12 alle 09 41 45" src="https://github.com/zampifre/DDD/assets/60720249/96f27e10-aaca-4fec-a6a5-8874602d0015">
