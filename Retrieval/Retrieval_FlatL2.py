import os

import mediapipe as mp
import faiss
import EmbeddingUtils as eu
import FaissRetrieval as fs
import QueryUtils as qu
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


query_path= "Data/Query"
yolo_path ="Model/best.pt"
gcn_path = "Model/GCN.pth"
embedding_path="Data/Embeddings"

database_retrieval_path ="Data/Database"




"""ESEGUIRE SE SI VUOLE FARE RETRIEVAL SALVANDO/CARICARNDO GLI INDICI"""
#faiss.write_index(index, "Retrieval/Retrieval.index") #per salvare gli indici in un file
#index=faiss.read_index("Retrieval/Indexes/retrievalL2.index") #per leggere gli indici da un file


d=1536
#eu.create_embedding_v2(database_retrieval_path)
#xb = fs.load_emb(embedding_path,d)
xq, query_names = qu.create_query(query_path,d)
index = (faiss.IndexFlatL2(d))
# print(f"Index is trained? : {index.is_trained}")
#index.add(xb)
print(f"*** Loaded {index.ntotal} index for Retrieval ***")
# aggiungo la matrice di array all'indice
"""ESEGUIRE SE SI VUOLE FARE RETRIEVAL SALVANDO/CARICARNDO GLI INDICI"""
#faiss.write_index(index, "Retrieval/Indexes/Retrieval.index") #per salvare gli indici in un file
index=faiss.read_index("Retrieval/Indexes/Retrieval.index") #per leggere gli indici da un file
print(f"*** Loaded {index.ntotal} index for Retrieval ***")
# ritorna la matrice delle distanze D  L2 in ordine crescente e la matrice degli indici I
print("*** Search queries ***")


list_k =[4]

fs.show_and_save_result(index,xq,query_names,list_k)
fs.plot_results()

