# import faiss
# import os
# import EmbeddingUtils as eu
# from Retrieval import FaissRetrieval as fs
# database_retrieval_path="Data/Retrieval/Database"
# #embedding_path="Data/Retrieval/Embeddings-Small"
# embedding_path="Data/Retrieval/Embeddings"
# d = 1536
#
# eu.create_embedding(database_retrieval_path,embedding_path,n_embedding=1000)
# xb = fs.load_emb(embedding_path,d)
# index = (faiss.IndexFlatL2(d))
# index.add(xb)
#
# fname="Retrieval/Indexes/Retrieval_"+os.path.basename(database_retrieval_path)+".index"
# faiss.write_index(index,fname)