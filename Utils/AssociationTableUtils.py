import json
import os
embedding_path="Data/Retrieval/Embeddings-Small"
directory_path="Data/Retrieval"
#file_name="Data/json/association_table.json"
file_name="Data/json/association_table_small.json"
association_table={}

def load_association_table_from_embeddings():
    for f in sorted(os.listdir(embedding_path)):
        if f != '.DS_Store':
            with open(os.path.join(embedding_path, f), "r") as fin:
                index = fin.read()
                index = index.split("\n")
                association_table[int(index[0])] = (f, index[1])
    return association_table


def write_association_table():
    with open(file_name, 'w') as file:
        json.dump(association_table, file, indent=2)


def load_association_table_from_json():
    with open(file_name, "r") as file:
        return json.load(file)



