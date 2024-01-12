import json
from GNN.GCNModel import GATNet
import torch
classes = "Data/json/classes.json"
gcn_params = "Data/json/gcn_params.json"
gcn_path = "Model/GCN.pth"

class GCN_NET:

    def __init__(self):
        # Read the classes.json file
        with open(classes, "r") as json_file:
            self.classes = json.load(json_file)
        #print(f" classes: {classes}")
        # Read the JSON file
        with open(gcn_params, "r") as json_file:
            self.gcn_parameters = json.load(json_file)

        input_dim = self.gcn_parameters["input_dim"]
        hidden_dim = self.gcn_parameters["hidden_dim"]
        output_dim = self.gcn_parameters["output_dim"]
        num_heads = self.gcn_parameters["num_heads"]
        #print(f" gcn_params: {self.gcn_parameters}")

        self.gcn_model = GATNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_heads=num_heads)

        self.gcn_model.load_state_dict(torch.load(gcn_path))
        self.gcn_model.eval()

    def get_GCN(self):
        return self.gcn_model
    def get_classes(self):
        return self.classes


def write_hyperparameters(hyper,file_name):
    try:
        with open(file_name, 'w') as file:
            json.dump(hyper, file)
        print(f"Dictionary writed on {file_name}")
    except Exception as e:
        print(f"Error in writing{file_name}: {str(e)}")



