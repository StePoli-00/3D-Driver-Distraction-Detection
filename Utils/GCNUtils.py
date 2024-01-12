from torch_geometric.data import DataLoader
from Utils import GeneralUtils as op
from Utils import GraphUtils as graph_op
import torch
import numpy as np


def get_class_weights(labels, mod):
    lab_set = set(labels)
    label_dict = {}

    for label in lab_set:
        label_dict[label] = 0

    label_count = 0
    for u_lab in lab_set:
        for t_lab in labels:
            if t_lab == u_lab:
                label_count += 1

        label_dict[u_lab] = label_count
        label_count = 0

    class_counts = []
    for key in label_dict.keys():
        class_counts.append(label_dict[key])

    class_weights = [sum(class_counts) / (len(class_counts) * count) for count in class_counts]

    if mod == 1:
        class_weights = [class_weights[0], 1.0, class_weights[2], class_weights[3], class_weights[4]]
        # class_weights = [class_weights[0], class_weights[1], class_weights[2], class_weights[3], class_weights[4]]

    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    return class_counts, class_weights


def get_label_dict(labels):
    unique_labels = np.unique(labels)
    samples_map = {}

    for label in unique_labels:
        label_samples = np.where(labels == label)[0]
        samples_map[label] = label_samples

    return samples_map


def get_balanced_dataset(org_dict, mod_dict, samples, labels):
    bal_samples, bal_labels = [], []

    for key in mod_dict.keys():
        selected_indices = np.random.choice(org_dict[key], mod_dict[key], replace=False)
        bal_samples.append(samples[selected_indices])
        bal_labels.append(labels[selected_indices])

    out_samples = np.concatenate(bal_samples)
    out_labels = np.concatenate(bal_labels)

    return out_samples, out_labels


def balance_data(samples, labels):
    org_dict = get_label_dict(labels)
    print(f" - Unique labels: {org_dict.keys()}")
    print(" - Current samples: ")
    print(" ")
    for key in org_dict.keys():
        print(f" - Class ID: {key} - Samples: {len(org_dict[key])}")

    print(" ")
    mod_dict = {}
    while True:
        user_input = input("- Enter the class id to tweak (or 'done' to finish): ")
        if user_input.lower() == 'done':
            break
        elif user_input.lower() == 'auto':
            break
        try:
            class_idx = int(user_input)
            if class_idx not in range(0, 5):
                print("- Invalid input. Integer must be in range [0, 4]")
            else:
                while True:
                    value = int(input(f" - Enter new number of samples for class {class_idx}: "))
                    try:
                        # if value > labels_count[class_idx]:
                        if value > len(org_dict[class_idx]):
                            print(" - Invalid input. New samples count cannot be bigger than the original.")
                        else:
                            mod_dict[class_idx] = value
                            break
                    except ValueError:
                        print(" - Invalid input. Please enter an integer.")

        except ValueError:
            print(" - Invalid input. Please enter an integer or 'done'.")

    if mod_dict.keys():
        print(f"\n - New counts: {mod_dict}")
        bal_samples, bal_labels = get_balanced_dataset(org_dict, mod_dict, samples, labels)

        return bal_samples, bal_labels
    else:
        print(" - You specified no new values for the - balance - task, returning the originals...")
        return samples, labels


def load_gnn_data(train_path, val_path, batch_size, mode, weights_sel):
    output = []

    print(f" - Loading GNN data...")
    train_samples, train_labels = op.load_csv_data(train_path)
    val_samples, val_labels = op.load_csv_data(val_path)

    if mode == 1:
        print(f" - Balancing dataset...")
        train_samples, train_labels = balance_data(train_samples, train_labels)
        val_samples, val_labels = balance_data(val_samples, val_labels)

    print(f" - Converting data to graphs...")
    _, train_graphs = graph_op.data_to_graph(train_samples)
    _, val_graphs = graph_op.data_to_graph(val_samples)

    print(f" - Converting graphs to tensors...")
    train_tensors = graph_op.graph_to_tensor(train_graphs, train_labels)
    val_tensors = graph_op.graph_to_tensor(val_graphs, val_labels)

    train_counts, train_weights = get_class_weights(train_labels, weights_sel)
    val_counts, val_weights = get_class_weights(val_labels, weights_sel)

    print(f" train_tensors num: {len(train_labels)}")
    print(f" val_tensors num: {len(val_samples)}")

    print(f" train_weights: {train_weights} ")
    print(f" val_weights: {val_weights} ")

    train_loader = DataLoader(train_tensors, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensors, batch_size=batch_size, shuffle=True)

    output.append(train_loader)
    output.append(val_loader)
    output.append(train_counts)
    output.append(train_weights)
    output.append(val_counts)
    output.append(val_weights)

    return output
