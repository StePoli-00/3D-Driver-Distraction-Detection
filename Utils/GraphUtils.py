import torch
import torch.sparse
import scipy.sparse as sp
import numpy as np
import networkx as nx
import json
import io
import matplotlib.pyplot as plt
from torch_geometric.data import Data

"""
    mp_connections: dictionary representing connections between certain joints of the human body, as shown in the 
    MediaPipe Pose documentation. In this representation, the keys of the dictionary are the nodes, and the values 
    are lists of their connected nodes. 

    https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
"""


# Used to visualize the 3D graph after converting the float array to graph
def get_graph_snapshot(graph_data):
    pos = {}
    rotated_pos = {}
    for i, data in enumerate(graph_data.nodes(data=True)):
        node = data[1]
        pos[i] = (node['x'], node['y'], node['z'])
        rotated_pos[i] = (node['z'], -node['x'], node['y'])

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Flip the y and z axes to correct the orientation
    ax.invert_yaxis()
    ax.invert_zaxis()

    # Draw nodes
    items = list(pos.items())
    for i, (node, (x, y, z)) in enumerate(items[:-1]):
        ax.scatter(x, y, z, label=node)

    for edge in graph_data.edges():
        i, j = edge
        x = [pos[i][0], pos[j][0]]
        y = [pos[i][1], pos[j][1]]
        z = [pos[i][2], pos[j][2]]
        ax.plot(x, y, z)

    # Save the image to a binary stream
    img_stream = io.BytesIO()
    plt.savefig(img_stream, format='png')
    plt.close()

    # Move the stream cursor to the beginning
    img_stream.seek(0)

    # Convert the binary stream to bytes
    img_bytes = img_stream.read()

    return img_bytes


# Used to check if a bounding box is detected
def check_object(detection_data):
    if np.all(detection_data):
        return True
    else:
        return False


"""def get_angle(vector1, vector2):
    # Calculate the angle between two vectors using the dot product formula
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Ensure the denominator is not zero (avoid division by zero)
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0  # Return 0 if either vector has zero magnitude

    # Calculate the angle in radians and convert to degrees
    radians = np.arccos(dot_product / (magnitude1 * magnitude2))
    degrees = np.degrees(radians)

    return degrees.round(4)"""


def process_additional_landmarks(graph, lm_list):
    left_shoulder = None
    right_shoulder = None

    for lm in lm_list:
        if lm[0] == 11:
            left_shoulder = np.array(lm[1:])[0]
            # print(f" left_shoulder {lm[1:]}")
        elif lm[0] == 12:
            right_shoulder = np.array(lm[1:])[0]
            # print(f" right_shoulder {lm[1:]}")

    # Additional Landmark 1: elaboration + addition to graph
    neck_lm = (left_shoulder + right_shoulder) / 2
    next_idx = len(graph.nodes)  # 33, needed for the angle

    x, y, z = neck_lm[0].round(4), neck_lm[1].round(4), neck_lm[2].round(4)
    # hip_distance = get_distance([x, y, z], hip_lm)

    # graph.add_node(next_idx, x=x, y=y, z=z, dist=hip_distance)
    graph.add_node(next_idx, x=x, y=y, z=z)
    return


def process_adj_matrix(node_features, mat):
    # Calculate degree matrix D
    deg = np.diag(np.sum(mat, axis=1))

    # Add a small positive value to the diagonal to ensure invertibility
    epsilon = 1e-6
    deg = deg + epsilon * np.identity(deg.shape[0])

    # Symmetrically normalize adjacency matrix
    deg_sqrt_inv = np.linalg.inv(np.sqrt(deg))
    hat_a = deg_sqrt_inv.dot(mat).dot(deg_sqrt_inv)

    # Add self-connections (optional)
    hat_a = hat_a + np.identity(hat_a.shape[0])

    x = np.vstack(node_features)

    return x, hat_a


def get_distance(ref_lm, lm):
    return (np.linalg.norm(np.array(ref_lm) - np.array(lm))).round(4)


# Function to calculate the angle between two vectors
def get_angle(vector1, vector2):
    cos_theta = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return angle.round(4)


def get_mp_angles(nodes, mp_angles):
    angle_matrix = np.zeros((len(nodes), 2))

    for i, node in enumerate(nodes):
        if str(i) in mp_angles.keys():
            for j, pair in enumerate(mp_angles[str(i)]):
                vector1 = np.array(nodes[pair[0]][1:]) - np.array(nodes[i][1:])
                vector2 = np.array(nodes[pair[1]][1:]) - np.array(nodes[i][1:])
                angle = get_angle(vector1, vector2)
                angle_matrix[i][j] = angle

    return angle_matrix


def get_rel_pos(nodes, mp_conn):
    relative_positions = np.zeros((len(nodes), 9))

    # Iterate through the connectivity information and compute relative positions
    for begin_node in mp_conn.keys():
        connections = mp_conn[begin_node]
        key_position = np.array(nodes[int(begin_node)])[1:]
        for idx, end_node in enumerate(connections):
            conn_node = connections[idx]

            connected_joint_position = np.array(nodes[conn_node])[1:]
            rel_pos = connected_joint_position - key_position

            for i in range(len(rel_pos)):
                relative_positions[int(begin_node)][3 * idx + i] = rel_pos[i]

    return relative_positions


def get_mp_conn(nodes, mp_conn):
    n = len(nodes)
    adj_matrix = np.zeros((n, n))

    for begin_node in mp_conn.keys():
        connections = mp_conn[begin_node]

        for end_node in connections:
            node1 = int(begin_node)
            node2 = end_node

            distance = get_distance(np.array(nodes[node1][1:]), np.array(nodes[node2][1:]))
            adj_matrix[node1, node2] = distance

    return adj_matrix


def get_graph_data(data, mp_connections, mp_angles):
    out = []

    for feature_set in data:
        mp_set = feature_set[0:99]
        det = feature_set[-1]

        raw_nodes = []

        num_landmarks = len(mp_set) // 3
        for i in range(num_landmarks):
            x, y, z = mp_set[i * 3:i * 3 + 3]

            raw_nodes.append([det, x, y, z])

        rel_nodes = get_rel_pos(raw_nodes, mp_connections)
        adj_matrix = get_mp_conn(raw_nodes, mp_connections)
        ang_matrix = get_mp_angles(raw_nodes, mp_angles)

        features = np.hstack((raw_nodes, rel_nodes, ang_matrix))

        node_features, opt_adj_matrix = process_adj_matrix(features, adj_matrix)

        out.append([node_features, opt_adj_matrix])

    return out


def get_graph_view(data, mp_connections):
    out = []
    for feature_set in data:
        mp_set = feature_set[0:99]
        det = feature_set[-1]

        g = nx.Graph()

        num_landmarks = len(mp_set) // 3
        for i in range(num_landmarks):
            x, y, z = mp_set[i * 3:i * 3 + 3]
            g.add_node(i, label=det, x=x, y=y, z=z)

        for begin_node in mp_connections.keys():
            connections = mp_connections[begin_node]

            for end_node in connections:
                node1 = int(begin_node)
                node2 = end_node

                vector1 = np.array([g.nodes[node1]['x'], g.nodes[node1]['y'], g.nodes[node1]['z']])
                vector2 = np.array([g.nodes[node2]['x'], g.nodes[node2]['y'], g.nodes[node2]['z']])

                distance = get_distance(vector1, vector2)
                # angle = get_angle(vector1, vector2)
                # g.add_edge(node1, node2, weight_distance=distance, weight_angle=angle)
                g.add_edge(node1, node2, weight_distance=distance)

        out.append(g)

    return out


def data_to_graph(data):
    mp_connections = "./Data/json/mp_conn.json"

    # Read the classes.json file
    with open(mp_connections, "r") as json_file:
        connections = json.load(json_file)

    gview = get_graph_view(data, connections['mp_conn'])
    gdata = get_graph_data(data, connections['mp_conn'], connections['mp_angles'])

    return gview, gdata


def graph_to_tensor(graph_data, class_labels=[]):
    out = []

    for idx, graph in enumerate(graph_data):

        x = torch.tensor(graph[0], dtype=torch.float32)

        # Check if the adjacency matrix is already in COO format, otherwise convert
        if sp.issparse(graph[1]):
            edge_index_coo = torch.sparse_coo_tensor(*graph[1].nonzero().T)
        else:
            # Convert to COO format
            adj_coo = sp.coo_matrix(graph[1])
            edge_index_coo = torch.sparse_coo_tensor(
                torch.LongTensor(np.array([adj_coo.row, adj_coo.col])),
                torch.FloatTensor(adj_coo.data),
            )

        # Convert COO format to edge_index
        edge_index = edge_index_coo.coalesce().indices().long()

        data = Data(x=x, edge_index=edge_index)

        if len(class_labels) != 0:
            label = torch.tensor(class_labels[idx], dtype=torch.long)
            data.label = label.unsqueeze(0)

        out.append(data)

    return out
