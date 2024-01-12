import os
from sklearn.metrics import precision_score, recall_score, accuracy_score
metric_save_path="Data/Retrieval/Plot-Results/files"


dict_precision_micro = {}
dict_recall_micro = {}
dict_accuracy = {}
dict_precision_macro = {}
dict_recall_macro = {}
dict_precision_weighted = {}
dict_recall_weighted = {}


def calculate_metrics(y_true, y_pred, k):
    accuracy = accuracy_score(y_true, y_pred)
    dict_accuracy[str(k)] = accuracy

    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=1)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=1)
    dict_precision_micro[str(k)] = precision_micro
    dict_recall_micro[str(k)] = recall_micro

    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=1)
    dict_precision_macro[str(k)] = precision_macro
    dict_recall_macro[str(k)] = recall_macro

    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    dict_precision_weighted[str(k)] = precision_weighted
    dict_recall_weighted[str(k)] = recall_weighted

    return

def save_metrics():
    with open(os.path.join(metric_save_path, "P_MICRO.txt"), "w") as p_micro:
        for index, value in dict_precision_micro.items():
            riga = f"{index}: {value}\n"
            p_micro.write(riga)
        p_micro.close()

    with open(os.path.join(metric_save_path, "R_MICRO.txt"), "w") as r_micro:
        for index, value in dict_recall_micro.items():
            riga = f"{index}: {value}\n"
            r_micro.write(riga)
        r_micro.close()

    with open(os.path.join(metric_save_path, "P_MACRO.txt"), "w") as p_macro:
        for index, value in dict_precision_macro.items():
            riga = f"{index}: {value}\n"
            p_macro.write(riga)
        p_macro.close()

    with open(os.path.join(metric_save_path, "R_MACRO.txt"), "w") as r_macro:
        for index, value in dict_recall_macro.items():
            riga = f"{index}: {value}\n"
            r_macro.write(riga)
        r_macro.close()

    with open(os.path.join(metric_save_path, "P_WEIGHTED.txt"), "w") as p_weighted:
        for index, value in dict_precision_weighted.items():
            riga = f"{index}: {value}\n"
            p_weighted.write(riga)
        p_weighted.close()

    with open(os.path.join(metric_save_path, "R_WEIGHTED.txt"), "w") as r_weighted:
        for index, value in dict_recall_weighted.items():
            riga = f"{index}: {value}\n"
            r_weighted.write(riga)
        r_weighted.close()

    with open(os.path.join(metric_save_path, "ACCURACY.txt"), "w") as acc:
        for index, value in dict_accuracy.items():
            riga = f"{index}: {value}\n"
            acc.write(riga)
        acc.close()
