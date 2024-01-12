import matplotlib.pyplot as plt
import os
plot_path="Data/Retrieval/Plot-Results/images"
data_for_plot="Data/Retrieval/Plot-Results/files"
metric_files={
        "P_MACRO.txt":"Precision Macro",
        "P_WEIGHTED.txt":"Precision Weighted",
        "R_MiCRO.txt": "Recall Micro",
        "R_MACRO.txt": "Recall Macro",
        "R_WEIGHTED.txt":"Recall Weighted",
        "ACCURACY.txt": "Accuracy"
        }

'''def plot_results():
    data={}
    for f_name,type_name in metric_files.items():
        with open(os.path.join(data_for_plot,f_name), 'r') as file:
            for linea in file:
                elemento = linea.strip().split(',')

                index = int(elemento[0].split(':')[1].strip())
                value = elemento[1].split(':')[1].strip()

                truncate_value = round(float(value), 3)
                data[index]=truncate_value


        plt.figure(figsize=(12, 10))
        plt.plot(data.keys(),data.values(), marker='o', linestyle='-')

        plt.title(str(type_name) + " al variare di K ")
        plt.xlabel('K')
        plt.ylabel('Valore')
        plt.grid(True)
        basename=type_name.split(".txt")
        basename=basename[0]+".png"
        path=os.path.join(plot_path,basename)
        plt.savefig(path)
        plt.show()
        data.clear()'''

def plot_results():


    indici = []
    valori = []
    for f_name, type_name in metric_files.items():
        with open(os.path.join(data_for_plot, f_name), 'r') as file:
            for linea in file:
                indice, valore = linea.strip().split(": ")

                valore_troncato = round(float(valore), 3)

                indici.append(indice)
                valori.append(valore_troncato)


        plt.figure(figsize=(12, 10))

        plt.plot(indici, valori, marker='o', linestyle='-')

        plt.title(str(type_name) + " al variare di K")
        plt.xlabel('K')
        plt.ylabel('Valore')

        plt.grid(True)
        plt.savefig(os.path.join(plot_path,type_name))
        plt.show()
        indici.clear()
        valori.clear()