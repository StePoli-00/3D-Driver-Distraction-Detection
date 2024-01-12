import os.path
from PIL import Image
import cv2
import numpy as np
import FaissRetrieval as fr
query_path="Data/Query"

f_name=["img_2.jpg","img_3.jpg","img_4.jpg","img_8.jpg","img_9.jpg","img_10.jpg"]

images=[cv2.imread(os.path.join(query_path,f)) for f in f_name]


def resize_and_concat_images(image_list, output_size):
    # Crea una lista vuota per immagini ridimensionate
    resized_images = []

    # Ridimensiona tutte le immagini nella lista
    for img_path in image_list:
        img = Image.open(os.path.join(query_path,img_path))
        img = img.resize(output_size)
        resized_images.append(img)

    # Calcola la larghezza e l'altezza totale dell'immagine risultante
    total_width = sum(img.width for img in resized_images)
    max_height = sum(img.height for img in resized_images)

    # Crea un'immagine vuota con la dimensione totale calcolata
    result_image = Image.new("RGB", (total_width, max_height))

    # Copia le immagini ridimensionate nell'immagine risultante
    x_offset = 0
    for img in resized_images:
        result_image.paste(img, (x_offset, 0))
        x_offset += img.width

    return result_image


from PIL import Image
import math


def find_dim(num):
    for i in range(2, num):
        if num % i == 0:
            first = i
            second = num // i
            return first,second
    return None, None



def create_frame(query,image_list,retr_img):

    dim=len(image_list)
    additional=False
    output_dim=[]
    query = cv2.imread(os.path.join(query_path, query))
    final=[]
    if dim==1:
        img=cv2.imread(os.path.join(query_path,image_list[0]))
        img=cv2.resize(img,(400,200))
        output_dim=(400,200)
        final=img
        # query=cv2.resize(query,(400,200))
        # final=cv2.hconcat([img,query])

    else:

        if(dim%2!=0):
            dim+=1
            additional=True

        rows,cols=find_dim(dim)
        if(rows==None and cols==None):
            rows=1
            cols=dim
        output_dim=(rows*retr_img[0],cols*retr_img[1])
        img=[cv2.imread(os.path.join(query_path,im)) for im in image_list]
        img=[cv2.resize(im,retr_img) for im in img]

        if additional==True:
            img.append(np.ones((retr_img[1],retr_img[0], 3), dtype=np.uint8) * 255)
        grid=[]

        for r in range(rows):
            frame=[]
            for c in range(cols):
                if c==0:
                    frame = img[r*cols+c]
                else:
                    frame=cv2.vconcat([frame,img[r*cols+c]])
            grid.append(frame)


        final=[]
        for i in range(len(grid)):

            if i==0:
                final=grid[i]
            else:
                final=cv2.hconcat([final,grid[i]])

    query=cv2.resize(query,output_dim)
    final=cv2.hconcat([query,final])
    cv2.namedWindow('Retrieval results', cv2.WINDOW_NORMAL)
    cv2.imshow('Retrieval results', final)  # mostro a schermo i risultati
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return
        #concatenazione orizzontale


# Esempio di utilizzo
if __name__ == "__main__":
    input_images = ["img_2.jpg", "img_3.jpg", "img_4.jpg","img_7.jpg","img_8.jpg","img_9.jpg","img_10.jpg","img_11.jpg","img_12.jpg"]
    output_dim = 300  # Imposta il numero di immagini per riga desiderato

   # create_frame(input_images[0],[input_images[1]],(400,300))
    create_frame(input_images[0],input_images[1:3],(400,300))
    create_frame(input_images[0],input_images[1:8],(400,300))



# if __name__ == "__main__":
#     input_images = ["img_2.jpg", "img_3.jpg", "img_4.jpg","img_7.jpg","img_8.jpg","img_9.jpg"]  # Sostituire con i percorsi delle vostre immagini
#     output_size = (300, 200)  # Sostituire con le dimensioni desiderate per l'output
#     result_image = resize_and_concat_images(input_images, output_size)
#     result_image.save("output_image.jpg")  # Salva l'immagine risultanti
