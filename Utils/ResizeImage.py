import cv2
import os
source="Data/Retrieval/To-resize"
dest="Data/Retrieval/Query"
def resize(images:list):


    for i in images:
        im=cv2.imread(os.path.join(source,i))
        im=cv2.resize(im,(640,440))
        cv2.imwrite(os.path.join(dest,i),im)




images=os.listdir(source)
resize(images)



