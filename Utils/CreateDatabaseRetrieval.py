import os
import random
import shutil

source_path="Data/state-farm-distracted-driver-detection/imgs/subj/train"
dest_path="Data/Retrieval/Database"
classes = ["c0", "c6", "c8", "c9"]
classes_with_phone = ["c1", "c2", "c3", "c4"]

def check_dimension(subject,img_per_classes):
    img_per_subj = 0
    if subject < img_per_classes:
        img_per_subj = img_per_classes // subject
    else:
        img_per_subj = 1

    return img_per_subj
def create_database(n_images):
    img_per_classes = n_images // 5
    img_per_classes_with_phone=img_per_classes//4


    #os.listdir(source_path)
    for c in classes:
      i=0
      subject=os.listdir(os.path.join(source_path,c))
      img_per_subj=check_dimension(len(subject),img_per_classes)
      isEnough=False
      for s in os.listdir(os.path.join(source_path,c)):

        files=os.listdir(os.path.join(source_path,c,s))
        for j in range(img_per_subj):
            index = random.randint(0, len(files) - 1)
            try:
                if i<img_per_classes:
                    shutil.copy(os.path.join(source_path, c, s, files[index]), dest_path)
                    i+=1
                else:
                    isEnough=True
                    break
            except FileExistsError:
                pass
        if isEnough==True:
            break

    for c in classes_with_phone:
        i = 0
        subject = os.listdir(os.path.join(source_path, c))
        img_per_subj = check_dimension(len(subject), img_per_classes_with_phone)
        isEnough = False
        for s in os.listdir(os.path.join(source_path, c)):

            files = os.listdir(os.path.join(source_path, c, s))
            for j in range(img_per_subj):
                index = random.randint(0, len(files) - 1)
                try:
                    if i < img_per_classes_with_phone:
                        shutil.copy(os.path.join(source_path, c, s, files[index]), dest_path)
                        i += 1
                    else:
                        isEnough = True
                        break
                except FileExistsError:
                    pass
            if isEnough == True:
                break


    return 0


create_database(400)