import cv2
import os
import re
import numpy as np
import random


# Used to pre_process the images using the CLAHE technique
def img_pre_process(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_ch)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img


def img_augmentation(img):
    max_rotation_angle = 30  # Maximum rotation angle in degrees
    max_scaling_factor = 1.2  # Maximum scaling factor
    max_translation = 20  # Maximum translation in pixels

    # Randomly rotate the image
    rotation_angle = random.uniform(-max_rotation_angle, max_rotation_angle)
    rows, cols, _ = img.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
    rotated_image = cv2.warpAffine(img, rotation_matrix, (cols, rows))

    # Randomly scale the image
    scaling_factor = random.uniform(1, max_scaling_factor)
    scaled_image = cv2.resize(rotated_image, None, fx=scaling_factor, fy=scaling_factor)

    # Randomly shift the image
    x_translation = random.randint(-max_translation, max_translation)
    y_translation = random.randint(-max_translation, max_translation)
    translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
    shifted_image = cv2.warpAffine(scaled_image, translation_matrix, (cols, rows))

    return rotated_image, scaled_image, shifted_image


# function used to check if the current sample has to be augmented
def check_label(label, classes_dict, sel_classes):
    for key in classes_dict.keys():
        if label in classes_dict[key] and int(key) in sel_classes:
            return True
    return False


def img_processor(names, drivers, img_df, classes_dict, sel_classes, path):
    class_values = []
    for values in classes_dict.values():
        for cx in values:
            class_values.append(cx)

    for name in names:
        subj_df = drivers[drivers['subject'] == name]

        for ucl in class_values:
            curr_class = subj_df[subj_df['classname'] == ucl]

            for idx, row in curr_class.iterrows():
                img_name = row['img']
                img_row = img_df[img_df['file_name'] == img_name]
                sample_data = img_row['sample'].values[0]

                if isinstance(sample_data, np.ndarray):
                    processed_img = img_pre_process(sample_data)
                    exp_path = path + ucl + "/" + name + "/"

                    if not os.path.exists(exp_path):
                        os.makedirs(exp_path)

                    processed_img_path = os.path.join(exp_path, img_name)
                    cv2.imwrite(processed_img_path, processed_img)

                    if check_label(ucl, classes_dict, sel_classes):
                        rot_img, scal_img, shift_img = img_augmentation(processed_img)
                        parts = img_name.split(".")
                        rot_name = parts[0] + "_rot." + parts[1]
                        scal_name = parts[0] + "_scal." + parts[1]
                        shift_name = parts[0] + "_shift." + parts[1]
                        rot_img_path = os.path.join(exp_path, rot_name)
                        scal_img_path = os.path.join(exp_path, scal_name)
                        shift_img_path = os.path.join(exp_path, shift_name)
                        cv2.imwrite(rot_img_path, rot_img)
                        cv2.imwrite(scal_img_path, scal_img)
                        cv2.imwrite(shift_img_path, shift_img)
                else:
                    print(f" Invalid image data.")


# Used to load images from disk;
# mode: parameter that specifies loading classes for "train" or "test" tasks (they have different path structures);
# path: parameter that specifies the img files path;
# classes: array with indexes of classes to extract; can be set to empty [] if you need to load all the img classes
def load_images(mode, path, classes=[], n_images=-1,shuffle=False):
    subjects = []
    imgs = []
    labels = []
    file_names = []
    samples_num_dict = {}

    if mode == "train":
        print(f" - selected classes: {classes} ")

        for cn in os.listdir(path):
            if cn == ".DS_Store":
                continue

            if int(cn[1]) not in classes:
                continue
            else:
                print(f" cn: {cn[1]}")

            curr_path = path + cn + "/"

            print(curr_path)

            for subject in os.listdir(curr_path):
                if subject == ".DS_Store":
                    continue

                sub_path = curr_path + subject + "/"

                for sample in os.listdir(sub_path):
                    if sample == ".DS_Store":
                        continue

                    img = cv2.imread(os.path.join(sub_path, sample))

                    if img is not None:
                        subjects.append(subject)
                        imgs.append(img)
                        labels.append(int(cn[1]))
                        file_names.append(sample)

        return subjects, imgs, labels, file_names

    elif mode == "test":

        if(n_images==-1):
            n_images=len(os.listdir(path))

        samples = os.listdir(path)
        if shuffle==True:
            random.shuffle(samples)
        for sample in samples:
            img = cv2.imread(os.path.join(path, sample))

            if img is not None:
                imgs.append(img)
                file_names.append(sample)

            if len(imgs) == n_images:
                return imgs, file_names

        return imgs, file_names

    elif mode == "img":

        for cn in os.listdir(path):
            if cn == ".DS_Store":
                continue

            cn_file_names = []
            int_label = int(re.findall(r'\d+', cn)[0])

            if int_label not in classes:
                continue

            curr_path = path + cn + "/"
            print(f" exploring path: {curr_path}")

            for object in os.listdir(curr_path):
                if object == ".DS_Store":
                    continue

                sample = curr_path + object

                if sample == ".DS_Store":
                    continue

                img = cv2.imread(sample)

                if img is not None:
                    imgs.append(img)
                    labels.append(int_label)
                    file_names.append(object)

                    cn_file_names.append(object)

            samples_num_dict[cn] = cn_file_names

        return imgs, labels, file_names, samples_num_dict
