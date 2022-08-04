import tensorflow as tf
import numpy as np
import cv2
import os
import random
import segmentation_models as sm
sm.set_framework("tf.keras")
sm.framework()

BACKBONE = "resnet34"
preprocessing_input = sm.get_preprocessing(BACKBONE)

class Dataset: 
    def __init__(self, image_path, mask_path, w, h):
        self.image_path = image_path
        self.mask_path = mask_path
        self.w = w
        self.h = h

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.image_path[i])
        image = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_AREA)
        image = preprocessing_input(image)

        mask = cv2.imread(self.mask_path[i], cv2.IMREAD_UNCHANGED)
        image_mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_AREA)
        image_mask = [(image_mask == v) for v in [1]]
        image_mask = np.stack(image_mask, axis=-1).astype('float')

        return image, image_mask

class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size, shape, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shape = shape
        self.indexes = np.arange(self.shape)

    def __getitem__(self, i):
        # collect batch size
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return tuple(batch)

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


# load thong tin tu folder dataset de tao 2 bien image_path, mask_path
def load_path(data_path):
    # get normal image and mask
    classes = ["Class1", "Class2", "Class3", "Class4", "Class5", "Class6"]

    normal_image_path = []
    normal_mask_path = []
    for class_ in classes:
        current_folder = os.path.join(data_path, class_)

        for file in os.listdir(current_folder):
            if file.endswith("png") and (not file.startswith(".")):
                image_path = os.path.join(current_folder, file)
                mask_path = os.path.join(current_folder + "_mask", file)
                normal_image_path.append(image_path)
                normal_mask_path.append(mask_path)

    # get defect image and mask
    defect_image_path = []
    defect_mask_path = []
    for class_ in classes: 
        class_ = class_ + "_def"
        current_folder = os.path.join(data_path, class_)

        for file in os.listdir(current_folder):
            if file.endswith("png") and (not file.startswith(".")):
                image_path = os.path.join(current_folder, file)
                mask_path = os.path.join(current_folder + "_mask", file)
                defect_image_path.append(image_path)
                defect_mask_path.append(mask_path)

    # Normal: normal_mask_path, normal_image_path
    # Defect: defect_mask_path, defect_image_path

    # Xu ly imbalance data -> Upsampling
    idx = random.sample(range(len(normal_mask_path)), len(defect_mask_path))

    normal_mask_path_new = []
    normal_image_path_new = []

    for i in idx:
        normal_image_path_new.append(normal_image_path[i])
        normal_mask_path_new.append(normal_mask_path[i])

    image_path = normal_image_path_new + defect_image_path
    mask_path = normal_mask_path_new + defect_mask_path

    return image_path, mask_path
