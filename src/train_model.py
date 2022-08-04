from sklearn.model_selection import train_test_split
import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import cv2
import numpy as np
from utils import DataLoader, Dataset, load_path
import warnings
warnings.filterwarnings("ignore")
from segmentation_models.metrics import iou_score
from segmentation_models import Unet
import segmentation_models as sm
from hydra import initialize, compose
from omegaconf import OmegaConf

HOME_PATH = "../"
with initialize(config_path="../configs/"):
    data_cfg = compose(config_name="data_path")
    parameter_cfg = compose(config_name="hyper_parameter")
data_cfg = OmegaConf.create(data_cfg)
parameter_cfg = OmegaConf.create(parameter_cfg)

CHECKPOINT_PATH = os.path.join(HOME_PATH, data_cfg.model.checkpoint)

# Import thu vien segmentation_models
sm.set_framework(parameter_cfg.model_train.framework)
sm.framework()
BACKBONE = parameter_cfg.model_train.backbone
preprocessing_input = sm.get_preprocessing(BACKBONE)

# Dinh nghia bien
data_path = os.path.join(HOME_PATH, data_cfg.data.data_path)
w, h = parameter_cfg.model_train.input_weight, parameter_cfg.model_train.input_height
batch_size = parameter_cfg.model_train.batch_size
no_epochs = parameter_cfg.model_train.number_epochs

# load duong dan
image_path, mask_path = load_path(data_path)

# chia du lieu train va test
image_train, image_test, mask_train, mask_test = train_test_split(image_path, mask_path, test_size=0.2, random_state=100)

# Tao dataset va dataloader
train_dataset = Dataset(image_train, mask_train, w, h)
test_dataset = Dataset(image_test, mask_test, w, h)

train_loader = DataLoader(train_dataset, batch_size, shape=len(image_train), shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shape=len(image_test), shuffle=False)

# Khoi tao model
# phan loai loi hay khong loi
model = Unet(
    BACKBONE, 
    encoder_weights="imagenet", 
    classes=1, 
    activation="sigmoid", 
    input_shape=(w, h, 3), 
    encoder_freeze=True
)

loss_func = sm.losses.categorical_focal_dice_loss
model.compile(optimizer="adam", loss=loss_func, metrics=[iou_score])

# train model
is_train = False
if is_train:
    filepath=CHECKPOINT_PATH
    callback = ModelCheckpoint(filepath, monitor="val_iou_score", verbose=1, save_best_only=True, mode="max")
    model.fit(
        train_loader,
        validation_data=test_loader, 
        epochs=no_epochs, 
        callbacks=[callback]
    )
else: 
    # load model de test
    model.load_weights(CHECKPOINT_PATH)
    
    ids = range(len(image_test))
    indexes = random.sample(ids, 10)

    for id in indexes:
        # Anh dau vao
        image = cv2.imread(image_test[id])
        image = cv2.resize(image, (w, h))

        # dua model de predict mask
        mask_predict = model.predict(image[np.newaxis, :, :, :])

        # Doc anh mask thuc te
        image_mask = cv2.imread(mask_test[id], cv2.IMREAD_UNCHANGED)
        image_mask = cv2.resize(image_mask, (w, h))

        plt.figure(figsize=(10, 6))
        plt.subplot(131)
        plt.title("Hinh anh san pham")
        plt.imshow(image)

        plt.subplot(132)
        plt.title("Vet loi that")
        plt.imshow(image_mask, cmap="gray")

        plt.subplot(133)
        plt.title("Vet loi du doan")
        plt.imshow(mask_predict[0], cmap="gray")

        plt.show()
    