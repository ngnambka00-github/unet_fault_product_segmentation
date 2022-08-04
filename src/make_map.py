import os
import shutil
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.draw import ellipse


# Tao map cho cac anh san pham khong bi loi
def make_map_normal(data_path):
    # 1. Lap qua cac folder khong bi loi
    for folder in sorted(os.listdir(data_path)):
        if (not folder.endswith("def")) and (not folder.startswith(".")) and (not folder.endswith("mask")):
            # 2. Tao ra folder map tuong ung: Class4 -> Class4_mask
            mask_folder = folder + "_mask"
            mask_folder = os.path.join(data_path, mask_folder)
            try: 
                shutil.rmtree(mask_folder)
            except: pass
            if not os.path.exists(mask_folder):
                os.mkdir(mask_folder)

            # 3. Lap qua cac file trong folder
            current_folder = os.path.join(data_path, folder)
            for file in tqdm(os.listdir(current_folder), desc=f"{folder}"):
                if file.endswith("png"):
                    # 4. Doc cac file, lay kich thuoc w, h
                    current_file = os.path.join(current_folder, file)
                    image = cv2.imread(current_file)
                    w, h = image.shape[0], image.shape[1]

                    # 5. Tao ra 1 file anh chi co mau den = kich thuoc w, h
                    mask_image = np.zeros((w, h), dtype=np.uint8)
                    mask_image = Image.fromarray(mask_image)

                    # 6. Luu file anh den vao thu muc mask
                    mask_image.save(os.path.join(mask_folder, file))

# Ve 1 vung loi tren anh map mau den
# Tra ve anh mau den co ve loi tren anh
def draw_defect(file, labels, w, h):
    # Lấy file id
    file_id = int(file.replace(".png", ""))

    # Lấy nhãn của file
    label = labels[file_id - 1]

    # Tách các thành phần trong nhãn
    label = label.replace("\t", "").replace("  ", " ").replace("  ", " ").replace("\n", "")
    label_array = label.split(" ")

    # Vẽ hình ellipse
    major, minor, angle, x_pos, y_pos = float(label_array[1]), float(label_array[2]), float(label_array[3]), float(
        label_array[4]), float(label_array[5])
    rr, cc = ellipse(y_pos, x_pos, r_radius=minor, c_radius=major, rotation=-angle)

    # Tạo ảnh màu đen
    mask_image = np.zeros((w, h), dtype=np.uint8)

    try:
        # Gán các điểm thuộc hình ellipse thành 1
        mask_image[rr, cc] = 1
    except:
        # Nếu lỗi chỉ gán các điểm trong ảnh
        rr_n = [min(511, rr[i]) for i in rr]
        cc_n = [min(511, cc[i]) for i in cc]
        mask_image[rr_n, cc_n] = 1
        # mask_image = Image.fromarray(mask_image)

    # Chuyển thành ảnh
    mask_image = np.array(mask_image, dtype=np.uint8)
    mask_image = Image.fromarray(mask_image)

    return mask_image

# Tao map cho cac anh san pham bi loi
def make_map_defect(data_path):
    # 1. Lap cac folder bi loi
    for folder in sorted(os.listdir(data_path)):
        if (folder.endswith("def")) and (not folder.startswith(".")):
            # 2. Tao thu muc mask tuong ung: VD: Class4_def -> Class4_def_mask
            mask_folder = folder + "_mask"
            mask_folder = os.path.join(data_path, mask_folder)
            try: 
                shutil.rmtree(mask_folder)
            except: 
                pass
            os.mkdir(mask_folder)

            current_folder = os.path.join(data_path, folder)
            # load txt file
            f = open(os.path.join(current_folder, 'labels.txt'))
            labels = f.readlines()
            f.close()

            # 3. Lap cac file trong thu muc Class4_def
            for file in tqdm(os.listdir(current_folder), desc=f"{folder}"):
                if file.find("(") > -1:
                    # xoa file neu bi trung (do dac thu du lieu)
                    os.remove(os.path.join(current_folder, file))
                    continue
                
                if file.endswith("png"):
                    # read image file
                    current_file = os.path.join(current_folder, file)
                    image = cv2.imread(current_file)
                    w, h = image.shape[0], image.shape[1]

                    # 4. Ve phan loi len tren nen den
                    mask_image = draw_defect(file, labels, w, h)

                    # 5. Luu vao file
                    mask_image.save(os.path.join(mask_folder, file))


if __name__ == "__main__":
    # dinh nghia bien
    data_path = "../dataset"
    # make_map_normal(data_path)
    make_map_defect(data_path)

