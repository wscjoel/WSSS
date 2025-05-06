import numpy as np
import torch
import torch.utils.data as Data
import cv2
import os


class ACDC_dataset():
    def __init__(self, data_path="a/train"):
        self.data_path = data_path

    # 读取一对数据 输入的是images的路径
    def read_data(self, img_path, label_path):

        img = cv2.imread(img_path, -1)
        label = cv2.imread(label_path, -1)
        img = self.crop_or_pad_slice_to_size(img, 224, 224)
        label = self.crop_or_pad_slice_to_size(label, 224, 224)
        img[img > 255] = 255
        img[img < 0] = 0
        img = img / 255.0

        img = np.expand_dims(img, axis=0)
        return img, label

    def crop_or_pad_slice_to_size(self, slice, nx, ny):
        x, y = slice.shape

        x_s = (x - nx) // 2
        y_s = (y - ny) // 2
        x_c = (nx - x) // 2
        y_c = (ny - y) // 2

        if x > nx and y > ny:
            slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
        else:
            slice_cropped = np.zeros((nx, ny))
            if x <= nx and y > ny:
                slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
            elif x > nx and y <= ny:
                slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
            else:
                slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

        return slice_cropped

    def generate_XY(self):
        filenames = os.listdir(self.data_path + "/images")
        images = []
        labels = []
        for file in filenames:
            path = self.data_path + "/images/" + file
            path_l = self.data_path + "/labels/" + file
            img, label = self.read_data(path, path_l)
            images.append(img)
            labels.append(label)

        X = torch.tensor(images, dtype=torch.float32)
        Y = torch.tensor(labels, dtype=torch.float32)
        print("X:", X.shape)
        print("Y:", Y.shape)
        data = Data.TensorDataset(X, Y)
        return data


def resh():
    path = "test/images"
    path_l = "test/labels"
    filesnames = os.listdir(path)
    print(len(filesnames))
    for file in filesnames:
        img = cv2.imread(path + "/" + file, -1)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite("ACDC/" + path + "/" + file, img)

        label = cv2.imread(path_l + "/" + file, -1)
        label = cv2.resize(label, (224, 224), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite("ACDC/" + path_l + "/" + file, label)


if __name__ == "__main__":
    data = ACDC_dataset("ACDC/test")
    data.generate_XY()
