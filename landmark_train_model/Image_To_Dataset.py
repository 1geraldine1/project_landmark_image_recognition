import os
from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import json
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import tensorboard
import Tools

BASE_DIR = Path(__file__).resolve().parent
data_dir = os.path.join(BASE_DIR, "landmark")
json_dir = os.path.join(BASE_DIR, "landmark_json")
cropped_data_dir = os.path.join(BASE_DIR,'cropped_landmark')
categories = os.listdir(data_dir)
num_classes = len(categories)


def create_dataset_noncrop_np():
    image_size = 128

    X = []
    Y = []

    for idx, category in enumerate(categories):
        label = [0 for i in range(num_classes)]
        label[idx] = 1
        category_dir = os.path.join(data_dir, category)

        for top, dir, f in os.walk(category_dir):
            for filename in f:
                img_dir = os.path.join(category_dir, filename)
                print(img_dir)
                ff = np.fromfile(img_dir, np.uint8)
                img = cv2.imdecode(ff, cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, None, fx=image_size / img.shape[1], fy=image_size / img.shape[0])
                X.append(img / 256)
                Y.append(label)

    X = np.array(X)
    Y = np.array(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    xy = (X_train, X_test, Y_train, Y_test)

    np.savez("./img_data_noncrop.npz", xy)


def create_image_data_crop():

    for idx, category in enumerate(categories):
        label = [0 for i in range(num_classes)]
        label[idx] = 1
        category_dir = os.path.join(data_dir, category)
        category_json_dir = os.path.join(json_dir, category)

        cropped_save_dir = os.path.join(cropped_data_dir,category)
        print(cropped_save_dir)

        for top, dir, f in os.walk(category_dir):
            for filename in f:
                # 불러올 디렉터리 지정
                img_dir = os.path.join(category_dir, filename)
                img_json_dir = os.path.join(category_json_dir, filename[:-4] + '.json')

                # 파일 불러오기
                with open(img_json_dir, "r", encoding='UTF-8') as j:
                    img_json = json.load(j)
                # 대부분의 경우, lx < rx, ly < ry
                lx, ly, rx, ry = img_json['regions'][0]['boxcorners']
                print(img_dir)

                # 경로에 한글 포함시 우회
                ff = np.fromfile(img_dir, np.uint8)
                img = cv2.imdecode(ff, cv2.IMREAD_UNCHANGED)

                # 이미지 자르기
                crop_img = img[ly:ry, lx:rx]

                # 예외처리. 가끔 lx > rx, ly > ry인 데이터 존재.
                # 해당 상황에서 crop_img.shape가 [0,0,3]이 되는 현상 발견
                if crop_img.shape[0] == 0:
                    crop_img = img[ry:ly, rx:lx]
                print(crop_img.shape)

                # 자른 이미지를 그대로 저장
                img = crop_img

                # 한글경로 인식
                extension = os.path.splitext(filename)[1]
                result, encoded_img = cv2.imencode(extension,img)

                # 인코드 성공시 파일 경로에 저장
                if result:
                    with open(os.path.join(cropped_save_dir,filename), mode="w+b") as f:
                        encoded_img.tofile(f)


create_image_data_crop()

