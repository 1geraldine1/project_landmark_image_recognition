import json

from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from landmark.models import imageModel

import tensorflow as tf
from pathlib import Path
import os
from PIL import Image
import pandas as pd
import numpy as np
from haversine import haversine

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = os.path.join(BASE_DIR, 'data_model')
MEDIA_DIR = os.path.join(BASE_DIR.parent, 'media')
LOCATION_DIR = os.path.join(BASE_DIR, 'location_data')

# 이미지 분석 모델 로드
model = tf.keras.models.load_model(MODEL_DIR)

landmark_data = pd.read_csv(os.path.join(LOCATION_DIR, 'landmark_list.csv'))
labels = landmark_data['이름']

# Create your views here.

def index(request):
    return render(request, 'index.html')


def result(request):
    if request.method == 'POST':
        img = request.FILES['file']
        uploaded_img = imageModel(imgfile=img)
        uploaded_img.save()  # DBMS에 저장 실행, INSERT SQL 실행

        recognition_label = recognition(img)
        uploaded_img.delete() # 인식 완료된 이미지는 삭제

        return analysis(recognition_label, request)
        # return analysis_new(uploaded_img.name,request)

    else:
        return render(request, 'index.html')


def recognition(img):
    characters = "\"'!?()"
    imgname = ''.join(x for x in img.name if x not in characters)
    imgname = imgname.replace(' ', '_')

    print('analysis : ', img.name)

    img_filepath = os.path.join(MEDIA_DIR, imgname)
    print(img_filepath)
    image = Image.open(img_filepath)
    image = image.resize((600, 600), Image.LANCZOS)
    img_arr = tf.reshape(np.asarray(image), [1, 600, 600, 3])
    print(img_arr.shape)
    pred = model.predict(img_arr)
    pred_label = labels[np.argmax(pred[0])]
    print(pred_label)

    return pred_label


def analysis(recognition_label, request):
    data = landmark_data[landmark_data['이름'] == recognition_label]

    # 이미지 분석 결과 출력
    pred_landmark_name = recognition_label
    location = data['주소'].values[0]
    lat = data['위도'].values[0]
    lng = data['경도'].values[0]
    result = {'pred_landmark_name': pred_landmark_name,
              'location': location,
              'lat': lat,
              'lng': lng,
              'api_key': settings.GOOGLE_MAPS_API_KEY}

    return render(request, 'result.html', {'result': result})





def recommand(request):
    if request.method == 'GET':
        recommand_site = request.GET['recommand_site']
        recommand_lat = request.GET['lat']
        recommand_lng = request.GET['lng']

        recommand_site_data = None

        if recommand_site == 'tour':
            recommand_site_txt = '관광지'
            recommand_site_data = pd.read_csv(os.path.join(LOCATION_DIR, 'seoul_history_processed.csv'))
            recommand_site_data = recommand_site_data.dropna(axis=0)
        elif recommand_site == 'restaurant':
            recommand_site_txt = '식당'
            recommand_site_data = pd.read_csv(os.path.join(LOCATION_DIR, 'seoul_restaurant_processed.csv'))
            recommand_site_data = recommand_site_data.dropna(axis=0)
        elif recommand_site == 'hotel':
            recommand_site_txt = '숙소'
            recommand_site_data = pd.read_csv(os.path.join(LOCATION_DIR, 'seoul_hotel_processed.csv'))
            recommand_site_data = recommand_site_data.dropna(axis=0)
        elif recommand_site == 'culture':
            recommand_site_txt = '문화/예술'
            recommand_site_data = pd.read_csv(os.path.join(LOCATION_DIR, 'seoul_culture_processed.csv'))
            recommand_site_data = recommand_site_data.dropna(axis=0)
        elif recommand_site == 'department':
            recommand_site_txt = '상점'
            recommand_site_data = pd.read_csv(os.path.join(LOCATION_DIR, 'seoul_department_processed.csv'))
            recommand_site_data = recommand_site_data.dropna(axis=0)

        recommand_list = {}

        print(recommand_lat, recommand_lng)
        if recommand_site_data is not None:
            for idx, row in recommand_site_data.iterrows():
                address = row['주소']
                name = row['이름']
                lat = row['위도']
                lng = row['경도']

                start = (float(recommand_lat), float(recommand_lng))
                goal = (float(lat),float(lng))
                dist = round(haversine(start, goal),4)

                if dist > 1.5:
                    continue
                elif 1.0 < dist <= 1.5:
                    dist_filter = 2
                elif 0.5 < dist <= 1.0:
                    dist_filter = 1
                else:
                    dist_filter = 0

                recommand_dict = {"address": address,
                                  "name": name,
                                  "lat": lat,
                                  "lng": lng,
                                  "dist": dist,
                                  "dist_filter": dist_filter}

                if recommand_site == 'tour':
                    explain = row['설명']
                    recommand_dict['explain'] = explain
                elif recommand_site == 'restaurant':
                    category = row['카테고리']
                    recommand_dict['category'] = category
                elif recommand_site == 'culture':
                    category = row['세부분류']
                    recommand_dict['category'] = category

                recommand_list[idx] = recommand_dict

        result = {
            'recommand': recommand_site_txt,
            'recommand_list': recommand_list,
            'api_key': settings.GOOGLE_MAPS_API_KEY
        }
        print(result)

        return render(request, 'recommand.html', {'result': result})

    else:
        return render(request, 'error_page.html')


def recommand_new(request):
    dist_filter = request.POST['dist_filter']

