import json

from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from landmark.models import imageModel

# Create your views here.

def index(request):
    return render(request,'index.html')


def result(request):
    if request.method == 'POST':
        img = request.FILES['file']
        uploaded_img = imageModel(imgfile=img)
        uploaded_img.save() # DBMS에 저장 실행, INSERT SQL 실행

        return analysis(img,request)

    else:
        return render(request,'index.html')

def analysis(img,request):

    print('analysis : ',img.name)

    #이미지 분석


    #이미지 분석 결과 출력
    pred_landmark_name = 'Landmark_name'
    pred_accuracy = "~~ %"
    location = '서울특별시 종로구 세종로 세종대로 172'
    lat = '37.57111'
    lng = '126.97696'
    result = {'pred_landmark_name': pred_landmark_name,
              'pred_accuracy': pred_accuracy,
              'location': location,
              'lat': lat,
              'lng': lng,
              'api_key': settings.GOOGLE_MAPS_API_KEY}


    return render(request,'result.html', {'result':result})


def recommand(request):

    if request.method == 'GET':
        recommand_site = request.GET['recommand_site']
        print(recommand_site)

        if recommand_site == 'tour':
            recommand_site_txt = '관광지'
        elif recommand_site == 'restaurant':
            recommand_site_txt = '식당'
        elif recommand_site == 'hotel':
            recommand_site_txt = '숙소'
        elif recommand_site == 'culture':
            recommand_site_txt = '문화/예술'

    recommand_list = {}
    recommand_001 = {"name":'001',
                     "category":'cate01',
                     "lat":'lat01',
                     "lng":'lng01'}
    recommand_002 = {"name": '002',
                     "category": 'cate02',
                     "lat": 'lat02',
                     "lng": 'lng02'}
    recommand_003 = {"name": '003',
                     "category": 'cate01',
                     "lat": 'lat03',
                     "lng": 'lng03'}

    recommand_list['recommand_001'] = recommand_001
    recommand_list['recommand_002'] = recommand_002
    recommand_list['recommand_003'] = recommand_003

    result = {
        'recommand': recommand_site_txt,
        'recommand_list': recommand_list
    }
    print(result)

    return render(request,'recommand.html',{'result':result})
