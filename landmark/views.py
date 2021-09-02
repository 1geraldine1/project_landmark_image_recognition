import json

from django.http import HttpResponse
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
              'lng': lng}


    return render(request,'result.html', {'result':result})


def recommand(request):

    recommand_data = {}

    return render(request,'recommand.html',{'result':result})
