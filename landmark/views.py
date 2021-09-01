import json

from django.http import HttpResponse
from django.shortcuts import render, redirect
from landmark.models import imageModel

# Create your views here.

def index(request):
    image = imageModel.objects.all()
    return render(request,'index.html',{'image':image})


def imageupload(request):
    if request.method == 'POST':
        img = request.FILES['file']
        uploaded_img = imageModel(imgfile=img)
        uploaded_img.save() # DBMS에 저장 실행, INSERT SQL 실행

    image = imageModel.objects.all()

    return render(request,'index.html',{'image':image})




