# Description

서울에 위치한 랜드마크의 이미지를 입력받아 해당 랜드마크의 위치를 반환하고, 근처 관광지 정보를 카테고리별, 거리별로 제공해주는 프로젝트입니다.

# Files

이 프로젝트는 다음과 같은 구성으로 이루어져 있습니다.

* 랜드마크 이미지 모델 훈련 도구(landmark_train_model)
* 랜드마크 데이터 수집 도구(landmark_image_crawler)
* 랜드마크 이미지 인식 서비스(landmark_image_recognition)

각각에 대한 설명은 해당 repository의 README.md문서에 작성해두었습니다.

# Used-Data

이번 프로젝트에서 사용한 공공데이터는 다음과 같습니다.

* [서울시 유적지 현황(한국어)](http://data.seoul.go.kr/dataList/OA-13003/S/1/datasetView.do)
    
* [서울특별시 일반음식점 인허가 정보](http://data.seoul.go.kr/dataList/OA-16094/S/1/datasetView.do)
    
* [서울특별시 숙박업 인허가 정보](http://data.seoul.go.kr/dataList/OA-16044/S/1/datasetView.do)

* [서울특별시 대규모점포 인허가 정보](http://data.seoul.go.kr/dataList/OA-16096/S/1/datasetView.do)
   
* [서울시 문화시설 현황(한국어)](http://data.seoul.go.kr/dataList/OA-12993/S/1/datasetView.do)

* [AI-Hub 한국형 사물 이미지](https://aihub.or.kr/aidata/132)


# Development Environment

이번 프로젝트는 다음과 같은 환경에서 개발되었습니다.

## Hardware 

* OS : Windows 10
* CPU : AMD Ryzen 7 2700X
* RAM : 16GB

## Software

* Used-Framework : Django 3.2.6 (landmark_image_recognition)
* Used-Library : Tensorflow 2.6.0
* Used-Language : Python 3.7 , HTML5
* DevTool : Pycharm, Visual Studio Code

# Demonstration Video

이미지 클릭시 시연 영상이 실행됩니다.
[![시연 영상](http://img.youtube.com/vi/98SOzJ3Fnz0/0.jpg)](https://www.youtube.com/embed//98SOzJ3Fnz0)



# Etc

[제 블로그](https://1geraldine1.github.io/categories/#%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EB%A6%AC%EB%A9%94%EC%9D%B4%ED%81%AC)에서 이번 프로젝트의 개발 일지를 볼 수 있습니다.



