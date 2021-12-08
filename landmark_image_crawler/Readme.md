# Description

랜드마크 이미지 수집 / 관광지 데이터 수집 및 전처리에 관한 코드입니다.

# Files

* google_image_crawler.py : 랜드마크 이미지를 수집하는 간단한 코드입니다. 
  * crawling_target/landmark_img_target 내부에 저장한 랜드마크 리스트를 읽어 해당 랜드마크의 이미지를 검색, 저장합니다.
  * 구글 검색의 이미지 결과 부분을 사용하며, 약 60~70장 정도 수집할수 있습니다.
  * 이미지 수집 결과는 image_collected 폴더를 통해 확인할수 있습니다.
  * 검색을 통해 수집한 이미지가 해당 랜드마크를 정확히 표시하지 않는 경우가 존재하므로(행사 홍보 포스터, 데포르메한 이미지 등), 수집 결과에 대한 사람의 검수가 필요합니다.
* tour_site_data_crawler.py
  * crawling_target/tour_site_data 폴더 내부에 저장한 서울시 공공데이터들을 재가공, 전처리 및 데이터 수집을 진행하는 코드입니다.
  * 각 함수는 다음의 데이터를 재가공합니다.
    * culture_site_processing : [서울시 문화시설 현황(한국어)](http://data.seoul.go.kr/dataList/OA-12993/S/1/datasetView.do)
    * history_site_processing : [서울시 유적지 현황(한국어)](http://data.seoul.go.kr/dataList/OA-13003/S/1/datasetView.do)
    * traditional_market_site_processing : 서울시 전통시장 현황 데이터를 사용하나, 대규모점포 인허가 정보 데이터에 포함되어 있으므로 사용하지 않습니다.
    * hotel_site_processing : [서울특별시 숙박업 인허가 정보](http://data.seoul.go.kr/dataList/OA-16044/S/1/datasetView.do)
    * department_processing : [서울특별시 대규모점포 인허가 정보](http://data.seoul.go.kr/dataList/OA-16096/S/1/datasetView.do)
    * restaurant_processing : [서울특별시 일반음식점 인허가 정보](http://data.seoul.go.kr/dataList/OA-16094/S/1/datasetView.do)
  * 가공 결과는 tour_site_data_processed 폴더에 csv파일로 저장됩니다.
  * 처리 결과물인 csv파일은 공통적으로 주소, 이름, 위도, 경도 column이 포함되며, 특정 데이터에는 세부분류 또는 설명등의 column이 추가되어 있습니다.


* secrets.json
  * 각종 api키를 보관하는 json입니다. tour_site_data_crawler.py의 정상적인 실행을 위해서 필요하며, 네이버 지오코딩 api키와 비밀번호가 필요합니다.
    ```
    {
    "NAVER_GEOCODING_API_Client_ID": "YOUR_API_CLIENT_ID",
    "NAVER_GEOCODING_API_Client_PWD": "YOUR_API_CLIENT_PASSWORD"
    }
    ```
    
* Tools.py
  * 개발 과정에서 사용한 각종 함수들을 모아둔 파일입니다.
  * 각 함수들은 다음과 같은 역할을 수행합니다.
    * get_secret : 'secret.json'에서 해당 키값을 가져옵니다.
    * coordinate_change : 지도 좌표계가 일반적인 위경도 좌표계가 아닌 중부원점 좌표계 등으로 되어있는 경우가 있는데, 그런 경우에 좌표계를 변환하는 코드입니다.
    * geocoding_naver : 네이버 지오코딩을 활용하여 입력받은 장소의 위경도를 반환합니다.
    * geocoding_google : 구글 지오코딩을 활용하여 입력받은 장소의 위경도를 반환합니다.  
      * 무료 지오코딩 사용량의 한계로 사용을 중단했습니다.
    * reverse_geocoding_google : 구글 역지오코딩을 활용하여 장소의 위경도를 입력받아 주소지를 출력합니다.   
      * 혹시 몰라 구글 지오코딩 기능과 함께 만들었으나, 역시 무료 사용량의 한계로 사용을 중단했습니다.
    * csv_splitter : csv 파일을 newsize만큼의 row로 분할하여 저장합니다.
    * csv_geocoder : csv 파일을 열어 네이버 지오코딩을 수행, 위경도를 포함하여 저장합니다.
    * csv_to_initial_data_json : csv파일의 데이터를 sqlite에서 사용하는 데이터 모델로 변환하는 함수입니다. 최종 버전에서는 사용하지 않습니다.


# Usage

## landmark image crawling

랜드마크 이미지 수집기의 사용 방법은 다음과 같습니다.

1. crawling_target/landmark_img_target 폴더 내부에 landmark_list를 생성하여 넣습니다.
2. 터미널에서 다음과 같이 입력합니다 
```
python google_image_crawler.py
```

## tour_site_data_crawler

관광지 데이터 수집/전처리기의 사용 방법은 다음과 같습니다.

1. crawling_target/tour_site_data에 Files 항목에서 링크한 공공데이터 파일을 저장합니다.
2. 처리하고자 하는 데이터에 맞는 함수를 사용합니다.
3. 전처리 결과는 tour_site_data_processed 폴더 내부에 저장됩니다.

# data-architecture

랜드마크 이미지 수집기 결과로 저장되는 이미지들은 다음과 같은 구조를 가집니다. 이는 image_model_train.py 에서 모델을 훈련시킬때 사용됩니다.
image_model_train.py 을 통해 훈련시킬 이미지들은 하나의 폴더 내부에 클래스 단위로 구분되어진 폴더, 해당 폴더 내부에 각 클래스의 이미지를 저장하는 방식으로 구성되어야 합니다.

***
folder_name  
├── landmark_name  
│   ├── landmark1-1.jpg  
│   ├── landmark1-2.jpg  
│   └── landmark1-3.jpg  
│   └── ...  
├── landmark2_name  
├── landmark3_name  
├── landmark4_name  
└── ...

***
