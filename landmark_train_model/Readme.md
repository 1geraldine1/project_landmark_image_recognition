# Description

landmark_image_crawler를 통해 수집한 랜드마크 이미지를 기반으로 이미지 인식 모델을 훈련하기 위한 소스코드입니다.

# data-architecture

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

# Usage

사용 방법은 다음과 같습니다.

1. 위의 data-architecture로 구성된 이미지 폴더를 Image_model_train.py와 같은 폴더에 넣습니다.
2. Image_model_train.py 의 data_dir의 큰따옴표 부분에 이미지 폴더의 이름을 넣습니다.
```
data_dir = os.path.join(BASE_DIR,"folder_name")
```
3. 터미널에서 다음과 같이 입력합니다.
```
python Image_model_train.py
```
4. 훈련이 정상적으로 종료되었다면 세개의 폴더가 생성됩니다.  
4-1. 'checkpoint'폴더는 훈련 도중 최선의 가중치를 기록하는 체크포인트를 저장하는 폴더입니다.  
4-2. 'efficientnetB7_full_data_train_done_fin' 폴더는 모델이 저장된 폴더로, 해당 폴더의 이름은 model_efficientnet 함수의 model.save() 부분을 변경하여 바꿀수 있습니다. 
landmark_image_recognition은 해당 폴더를 사용하는것을 전제로 제작하였습니다.  
4-3. 'trained_model'폴더는 모델을 각각 json, h5, pkl 형태로 저장하여 보관합니다.  

# Files

각 파일들의 역할은 다음과 같습니다.




