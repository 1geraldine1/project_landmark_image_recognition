# Description

landmark_image_crawler를 통해 수집한 랜드마크 이미지를 기반으로 이미지 인식 모델을 훈련하기 위한 소스코드입니다.

# data-architecture

landmark_image_crawler 이외의 방법으로 랜드마크의 이미지를 수집하였을때 image_model_train.py 코드를 사용하기 위한 데이터 구조도입니다.
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

* Image_model_train.py : 현재 프로젝트에서 사용되는 이미지 인식 모델의 훈련 코드가 포함되어 있습니다. 
  * model_cnn함수는 간단한 CNN모델을 구성하여 빠르고 쉽게 이미지 인식 모델을 제작할수 있습니다.
  * model_efficientnet함수는 efficientnetB7모델을 전이학습하는 방식으로 이미지 인식 모델의 훈련을 진행합니다. 이 함수를 사용하는것을 권장합니다.
* Image_to_Dataset.py
  * create_dataset_noncrop_np함수는 지정한 디렉터리의 이미지를 모아 npz형태의 데이터셋으로 변환합니다.
  * create_image_data_crop함수는 [AI-HUB의 한국형 사물 이미지](https://aihub.or.kr/aidata/132) 데이터중 랜드마크 이미지를 사용할때 랜드마크 이미지의 영역을 표시해주는 json파일을 통해 랜드마크 이미지의 crop을 진행하여 저장해줍니다. 이 작업을 통해 이미지 인식 모델의 정확성을 높일수 있지만, 영역을 표시한 json파일이 별도로 필요합니다.




