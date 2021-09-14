import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from tqdm import tqdm
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
import urllib.request
import Tools

BASE_DIR = Path(__file__).resolve().parent
tour_site_data_dir = "./crawling_target/tour_site_data/"
processed_data_dir = "./tour_site_data_processed/"


def culture_site_processing():
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get("https://www.google.co.kr")

    culture_site_data_filename = "서울시 문화시설 현황 (한국어).csv"
    culture_site_data_fullpath = Path(BASE_DIR, tour_site_data_dir, culture_site_data_filename)

    df = pd.read_csv(culture_site_data_fullpath, encoding="CP949")
    out_df = pd.DataFrame(df['명칭'], columns=['이름'])
    address = []
    new_name = []

    for city, district, dong, site_name in tqdm(df[['행정 시', '행정 구', '행정 동', '명칭']].iloc):
        print("keyword : ", city, district, dong, site_name)
        elem = driver.find_element_by_name("q")
        elem.clear()
        elem.send_keys(city + ' ' + district + ' ' + dong + ' ' + site_name)
        elem.send_keys(Keys.RETURN)

        # 검색 결과가 정상적으로 나올때
        try:
            target = driver.find_element_by_class_name('SPZz6b').find_element_by_tag_name('span')
            target_new_name = driver.find_element_by_class_name('LrzXr')
            address.append(target.text)
            new_name.append(target_new_name.text)

            print(target.text)
            print(target_new_name.text)

        # 검색 결과가 두건 이상 나오거나, 검색 결과가 없을때 재검색 실행
        except Exception:
            try:
                print('2차검색 시작')
                elem = driver.find_element_by_name("q")
                elem.clear()
                elem.send_keys(site_name)
                elem.send_keys(Keys.RETURN)

                wait = WebDriverWait(driver, 4).until(EC.presence_of_element_located((By.CLASS_NAME, 'SPZz6b')))
                target_new_name = driver.find_element_by_class_name('SPZz6b').find_element_by_tag_name('span')
                target_address = driver.find_element_by_class_name('LrzXr')

                new_name.append(target_new_name.text)
                address.append(target_address.text)

                print(target_new_name.text)
                print(target_address.text)

            # 검색결과가 존재하지 않을때
            except Exception:
                print("검색결과가 없습니다 : ", site_name)
                address.append('NaN')
                new_name.append(site_name)
        finally:
            time.sleep(1)

    out_df['세부분류'] = df['분류3']
    out_df['주소'] = address
    out_df['이름'] = new_name

    processed_filename = "seoul_culture_processed.csv"
    processed_data_fullpath = Path(BASE_DIR, processed_data_dir, processed_filename)
    out_df.to_csv(processed_data_fullpath, mode='w', encoding='UTF-8')

    print(out_df)
    driver.quit()


def history_site_processing():
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get("https://www.google.co.kr")

    history_site_data_filename = "서울시 유적지 현황 (한국어).csv"
    history_site_data_fullpath = Path(BASE_DIR, tour_site_data_dir, history_site_data_filename)

    df = pd.read_csv(history_site_data_fullpath, encoding="CP949")
    out_df = pd.DataFrame(df['명칭'], columns=['이름'])
    address = []
    new_name = []
    explain = []

    for city, district, dong, site_name in tqdm(df[['행정 시', '행정 구', '행정 동', '명칭']].iloc):
        print("keyword : ", city, district, dong, site_name)
        elem = driver.find_element_by_name("q")
        elem.clear()
        elem.send_keys(city + ' ' + district + ' ' + dong + ' ' + site_name)
        elem.send_keys(Keys.RETURN)

        # 검색 결과가 정상적으로 나올때
        try:
            target_new_name = driver.find_element_by_class_name('SPZz6b').find_element_by_tag_name('span')
            target_address = driver.find_element_by_class_name('LrzXr')
            target_explain = driver.find_element_by_class_name('kno-rdesc').find_element_by_tag_name('span')

            new_name.append(target_new_name.text)
            address.append(target_address.text)
            explain.append(target_explain.text)

            print(target_new_name.text)
            print(target_address.text)
            print(target_explain.text)

        # 검색 결과가 두건 이상 나오거나, 검색 결과가 없을때 재검색 실행
        except Exception:
            try:
                print('2차검색 시작')
                elem = driver.find_element_by_name("q")
                elem.clear()
                elem.send_keys(site_name)
                elem.send_keys(Keys.RETURN)

                wait = WebDriverWait(driver, 4).until(EC.presence_of_element_located((By.CLASS_NAME, 'SPZz6b')))
                target_new_name = driver.find_element_by_class_name('SPZz6b').find_element_by_tag_name('span')
                target_address = driver.find_element_by_class_name('LrzXr')
                target_explain = driver.find_element_by_class_name('kno-rdesc').find_element_by_tag_name('span')

                new_name.append(target_new_name.text)
                address.append(target_address.text)
                explain.append(target_explain.text)

                print(target_new_name.text)
                print(target_address.text)
                print(target_explain.text)

            # 검색결과가 존재하지 않을때
            except Exception:
                print("검색결과가 없습니다 : ", site_name)
                new_name.append(site_name)
                address.append('NaN')
                explain.append('NaN')
        finally:
            time.sleep(1)

    out_df['주소'] = address
    out_df['이름'] = new_name
    out_df['설명'] = explain

    out_df = out_df.drop_duplicates(subset=['이름'], keep='first', ignore_index=True)

    processed_filename = "seoul_history_processed.csv"
    processed_data_fullpath = Path(BASE_DIR, processed_data_dir, processed_filename)
    out_df.to_csv(processed_data_fullpath, mode='w', encoding='UTF-8')

    print(out_df)
    driver.quit()


def traditional_market_site_processing():
    traditional_market_site_data_filename = "서울시 전통시장 현황.csv"
    traditional_site_data_fullpath = Path(BASE_DIR, tour_site_data_dir, traditional_market_site_data_filename)

    df = pd.read_csv(traditional_site_data_fullpath, encoding="CP949")
    out_df = pd.DataFrame()

    out_df['주소'] = df['주소명']
    out_df['이름'] = df['전통시장명']
    out_df['위도'] = df['위도']
    out_df['경도'] = df['경도']

    processed_filename = "seoul_traditional_market_processed.csv"
    processed_data_fullpath = Path(BASE_DIR, processed_data_dir, processed_filename)
    out_df.to_csv(processed_data_fullpath, mode='w', encoding='UTF-8')

    print(out_df)


def hotel_site_processing():
    tour_hotel_site_data_filename = "서울특별시 숙박업 인허가 정보.csv"
    tour_hotel_site_data_fullpath = Path(BASE_DIR, tour_site_data_dir, tour_hotel_site_data_filename)

    df = pd.read_csv(tour_hotel_site_data_fullpath, encoding="CP949")
    out_df = pd.DataFrame()

    address = []
    site_name = []
    lat_arr = []
    lng_arr = []

    for line in tqdm(df.iloc):
        if line['영업상태코드'] == 1:
            if line['도로명주소'] is not np.nan:
                addr = line['도로명주소']
            elif line['지번주소'] is not np.nan:
                addr = line['지번주소']
            else:
                addr = ''
            address.append(addr)

            site_name.append(line['사업장명'])

            lat, lng = Tools.geocoding_naver(addr)

            lat_arr.append(lat)
            lng_arr.append(lng)

    out_df['주소'] = address
    out_df['이름'] = site_name
    out_df['위도'] = lat_arr
    out_df['경도'] = lng_arr

    processed_filename = "seoul_hotel_processed.csv"
    processed_data_fullpath = Path(BASE_DIR, processed_data_dir, processed_filename)
    out_df.to_csv(processed_data_fullpath, mode='w', encoding='UTF-8')

    print(out_df)


def department_processing():
    department_site_data_filename = "서울특별시 대규모점포 인허가 정보.csv"
    department_site_data_fullpath = Path(BASE_DIR, tour_site_data_dir, department_site_data_filename)

    df = pd.read_csv(department_site_data_fullpath, encoding="CP949")
    out_df = pd.DataFrame()

    address = []
    site_name = []
    lat_arr = []
    lng_arr = []

    for line in tqdm(df.iloc):
        if line['영업상태코드'] == 1:
            if line['도로명주소'] is not np.nan:
                addr = line['도로명주소']
            elif line['지번주소'] is not np.nan:
                addr = line['지번주소']
            else:
                addr = ''
            address.append(addr)

            site_name.append(line['사업장명'])

            lat, lng = Tools.geocoding_naver(addr)

            lat_arr.append(lat)
            lng_arr.append(lng)

    out_df['주소'] = address
    out_df['이름'] = site_name
    out_df['위도'] = lat_arr
    out_df['경도'] = lng_arr

    out_df = out_df.replace([np.inf], np.nan)
    out_df = out_df.drop_duplicates(subset=['이름'], keep='first', ignore_index=True)
    out_df = out_df.dropna(axis=0, subset=['주소', '위도', '경도'])

    processed_filename = "seoul_department_processed.csv"
    processed_data_fullpath = Path(BASE_DIR, processed_data_dir, processed_filename)
    out_df.to_csv(processed_data_fullpath, mode='w', encoding='UTF-8')

    print(out_df)


def restaurant_processing():
    restaurant_site_data_filename = "서울특별시 일반음식점 인허가 정보.csv"
    restaurant_site_data_fullpath = Path(BASE_DIR, tour_site_data_dir, restaurant_site_data_filename)

    df = pd.read_csv(restaurant_site_data_fullpath, encoding="CP949")
    out_df = pd.DataFrame()

    address = []
    site_name = []
    lat_arr = []
    lng_arr = []
    category = []

    for line in tqdm(df.iloc):
        if line['영업상태코드'] == 1:
            if line['도로명주소'] is not np.nan:
                addr = line['도로명주소']
            elif line['지번주소'] is not np.nan:
                addr = line['지번주소']
            else:
                addr = ''
            address.append(addr)

            site_name.append(line['사업장명'])

            lat, lng = Tools.geocoding_naver(addr)

            lat_arr.append(lat)
            lng_arr.append(lng)

            if line['업태구분명'] is not np.nan:
                category.append(line['업태구분명'])
            else:
                category.append('기타 음식점')

    out_df['카테고리'] = category
    out_df['주소'] = address
    out_df['이름'] = site_name
    out_df['위도'] = lat_arr
    out_df['경도'] = lng_arr

    processed_filename = "seoul_restaurant_processed2.csv"
    processed_data_fullpath = Path(BASE_DIR, processed_data_dir, processed_filename)
    out_df.to_csv(processed_data_fullpath, mode='w', encoding='UTF-8')

    print(out_df)


restaurant_processing()
