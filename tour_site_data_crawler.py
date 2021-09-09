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

    for city, district, dong, site_name in df[['행정 시', '행정 구', '행정 동', '명칭']].iloc:
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

    culture_site_data_filename = "서울시 유적지 현황 (한국어).csv"
    culture_site_data_fullpath = Path(BASE_DIR, tour_site_data_dir, culture_site_data_filename)

    df = pd.read_csv(culture_site_data_fullpath, encoding="CP949")
    out_df = pd.DataFrame(df['명칭'], columns=['이름'])
    address = []
    new_name = []
    explain = []

    for city, district, dong, site_name in df[['행정 시', '행정 구', '행정 동', '명칭']].iloc:
        print("keyword : ", city, district, dong, site_name)
        elem = driver.find_element_by_name("q")
        elem.clear()
        elem.send_keys(city + ' ' + district + ' ' + dong + ' ' + site_name)
        elem.send_keys(Keys.RETURN)

        # 검색 결과가 정상적으로 나올때
        try:
            target = driver.find_element_by_class_name('SPZz6b').find_element_by_tag_name('span')
            target_new_name = driver.find_element_by_class_name('LrzXr')
            target_explain = driver.find_element_by_class_name('kno-rdesc').find_element_by_tag_name('span')

            address.append(target.text)
            new_name.append(target_new_name.text)
            explain.append(target_explain.text)

            print(target.text)
            print(target_new_name.text)
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
                address.append('NaN')
                new_name.append(site_name)
                explain.append('NaN')
        finally:
            time.sleep(1)

    out_df['주소'] = address
    out_df['이름'] = new_name
    out_df['설명'] = explain
    

    processed_filename = "seoul_history_processed.csv"
    processed_data_fullpath = Path(BASE_DIR, processed_data_dir, processed_filename)
    out_df.to_csv(processed_data_fullpath, mode='w', encoding='UTF-8')

    print(out_df)
    driver.quit()


history_site_processing()
