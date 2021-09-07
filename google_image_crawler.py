import os
import time
from pathlib import Path
from tqdm import tqdm
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
import urllib.request

driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get("https://www.google.co.kr/imghp?hl=ko&tab=wi&authuser=0&ogbl")

BASE_DIR = Path(__file__).resolve().parent
landmark_list_dir = "./crawling_target/landmark_img_target/"
landmark_list_filename = "landmark_list.txt"

landmark_list_fullpath = Path(BASE_DIR, landmark_list_dir, landmark_list_filename)

with open(landmark_list_fullpath, 'r', encoding='UTF-8') as f:
    for line in tqdm(f.readlines()):
        # 줄넘김문자 제거
        landmark_name = line.replace('\n','')
        print(landmark_name)

        elem = driver.find_element_by_name("q")
        elem.clear()
        elem.send_keys(landmark_name)
        elem.send_keys(Keys.RETURN)

        try:
            driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
            driver.implicitly_wait(10)
            images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd")
            count = 0
            for image in images:
                imgUrl = image.get_attribute("src")
                print(imgUrl)
                if imgUrl is not None:
                    count += 1
                    if not os.path.isdir("./image_collected/" + landmark_name):
                        os.makedirs("./image_collected/" + landmark_name)
                    try:
                        urllib.request.urlretrieve(imgUrl,
                                                   "image_collected/" + landmark_name + "/" + "{0:04}".format((count)) + ".jpg")
                        print('Save images : ', "images/" + landmark_name + "/" + "{0:04}".format((count)) + ".jpg")
                    except Exception as e:
                        print('Save Failed : ' + str(e))
                else:
                    pass
        except Exception as e:
            print('Error : ' + str(e))

        time.sleep(1)

driver.quit()

