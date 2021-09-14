from urllib import parse
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from pyproj import Proj, transform, Transformer
import googlemaps
import os, json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
secret_file = os.path.join(BASE_DIR, 'secrets.json')  # secrets.json 파일 위치를 명시

with open(secret_file) as f:
    secrets = json.loads(f.read())


def get_secret(setting, secrets=secrets):
    """비밀 변수를 가져오거나 명시적 예외를 반환한다."""
    try:
        return secrets[setting]
    except KeyError:
        error_msg = "Set the {} environment variable".format(setting)
        raise error_msg


def coordinate_change(x, y, before_type, after_type):
    before_type = int(before_type.split(":")[1])
    after_type = int(after_type.split(":")[1])

    transformer = Transformer.from_crs(before_type, after_type, always_xy=True)

    x_, y_ = transformer.transform(x, y)
    # proj_1 = Proj(init=before_type)
    # proj_2 = Proj(init=after_type)
    #
    # x_, y_ = transform(proj_1, proj_2, x, y)

    return x_, y_


def geocoding_naver(address):
    client_id = get_secret("NAVER_GEOCODING_API_Client_ID")
    client_pw = get_secret("NAVER_GEOCODING_API_Client_PWD")

    api_url = 'https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode?query='

    add_urlenc = parse.quote(address)

    url = api_url + add_urlenc
    request = Request(url)
    request.add_header('X-NCP-APIGW-API-KEY-ID', client_id)
    request.add_header('X-NCP-APIGW-API-KEY', client_pw)
    try:
        response = urlopen(request)
    except HTTPError as e:
        print('HTTP Error')
        lat = None
        lng = None
    else:
        rescode = response.getcode()
        if rescode == 200:
            response_body = response.read().decode('utf-8')
            response_body = json.loads(response_body)
            if not response_body['addresses']:
                print('result not exist')
                lat = None
                lng = None
            else:
                lat = response_body['addresses'][0]['y']
                lng = response_body['addresses'][0]['x']
        else:
            print('Response error code : %d' % rescode)
            lat = None
            lng = None

    return lat,lng


def geocoding_google(address):
    API_KEYS = get_secret("GEOCODING_API_KEY")
    gmaps = googlemaps.Client(key=API_KEYS)
    try:
        geocode_result = gmaps.geocode(address, language='ko')
        lat = geocode_result[0]['geometry']['location']['lat']
        lng = geocode_result[0]['geometry']['location']['lng']
    except Exception:
        lat = ''
        lng = ''
    finally:
        return lat, lng


def reverse_geocoding_google(lat, lng):
    API_KEYS = get_secret("GEOCODING_API_KEY")
    gmaps = googlemaps.Client(key=API_KEYS)
    try:
        reverse_geocode_result = gmaps.reverse_geocode((lat, lng), language='ko')
        address = reverse_geocode_result[0]['formatted_address']
    except Exception:
        address = ''
    finally:
        return address


def csv_splitter(filedir, filename):
    rowsize = sum(1 for row in open(filedir + filename, encoding='UTF-8'))
    newsize = 4000
    times = 0
    for i in range(1, rowsize, newsize):
        times += 1
        df = pd.read_csv(filedir + filename, header=None, nrows=newsize, skiprows=i)
        csv_output = filename[:-4] + '_' + str(times) + '.csv'
        df.to_csv(filedir + csv_output, index=False, header=False, mode='a', chunksize=rowsize)

