import base64
import requests
import json


def read_img_base64(path):
    with open(path, 'rb') as f:
        imgString = base64.b64encode(f.read())
    imgString = imgString.decode('utf8')
    return imgString


def post(img):
    URL = 'http://35.247.91.200:8080/simpleocr'
    imgString = read_img_base64(img)
    param = {'img': imgString}
    res = requests.post(URL, data=param)
    # 如果服务器没有报错，传回json格式数据
    print(eval(res.text))

if __name__ == '__main__':
    img = 'demo/fig1.bmp'
    post(img)