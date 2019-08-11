import pytesseract
import cv2
import os
import numpy as np
import re

path = './verify_pictures/'

file_name = []
for k in os.walk(path):
    file_name = k[-1]

print('识别值' + '-----' + '真实值')
num = 0
for i in file_name:
    img = cv2.imdecode(np.fromfile(path + i, dtype=np.uint8), 1)

    # 对数据的处理
    # blur = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯滤波函数
    # blur = cv2.medianBlur(img, 3)  # 中值滤波函数
    blur = cv2.bilateralFilter(img, 3, 560, 560)  # 双边滤波函数 560：0.28

    a = pytesseract.image_to_string(blur)

    # 对结果的处理
    st = re.sub(r'[^A-Za-z0-9]+', '', a)
    st = st.lower()
    if len(st) > 4:
        b = st[-4:]
    else:
        b = st

    true_value = i[-8:-4]
    print(b + '-----' + true_value)
    if b == true_value:
        num += 1

print('识别的准确率为：' + str(num/100))
