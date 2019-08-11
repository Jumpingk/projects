# 视频形状、尺寸的处理，生成正样本
# 步骤：
# 1、加载视频
# 2、读取视频的info信息
# 3、解码 parse 拿到单帧视频
# 4、imshow、imwrite
import cv2
import numpy as np
cap = cv2.VideoCapture('video.mp4')  # 获取一个视频
isOpened = cap.isOpened  # 判断是否打开文件
# 人眼最低的帧分辨率是15帧
fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率(一秒钟显示多少张图片)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 宽度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 高度
print('fps:', fps, ' width:', width, ' height:', height)
i = 0
while isOpened:
    if i == 821:
        break
    else:
        i += 1
     # 读取每一张 flag表明是否读取成功(True/False)
    flag, frame = cap.read()
    img1 = cv2.transpose(frame, (width, height, 3))
    img1 = cv2.resize(img1, (64, 128), interpolation=cv2.INTER_AREA)
    fileName = './pos/' + str(i) + '.jpg'
    print(fileName)
    if flag == True:
        cv2.imwrite(fileName, img1, [cv2.IMWRITE_JPEG_QUALITY, 100])
    else:
        break
print('end!')
# cv2.waitKey(0)
# cv2.destroyAllWindows()

