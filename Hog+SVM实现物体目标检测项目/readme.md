## Hog + SVM实现物体目标检测
- 1、首先构建了正样本与负样本
    - 正样本采用的是视频获取图片的办法
    - process.py文件通过对video.mp4视频的处理，生成了正样本，并存入pos文件夹下。
    - ![image](./result/pos_pic.png)
    - 负样本表示不含目标物体的样本，存放在neg文件夹下
    - ![image](./result/neg_pic.png)

- 