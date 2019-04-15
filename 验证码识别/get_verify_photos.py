from urllib import request
import time
import random


def get_and_save_verify(i):
    try:
        url = 'http://jwxt.qlu.edu.cn/verifycode.servlet'
        request.urlretrieve(url, './verify_pictures/' + 'verify_' + str(i) + '.png')
        print('第' + str(i) + '张图片下载成功')
    except Exception:
        print('第' + str(i) + '张图片下载失败')


def get_proxy():
    # 使用代理步骤
    # - 1、设置代理地址
    proxys = [{'http': '39.137.69.10:8080'},
              {'http': '111.206.6.101:80'},
              {'http': '120.210.219.101:8080'},
              {'http': '111.206.6.101:80'},
              {'https': '120.237.156.43:8088'}]
    # - 2、创建ProxyHandler
    proxy = random.choice(proxys)
    proxy_handler = request.ProxyHandler(proxy)
    # - 3、创建Opener
    opener = request.build_opener(proxy_handler)
    # - 4、导入Opener
    request.install_opener(opener)


if __name__ == '__main__':
    for i in range(1, 101):
        get_proxy()
        time.sleep(random.randint(1, 4))
        get_and_save_verify(i)