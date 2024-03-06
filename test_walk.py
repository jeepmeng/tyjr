import os

pth = r'/Users/liufucong/Downloads/yolov5-7.0'
dir = os.walk(pth)
for i in dir:
    print(i[-1])