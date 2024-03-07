import os
import PIL

# pth = r'/Users/liufucong/Downloads/yolov5-7.0'


def get_img_pth(root_pth,):
    pth_list = list()
    for i in os.walk(root_pth):
        for k in i[-1]:
            if '.jpg' in k or '.jpeg' in k:
            # if k.__contains__('.jpg','.jpeg'):
                pth_list.append(os.path.join(i[0],k))
    return pth_list



if __name__ == "__main__":
    print(get_img_pth(r'/Users/liufucong/Downloads/公司人脸'))