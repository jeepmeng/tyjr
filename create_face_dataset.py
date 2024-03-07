import os
from face_reg import *
# from face_recognition import face_encodings as encode_face
# from face_recognition import face_distance as face_distance
from test_walk import get_img_pth
from pymilvus import connections, db
#
fmt = "\n=== {:30} ===\n"



# m_cont = connections.connect(
#   alias="default",
#   user="root",
#   password="xtjc@CC1234!",
#   host="172.19.16.103",
#   port="22",
#   uri="http://172.16.19.103:19530",
#
#   # token="root:xtjc@CC1234!",
#
# )
#
local_connect = connections.connect(
  alias="default",
  uri="localhost:19530",
  token="root:Milvus"
)
print(fmt.format("start connecting to Milvus"))
connections.connect("default", host="localhost", port="19530")




root_pth = r'/tyjr/avatar'
test_root_pth = r'/Users/liufucong/Downloads/公司人脸'
#获取所有图片路径
pth_list = get_img_pth(test_root_pth)
# print(pth_list)



#加载模型
yolonet = yolov5()


for i in pth_list:

    print(i[-28:])
    face_img = cv2.imread(i)
    locations = yolonet.detect(face_img)
    # top, right, bottom, left = yolonet.postprocess(face_img, dets)
    face_encodings = face_recognition.face_encodings(face_img, locations)
    face_id = (os.path.split(i)[-1]).split('.')[0]


