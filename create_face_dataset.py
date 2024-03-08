import os
import argparse
from face_utils import *
from face_reg import *
from milvus_utils import *
# from face_recognition import face_encodings as encode_face
# from face_recognition import face_distance as face_distance
from test_walk import get_img_pth
# from pymilvus import connections, db

# parser = argparse.ArgumentParser()
# parser.add_argument('--yolo_type', type=str, default='yolov5l', choices=['yolov5s', 'yolov5m', 'yolov5l'], help="yolo type")
# parser.add_argument("--root_pth", type=str, default=r'/tyjr/avatar', help="image root path")
# parser.add_argument('--confThreshold', default=0.3, type=float, help='class confidence')
# parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
# parser.add_argument('--objThreshold', default=0.2, type=float, help='object confidence')
# parser.add_argument('--dim', default=128, type=int, help='face feature dimension')
# args = parser.parse_args()

dim = 128


m_cont = connections.connect(
  alias="default",
  user="root",
  password="xtjc@CC1234!",
  host="172.19.16.103",
  port="22",
  uri="http://172.16.19.103:19530",

  # token="root:xtjc@CC1234!",

)

root_pth = r'/tyjr/avatar'
test_root_pth = r'/Users/liufucong/Downloads/公司人脸'
#获取所有图片路径
pth_list = get_img_pth(test_root_pth)
print(pth_list)
#加载模型
yolonet = yolov5()
# yolonet = yolov5(args.yolo_type, confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold, objThreshold=args.objThreshold)
face_encodings = []
face_ids = []

collection = connect_milvus()
print("Create collection face_collection is DONE")
for i in pth_list:
    print(i)
    face_img = cv2.imread(i)
    print('face shape {}'.format(face_img.shape))
    locations = yolonet.detect(face_img)
    locations = postprocess(face_img.shape[0],face_img.shape[1],locations)
    print(locations)
    print('detect is done')
    face_encoding = face_recognition.face_encodings(face_img, locations)
    print(len(face_encoding))
    # face_encodings.append(face_recognition.face_encodings(face_img, locations))
    face_id = (os.path.split(i)[-1]).split('.')[0]
    # face_ids.append((os.path.split(i)[-1]).split('.')[0])
    extract_features_to_milvus(collection,face_id,face_encoding)











# print(fmt.format("Start inserting entities"))
#
#
# entities = [
#     # face_id集合
#     face_ids,
#     face_encodings,    # face_encodings集合
# ]
#
#
# insert_result = face_collection.insert(entities)
#
#
# face_collection.flush()
# print(f"Number of entities in Milvus: {face_collection.num_entities}")  # check the num_entities
