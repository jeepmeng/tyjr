from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from face_reg import *
from face_utils import *
import face_recognition
import cv2
import time

yolonet = yolov5()
face_img = cv2.imread(r'/Users/liufucong/Downloads/公司人脸/220104198807260013.jpg')
locations = yolonet.detect(face_img)
locations = postprocess(face_img.shape[0],face_img.shape[1],locations)
print(locations)
print('detect is done')
face_encoding = face_recognition.face_encodings(face_img, locations)



st = time.time()
m_cont = connections.connect(
  alias="default",
  user="root",
  password="xtjc@CC1234!",
  host="172.19.16.103",
  port="22",
  uri="http://172.16.19.103:19530"
)
has = utility.has_collection("face_collection")
if has:
    print('Milvus has face_collection:{}'.format(has))
collection = Collection("face_collection")      # 获取一个已存在的集合。
index = {
    # "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "offset": 0,
    # "params": {"nlist": 128},
}
# collection.create_index("face_embeddings", index)
# print('获得所有的集合:', list_collections(using='main'))
collection.load()

result = collection.search(face_encoding, "face_embeddings", index, limit=3, output_fields=["*"])
end = time.time()
print(result[0].ids)
print(result[0].distances)
hit = result[0][1]
print(hit)
print(hit.entity.get('face_id'))

print(end-st)
# for i in result:
#     print(i)