from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
import requests
import json
from face_reg import *
from face_utils import *
import face_recognition
import cv2
from test_walk import *
import time
from log_utils import *


class m_serrch():
    def __init__(self):
        # self.logger = setup_log("tyjr_log")
        self.yolonet = yolov5()
        self.m_cont = connections.connect(
            alias="default",
            user="root",
            password="xtjc@CC1234!",
            host="172.19.16.103",
            port="22",
            uri="http://172.16.19.103:19530"
        )
        # has = utility.has_collection("face_collection")
        # if has:
        #     print('Milvus has face_collection:{}'.format(has))
        self.collection = Collection("face_collection")  # 获取一个已存在的集合。
        self.index = {
            # "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "offset": 0,
            # "params": {"nlist": 128},
        }
        # collection.create_index("face_embeddings", index)
        # print('获得所有的集合:', list_collections(using='main'))
        self.collection.load()
    def face_search(self,face_img):

        # face_img = cv2.imread(r'/Users/liufucong/Downloads/公司人脸/220104198807260013.jpg')
        if any(face_img):

            locations = self.yolonet.detect(face_img)
            locations = postprocess(face_img.shape[0], face_img.shape[1], locations)
            # print(locations)
            # print('detect is done')
            face_encoding = face_recognition.face_encodings(face_img, locations)
            result = self.collection.search(face_encoding, "face_embeddings", self.index, limit=3, output_fields=["*"])
            # p_format = ("Person_if:{0} \n" + space + "Img_url:{1}").format('ID', 'URL')
            # logger.info('')

            return result[0].ids[0],result[0].distances[0]
        else:
            print('Img array is None!')


def img_post(person_id,img_url,url = ''):
    # url = ''
    data = {'person_id':person_id,'img_url':img_url}
    requests.post(url,data)


def upload(token,img):
    url = 'http://116.142.242.187:10444/prod-api/file/upload'


    # payload={}
    files=[
       ('file',('image',img,'image/jpeg'))
    ]
    headers = {
       'User-Agent': 'apifox/1.0.0 (https://www.apifox.cn)',
       'Authorization': token
    }

    response = requests.request("POST", url, headers=headers, files=files)

#     print(response.text)
#     print("upload",response.json())
    return(response.json()['data']["url"])
#


def key_token():
    url = "http://116.142.242.187:10444/prod-api/auth/appLogin"

    payload = json.dumps({
        "password": "123456",
        "username": "1049"
    })
    headers = {
            'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
#     print(response.json())
#     print('token-------------',response.json()["data"]['access_token'])
    return response.json()["data"]['access_token']
    # print(type(response))
    # print(response.json()["token"])
# st = time.time()
# m_cont = connections.connect(
#   alias="default",
#   user="root",
#   password="xtjc@CC1234!",
#   host="172.19.16.103",
#   port="22",
#   uri="http://172.16.19.103:19530"
# )
# has = utility.has_collection("face_collection")
# if has:
#     print('Milvus has face_collection:{}'.format(has))
# collection = Collection("face_collection")      # 获取一个已存在的集合。
# index = {
#     # "index_type": "IVF_FLAT",
#     "metric_type": "L2",
#     "offset": 0,
#     # "params": {"nlist": 128},
# }
# # collection.create_index("face_embeddings", index)
# # print('获得所有的集合:', list_collections(using='main'))
# collection.load()
#
# result = collection.search(face_encoding, "face_embeddings", index, limit=3, output_fields=["*"])
# end = time.time()
# print(result[0].ids)
# print(result[0].distances)
# hit = result[0][1]
# print(hit)
# print(hit.entity.get('face_id'))
#
# print(end-st)
# # for i in result:
# #     print(i)
#
#
#
# yolonet = yolov5()
# face_img = cv2.imread(r'/Users/liufucong/Downloads/公司人脸/220104198807260013.jpg')
# locations = yolonet.detect(face_img)
# locations = postprocess(face_img.shape[0],face_img.shape[1],locations)
# print(locations)
# print('detect is done')
# face_encoding = face_recognition.face_encodings(face_img, locations)

if __name__ == "__main__":

    test_root_pth = r'/Users/liufucong/Downloads/公司人脸'
    # 获取所有图片路径
    pth_list = get_img_pth(test_root_pth)
    new_search = m_serrch()
    # print(len(pth_list))
    for i in pth_list:
        print(i)
        st = time.time()
        face_img = cv2.imread(i)
        st_2 = time.time()
        id,dis = new_search.face_search(face_img)
        ed = time.time()
        print('---ID is {}, ---Confidence is {}, ----Load face image time cost is {},----Search time cost is {}'.format(id,dis,st_2-st,ed-st_2))
    # for i in result:
    #     print(i)
    # img_2 = cv2.imread(r'/Users/liufucong/Downloads/公司人脸/sunfeiyue.jpeg')
    # result = new_search.face_search(img_2)
    # for i in result:
    #     print(i)