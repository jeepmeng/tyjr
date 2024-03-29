from test_walk import *
from milvus_utils import *
# from pymilvus import list_collections


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
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128},
}
collection.create_index("face_embeddings", index)
# print('获得所有的集合:', list_collections(using='main'))
collection.load()

# result = collection.query(...)
# res = collection.query(
#   expr ='face_id in [*]',
#   # offset = 0,
#   # limit = 10,
#   # output_fields = ["face_embeddings",'face_id'],
# output_fields = ['*'],
# )
#
# for i in res:
#     print(i)



res = collection.query(
  expr="",
    limit = 100,
  output_fields = ['*'],
)
for i in res:
    print(i)
# print(res[0])


test_root_pth = r'/Users/liufucong/Downloads/公司人脸'
#获取所有图片路径
pth_list = get_img_pth(test_root_pth)
print(len(pth_list))
for i in pth_list:
    print(i)