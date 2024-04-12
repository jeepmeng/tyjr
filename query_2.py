from test_walk import *
from milvus_utils import *
from pymilvus import connections, db,list_collections
# from pymilvus import list_collections
from creat_feature import *


m_cont = connections.connect(
  alias="default",
  user="root",
  password="xtjc@CC1234!",
  host="172.19.16.100",
  port="22",
  uri="http://172.16.19.100:19530",
db_name="tyjr"
)

# database = db.create_database("tyjr")
# db.using_database("tyjr")
# db.using_database("fucongliu")
print(db.list_database())

collection = connect_milvus()


# print(db.list_database())
# has = utility.has_collection("face_collection")
# if has:
#     print('Milvus has face_collection:{}'.format(has))
#
#
crt_fea = Generate_fea(r'/Users/liufucong/Downloads/tyjr_test')
face_id,face_encoding = crt_fea.gen_fea()
extract_features_to_milvus(collection, face_id, face_encoding)