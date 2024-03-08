from milvus_utils import *



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
# collection = Collection("face_collection")
    utility.drop_collection("face_collection")
    print('Drop collection "face_collection"')