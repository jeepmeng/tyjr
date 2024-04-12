# from pymilvus import connections, db
from milvus_utils import *

# m_cont = connections.connect("default", host="localhost", port="19530")
m_cont = connections.connect(
  alias="default",
  user="root",
  password="xtjc@CC1234!",
  host="172.19.16.100",
  port="22",
  uri="http://172.16.19.100:19530",

  # token="root:xtjc@CC1234!",

)
from pymilvus import utility
has = utility.has_collection("hello_milvus")
print(f"Does collection hello_milvus exist in Milvus: {has}")
collection = connect_milvus()
collection.release()
utility.drop_collection("hello_milvus")

