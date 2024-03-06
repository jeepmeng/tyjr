import os

from pymilvus import connections, db

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
for img_file in os.listdir(root_pth):
    os.path.join(root_pth, img_file)
    for img_name in os.listdir(os.path.join(root_pth,img_file)):
        img_pth = os.path.join()

