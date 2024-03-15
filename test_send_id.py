import requests
import json

data = {'ID': '18584322325'}
data = json.dumps(data)

url = 'http://10.100.1.208:8080/xfxt/access/setIdCardNo'
# data = 'python'

r = requests.post(url, data=data)  # 使用data携带参数
print(r.text)