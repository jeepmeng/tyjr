from flask import Flask,make_response, request
from flask_cors import CORS,cross_origin
from flask import jsonify
import io
import cgi
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')



app = Flask(__name__)
CORS(app, resources=r'/*')


@app.route('/upload_img/', methods=['POST', "OPTIONS", "HEAD", "GET"])
# @app.route('/tabcco_ocr/', methods=['POST',"HEAD","GET"])
@cross_origin()
def upload_pic(**kwgs):
    # 来获取多个上传文件

    # multiple = request.args.get('num')
    # print("multiple------",multiple)

#     data = request.files('file')

    #    print("111")
    data = request.get_data()
    # print(data)
    # print(type(data))

    # content_type = request.content_type
    # print('content_type',content_type)

    content_type = request.content_type
    print('content_type',content_type)
    if content_type:
        boundary = content_type.split("=")[1].encode()  # 获取boundary字符串，并转换为bytes类型
        # print(boundary)
        fields = cgi.parse_multipart(io.BytesIO(data), {'boundary': boundary})
        # print(fields)

        # *********************************
        img_io = io.BytesIO(fields['file'][0])
        img_PIL = Image.open(img_io).convert('RGB')
        #         print(type(img_PIL))
        img_np = np.array(img_PIL)
        print(img_np.shape)
        # print(img_np)
        # cv2.imshow('hh',img_np)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    respose = {
        # "status": jiancezhi,
        "code": 200,
        # "urls": img_url,
        # "tongjijieguo": haoyunlai,
        # 'num': multiple

    }
    return jsonify(respose)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8089)