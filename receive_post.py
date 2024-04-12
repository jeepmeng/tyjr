from flask import Flask,make_response, request
from flask_cors import CORS,cross_origin
from flask import jsonify
import io
import cgi


app = Flask(__name__)


@app.route('/receive_id/', methods=['POST', "OPTIONS", "HEAD", "GET"])


def receive_ID(**kwgs):
    data = request.get_data()
    print(data)

    respose = {
        # "status": jiancezhi,
        "code": 200,
        # "urls": img_url,
        # "tongjijieguo": haoyunlai,
        # 'num': multiple

    }
    return jsonify(respose)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8118)