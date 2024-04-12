from face_utils import *
from face_reg import *


class Generate_fea():
    def __init__(self,pth):
        # self.root_pth = pth
        self.yolonet = yolov5()
        self.pth_list = self.get_img_pth(pth)

    def gen_fea(self):
        for i in self.pth_list:
            print(i)
            face_img = cv2.imread(i)
            print('face shape {}'.format(face_img.shape))
            locations = self.yolonet.detect(face_img)
            locations = postprocess(face_img.shape[0], face_img.shape[1], locations)
            print(locations)
            print('detect is done')
            face_encoding = face_recognition.face_encodings(face_img, locations)
            print(len(face_encoding))
            # face_encodings.append(face_recognition.face_encodings(face_img, locations))
            face_id = (os.path.split(i)[-1]).split('.')[0]
            # face_ids.append((os.path.split(i)[-1]).split('.')[0])
            return face_id,face_encoding
            # extract_features_to_milvus(collection, face_id, face_encoding)

    def get_img_pth(self,root_pth ):
        pth_list = list()
        for i in os.walk(root_pth):
            for k in i[-1]:
                if '.jpg' in k or '.jpeg' in k:
                    # if k.__contains__('.jpg','.jpeg'):
                    pth_list.append(os.path.join(i[0], k))
        return pth_list


# for i in pth_list:
#     print(i)
#     face_img = cv2.imread(i)
#     print('face shape {}'.format(face_img.shape))
#     locations = yolonet.detect(face_img)
#     locations = postprocess(face_img.shape[0],face_img.shape[1],locations)
#     print(locations)
#     print('detect is done')
#     face_encoding = face_recognition.face_encodings(face_img, locations)
#     print(len(face_encoding))
#     # face_encodings.append(face_recognition.face_encodings(face_img, locations))
#     face_id = (os.path.split(i)[-1]).split('.')[0]
#     # face_ids.append((os.path.split(i)[-1]).split('.')[0])
#     extract_features_to_milvus(collection,face_id,face_encoding)