from keras.models import load_model
import numpy as np
import cv2

MODEL_PATH = r'data/model/Model_VGGFace.h5'
MODEL_DATA = r'labels.npy'
FACE_CASCADE = r'data/haarcascade_frontalface_alt.xml'
IMAGE_SIZE = 224

class LiveRecongize(object):

    def __init__(self):
        self.model = load_model(MODEL_PATH)

    def recongize(self):
        name_dict = np.load(MODEL_DATA, allow_pickle='TRUE').item()
        classfier = cv2.CascadeClassifier(FACE_CASCADE)

        cameraCapture = cv2.VideoCapture(0)
        success, frame = cameraCapture.read()

        if not success:
            print('Error about camera occured!')

        while success and cv2.waitKey(1) == -1:
            success, frame = cameraCapture.read()
            faces = classfier.detectMultiScale(frame, 1.3, 5)  # 识别人脸
            for (x, y, w, h) in faces:
                face = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                face = cv2.resize(face, (224, 224))
                face = np.reshape(face, [1, 224, 224, 3])
                preds = self.model.predict(face)
                index = np.argmax(preds)
                color = (255, 0, 0)
                if float(preds[0][index]) > 0.8:  # 如果模型认为概率高于80%则显示为模型中已有的label
                    show_name = name_dict[index]
                else:
                    color = (0, 0, 0)
                    show_name = 'Stranger'
                cv2.putText(frame, show_name, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)  # 显示名字
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)  # 在人脸区域画一个正方形
            cv2.imshow("Camera", frame)

        cameraCapture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    camera = LiveRecongize()
    camera.recongize()
