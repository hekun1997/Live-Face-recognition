from keras.models import load_model,Sequential
from keras.preprocessing import image
import numpy as np
import cv2

if __name__ == "__main__":
    img_path = r'C:\Users\administered\PycharmProjects\Live-Face-recognition\images\test\0.jpg'
    name_dict = np.load(r'labels.npy', allow_pickle='TRUE').item()
    model = load_model(r'data/model/Model_VGGFace.h5')
    #img = image.load_img(r'E:\github\Face_Recognision\faces\hekun\0.jpg')
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, [1, 224, 224, 3])

    base = Sequential()
    base.add(model)
    base.summary()
    result = base.predict_proba(img)
    print(result)
    # preds = model.predict(img)
    # index = np.argmax(preds)
    # print('识别结果为 -> ', name_dict[index])
    # print('概率为 -> ', str(preds[0][index] * 100))