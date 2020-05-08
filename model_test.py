from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2

if __name__ == "__main__":
    img_path = r'images\hekun\74.jpg'
    name_dict = np.load('labels.npy', allow_pickle='TRUE').item()
    model = load_model(r'C:\Users\administered\PycharmProjects\untitled1\vgg16\Model_VGGFace_vgg.h5')
    #img = image.load_img(r'E:\github\Face_Recognision\faces\hekun\0.jpg')
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, [1, 224, 224, 3])
    preds = model.predict(img)
    index = np.argmax(preds)
    print('识别结果为 -> ', name_dict[index])
    print('概率为 -> ', str(preds[0][index] * 100))