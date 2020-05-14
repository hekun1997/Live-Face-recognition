import cv2

CASCADE_PATH = r'data/haarcascade_frontalface_alt.xml'
IMAGES_PATH = r'images/hekun'

def CatchPICFromVideo(window_name, camera_idx, catch_pic_num, path_name):
    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(camera_idx)
    classfier = cv2.CascadeClassifier(CASCADE_PATH)

    color = (0, 255, 0)
    num = 0
    while cap.isOpened():
        ok, frame = cap.read()  # 读取一帧数据
        if not ok:
            break

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=5, minSize=(32, 32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect
                img_name = '%s/%d.jpg' % (path_name, num)
                image = cv2.resize(frame[y - 10: y + h + 10, x - 10: x + w + 10], (256, 256))
                cv2.imwrite(img_name, image)

                num += 1
                if num > (catch_pic_num):
                    break
                # 画出矩形框
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'num:%d' % (num), (x + 30, y + 30), font, 1, (255, 0, 255), 4)

        if num > (catch_pic_num): break

        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
     CatchPICFromVideo("pick face", 1, 1000, IMAGES_PATH)
