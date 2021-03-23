import cv2

def get_default_video():
    ID = 0
    while (1):
        camera_capture = cv2.VideoCapture(ID)
        ret, frame = camera_capture.read()

        if not ret:
            ID += 1
        else:
            print(ID)
            break
    return ID

if __name__ == '__main__':
    Id = 0#get_default_video()
    cameraCapture = cv2.VideoCapture(Id)
    success, frame = cameraCapture.read()

    while success and cv2.waitKey(1) == -1:
        success, frame = cameraCapture.read()
        cv2.imshow("Camera", frame)
    cameraCapture.release()
    cv2.destroyAllWindows()