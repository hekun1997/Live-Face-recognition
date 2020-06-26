import cv2

if __name__ == '__main__':

    cameraCapture = cv2.VideoCapture(0)
    success, frame = cameraCapture.read()

    while success and cv2.waitKey(1) == -1:
        success, frame = cameraCapture.read()
        cv2.imshow("Camera", frame)
    cameraCapture.release()
    cv2.destroyAllWindows()