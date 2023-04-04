from PIL import Image
from retinaface import RetinaFace
import cv2
import time
import numpy as np
from process_file import fer

if __name__ == "__main__":
    capture = cv2.VideoCapture(0)  # Open camera
    ref, frame = capture.read()
    if not ref:
        raise ValueError("Cannot open the camera.")

    fps = 0.0
    while True:
        t1 = time.time()
        ref, frame = capture.read()
        if not ref:
            break
        image = frame
        img_size = np.asarray(image.shape)[0:2]

        results = RetinaFace.detect_faces(frame)
        if type(results) == dict:
            for key in results:
                identity = results[key]
                facial_area = identity["facial_area"]
                if facial_area[3] - facial_area[1] <= 10 or facial_area[2] - facial_area[0] <= 10:
                    continue
                else:
                    facial_img = frame[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]
                    w = facial_area[2] - facial_area[0]
                    h = facial_area[3] - facial_area[1]

                    fps = (fps + (1. / (time.time() - t1))) / 2
                    label = fer(Image.fromarray(facial_img))
                    image = cv2.rectangle(image, (facial_area[0], facial_area[1]),
                                          (facial_area[0] + w, facial_area[1] + h), (36, 255, 12), 1)
                    cv2.putText(image, label, (facial_area[0], facial_area[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (36, 255, 12), 2)

        cv2.imshow("video", image)
        c = cv2.waitKey(1) & 0xff

        if c == 27:
            capture.release()
            break

    print("Done!")
    capture.release()
    cv2.destroyAllWindows()
