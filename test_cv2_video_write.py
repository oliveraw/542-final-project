import cv2
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter("test.mp4", fourcc, 30, (30, 60))

for i in range(100):
    frame = (np.random.rand(60, 30, 3) * 255).astype(np.uint8)
    video.write(frame)

video.release()
cv2.destroyAllWindows()