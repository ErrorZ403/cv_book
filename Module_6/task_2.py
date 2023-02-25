import cv2
import numpy as np

image_path_1 = 'Module_6/car1.jpg'
image_path_2 = 'Module_6/car2.jpg'

image_1 = cv2.imread(image_path_1)
image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
image_2 = cv2.imread(image_path_2)
image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

flow = cv2.calcOpticalFlowFarneback(image_1, image_2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

cv2.imshow()