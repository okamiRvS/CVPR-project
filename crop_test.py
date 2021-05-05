from matplotlib import pyplot as plt
import cv2

img = cv2.imread('WSC sample.png')
mask = cv2.imread('mask.png', 0)

res = cv2.bitwise_and(img, img, mask=mask)

plt.imshow(res)
plt.show()





                    


