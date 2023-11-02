import cv2
# def dilate_image(image):
#     kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
#     dst = cv2.dilate(image, kernal)
#     return dst

a = cv2.imread('/home/dell/下载/3090_deocclusion/deocclusion-master/data/Grap/putao/test_for_predict/5126/out_image_1.png',0)
#b = cv2.imread('/home/dell/下载/3090_deocclusion/deocclusion-master/data/Grap/putao/test_for_predict/5126/1_output_mask.png',0)

# c = b[3:855,1536:]
# d = a[3:855,1536:]
#
# c = dilate_image(c)
# d = dilate_image(d)
#
# cv2.imwrite('./occ.png',c)
# cv2.imwrite('./occder.png',d)

# cv2.imshow('1', c+d)
# cv2.waitKey(0)
import numpy as np
blanck_image = np.zeros((1080,1920),np.uint8)
blanck_image[3:855,1536:]=a
cv2.imwrite('./blank_image.png',blanck_image)