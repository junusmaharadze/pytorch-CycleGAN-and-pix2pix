
import cv2


#img.save('/test_latest/images/04002_fake_B.png')
img = cv2.imread('./mnist_4_channel_bs64_200dec/test_latest/images/00088_fake_B.png', cv2.IMREAD_UNCHANGED)
print(img.shape)
