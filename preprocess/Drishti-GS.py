import scipy.io
import matplotlib.pyplot as plt
import numpy as np

# 读取MATLAB的.mat文件
image_data = scipy.io.loadmat('F:/DL-Data/eyes/glaucoma_OD_OC/ORIGA/650mask/AGLAIA_GT_001.mat')



print(image_data)
# 获取图像数据
image = image_data['maskFull']
print(np.unique(image))
# # #
# # # # 显示图像
# plt.imshow(image, cmap='gray')
# plt.axis('off')
# plt.show()
