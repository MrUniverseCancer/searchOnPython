import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 读取图像文件
image = cv2.imread('./test_figure.jpg', 0)  # 以灰度模式读取

# 提取图像的边缘
edge_image = cv2.Canny(image, 100, 200)
edge_image = edge_image.astype(float)
# 创建图形和轴
fig, ax = plt.subplots()
im = ax.imshow(edge_image, cmap='gray', interpolation='nearest')

# 计算图像的傅里叶展开
f_transform = np.fft.fft2(edge_image)
f_shift = np.fft.fftshift(f_transform)

# 更新函数：逐步增加展开的频率并绘制
def update(frame):
    f_copy = f_shift.copy()
    rows, cols = edge_image.shape
    center_row, center_col = rows // 2, cols // 2

    # 将除去低频分量外的部分置零
    radius = frame
    f_copy[center_row-radius:center_row+radius, center_col-radius:center_col+radius] = 0

    # 傅里叶逆变换
    f_inverse = np.fft.ifftshift(f_copy)
    image_reconstructed = np.abs(np.fft.ifft2(f_inverse))

    im.set_data(image_reconstructed)
    return [im]

# 动画初始化函数
def init():
    im.set_data(edge_image)
    return [im]

# 创建动画
ani = FuncAnimation(fig, update, frames=1, init_func=init, blit=True)

# 显示动画
plt.show()