# print("Hello World")
# def temp(i):
#     print(i)

# temp(100)
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import image

_image = cv2.imread("./test_figure.jpg", cv2.IMREAD_GRAYSCALE)
contours, _ = cv2.findContours(_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# image.cv_show(_image, "tmep")

# 提取第一个轮廓
contour = contours[0][:, 0, :]

# 对轮廓进行傅立叶变换
contour_complex = np.empty(contour.shape[:-1], dtype=complex)
contour_complex.real = contour[:, 0]
contour_complex.imag = contour[:, 1]

# 进行傅立叶变换
contour_fft = np.fft.fft(contour_complex)


# 设置动画参数
num_frames = 100  # 动画帧数
max_radius = 200  # 最大半径
angles = np.linspace(0, 2 * np.pi, num_frames)  # 在0到2π之间均匀分布的角度

# 生成圆周运动轨迹
trajectory = np.zeros((num_frames, len(contour), 2), dtype=float)
for i in range(num_frames):
    for j in range(len(contour)):
        radius = np.abs(contour_fft[j])  # 使用傅里叶变换结果确定半径
        phase = np.angle(contour_fft[j])  # 使用傅里叶变换结果确定相位
        x = contour[j, 0] + radius * np.cos(phase + angles[i])  # x 坐标
        y = contour[j, 1] + radius * np.sin(phase + angles[i])  # y 坐标
        trajectory[i, j] = [x, y]

# 绘制轨迹动画
fig, ax = plt.subplots()
ax.set_xlim(0, _image.shape[1])
ax.set_ylim(0, _image.shape[0])

line, = ax.plot([], [], 'r-')

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data(trajectory[frame, :, 0], trajectory[frame, :, 1])
    return line,

ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True)
plt.show()