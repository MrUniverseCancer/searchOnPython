# # print("Hello World")
# # def temp(i):
# #     print(i)

# # temp(100)
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# import matplotlib.animation as animation
# import image

# _image = cv2.imread("./test_figure.jpg", cv2.IMREAD_GRAYSCALE)
# contours, _ = cv2.findContours(_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # image.cv_show(_image, "tmep")

# # 提取第一个轮廓
# contour = contours[0][:, 0, :]

# # 对轮廓进行傅立叶变换
# contour_complex = np.empty(contour.shape[:-1], dtype=complex)
# contour_complex.real = contour[:, 0]
# contour_complex.imag = contour[:, 1]

# # 进行傅立叶变换
# contour_fft = np.fft.fft(contour_complex)


# # 设置动画参数
# num_frames = 100  # 动画帧数
# max_radius = 200  # 最大半径
# angles = np.linspace(0, 2 * np.pi, num_frames)  # 在0到2π之间均匀分布的角度

# # 生成圆周运动轨迹
# trajectory = np.zeros((num_frames, len(contour), 2), dtype=float)
# for i in range(num_frames):
#     for j in range(len(contour)):
#         radius = np.abs(contour_fft[j])  # 使用傅里叶变换结果确定半径
#         phase = np.angle(contour_fft[j])  # 使用傅里叶变换结果确定相位
#         x = contour[j, 0] + radius * np.cos(phase + angles[i])  # x 坐标
#         y = contour[j, 1] + radius * np.sin(phase + angles[i])  # y 坐标
#         trajectory[i, j] = [x, y]

# # 绘制轨迹动画
# fig, ax = plt.subplots()
# ax.set_xlim(0, _image.shape[1])
# ax.set_ylim(0, _image.shape[0])

# line, = ax.plot([], [], 'r-')

# def init():
#     line.set_data([], [])
#     return line,

# def update(frame):
#     line.set_data(trajectory[frame, :, 0], trajectory[frame, :, 1])
#     return line,

# ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True)
# plt.show()

# str_temp = "Hello"
# change = ord('A') - ord('a')
# for ichar in str_temp:
#     if ichar > 'A' and ichar < 'Z':
#         ichar = chr(ord(ichar) - change)
# else:
#     print(str_temp)  

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # 创建一个空图形
# fig, ax = plt.subplots(figsize=(10, 6))

# # 在图形中创建一个空曲线
# line, = ax.plot([], [])
# ax.set_ylim(-1.2, 1.2)
# ax.set_xlim(-1,10)
# # 定义一个更新函数，每次调用会更新曲线的数据
# def update(frame):
#     x = np.linspace(0, 0.5*np.pi, 50) 
#     temp = np.linspace(1.5*np.pi, 2*np.pi,50)
#     x = np.hstack((x,temp))
#     y = np.sin(x + frame/10.0)
#     line.set_data(x, y)
#     return line,

# # 创建一个动画对象，每隔50毫秒调用一次更新函数
# ani = FuncAnimation(fig, update, interval=50)

# # 显示动画
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 创建一个大小为10x6的图形
fig, ax = plt.subplots(figsize=(10, 6))

# 初始化数据
data = np.random.rand(100, 100)

# 在图形中创建一个imshow对象
im = ax.imshow(data, cmap='gray')

# 定义一个更新函数，每次调用会更新数据集并更新imshow对象
def update(frame):
    # data = np.random.rand(100, 100)  # 更新数据集
    data = np.zeros((100,100)) + 255 
    im.set_array(data)  # 更新imshow对象的数据
    return im,

# 创建一个动画对象，每隔50毫秒调用一次更新函数
ani = FuncAnimation(fig, update, frames=100, interval=50)

# 显示动画
plt.show()