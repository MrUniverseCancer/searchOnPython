import head
import numpy as np
from head import cv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def cv_show(img,name):
    cv.imshow(name,img) 
    cv.waitKey(0) 
    cv.destroyAllWindows()

def temp(img, a, b):
    v = cv.Canny(img, a, b)
    cv_show(v,'img')

def r_cal(x, y, center):
    fact_x = x - center[0]
    fact_y = y - center[1]
    return np.sqrt( fact_x**2 + fact_y**2 )

def theta_cal(x, y ,center):
    fact_x = x - center[0]
    fact_y = y - center[1]
    return np.arctan2(fact_y,fact_x)

def GetNew(sum,number):
    if sum == 10 

# def ani_Make():
#     # 创建一个大小为10x6的图形
#     fig, ax = plt.subplots(figsize=(10, 6))

#     # 初始化数据
#     data = np.random.rand(100, 100)

#     # 在图形中创建一个imshow对象
#     im = ax.imshow(data, cmap='gray')
    
#     # 定义一个更新函数，每次调用会更新数据集并更新imshow对象
#     def update(frame):
#         data = np.random.rand(100, 100)  # 更新数据集
#         im.set_array(data)  # 更新imshow对象的数据
#         return im,
    
#     # 创建一个动画对象，每隔50毫秒调用一次更新函数
#     ani = FuncAnimation(fig, update, frames=100, interval=50)

#     # 显示动画
#     plt.show()


if __name__ == '__main__':
    
    img=cv.imread('./test_figure.jpg',cv.IMREAD_GRAYSCALE)
    v1 = cv.Canny(img, 80, 150)
    # v2 = cv.Canny(img, 50, 100)
    # res = np.hstack((v1,v2))
    # cv_show(res,'res')
    res = v1
    x = res.shape[0]
    y = res.shape[1]
    fact = np.zeros((x,y))
    center = (x//2, y//2)
    i = 0
    j = 0
    ############################
    # Polar_r = []
    # Polar_theta = []
    # while i < x :
    #     while j < y:
    #         if v1[i][j] != 0:
    #             fact[i][j] = 1
    #             Polar_r.append(r_cal(i,j,center))
    #             Polar_theta.append(theta_cal(i,j,center))
    #         j += 1
    #     i += 1
    #     j = 0
    ########################
    (place_x, place_y) = np.nonzero(res)
    Polar_r = r_cal(place_x, place_y, center)
    Polar_theta = theta_cal(place_x, place_y, center)
    valid_length = place_x.size
    # ani_Make()


    # 创建一个大小为10x6的图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 初始化数据
    data = np.random.rand(x, y)
    
    # 在图形中创建一个imshow对象
    im = ax.imshow(data, cmap='gray')
    data = np.zeros((x,y)) + 1
    sum = 0 #Make begin check
    number = 0
    # fact = np.zeros((x,y)) + 1
    # fact[50:100,50:100] = 0
    # 定义一个更新函数，每次调用会更新数据集并更新imshow对象
    def update(frame):
        global data 
        global sum
        global number
        result = GetNew(sum, number) #添加的像素点
        if sum < 10 :
            data = np.zeros((x,y)) + 1
            sum += 1
        elif number < valid_length :
            data[place_x[number]][place_y[number]] = 0 #原始图像
            result *= data #最终结果
            number += 1
        im.set_array(result)  # 更新imshow对象的数据
        return im,

    # 创建一个动画对象，每隔50毫秒调用一次更新函数
    ani = FuncAnimation(fig, update, frames=1000, interval=1)

    # 显示动画
    plt.show()
    