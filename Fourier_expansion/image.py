import head
from head import cv


def cv_show(img,name):
    cv.imshow(name,img) 
    cv.waitKey(0) 
    cv.destroyAllWindows()

def temp(img, a, b):
    v = cv.Canny(img, a, b)
    cv_show(v,'img')

if __name__ == '__main__':
    
    img=cv.imread('./test_figure.jpg',cv.IMREAD_GRAYSCALE)
    v1 = cv.Canny(img, 80, 150)
    # v2 = cv.Canny(img, 50, 100)
    # res = head.np.hstack((v1,v2))
    # cv_show(res,'res')
    res = v1
    x = res.shape[0]
    y = res.shape[1]
    fact = head.np.zeros((x,y))
    center = (x//2, y//2)
    i = 0
    j = 0
    while i < x :
        while j < y:
            if v1[i][j] != 0:
                fact[i][j] = 1
                print("Hihi")
            j += 1
        i += 1
    
    Polar_r = head.np.sqrt((-center[0])**2 + (-center[1])**2)
    Polar_theta = head.np.arctan2(y - center[1], x - center[0])
    