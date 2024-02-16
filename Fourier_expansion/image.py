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
    v2 = cv.Canny(img, 50, 100)
    res = head.np.hstack((v1,v2))
    cv_show(res,'res')
