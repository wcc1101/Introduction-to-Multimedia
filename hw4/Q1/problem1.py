import numpy as np
import cv2

import numpy as np
import matplotlib.pyplot as plt

##for(a)
def subplot(points, result1, result2, img):
    plt.imshow(img)
    plt.scatter(points[:, 0], points[:, 1],  s=2)
    plt.plot(result1[:, 0], result1[:, 1], 'b-',linewidth=0.5)
    plt.plot(result2[:, 0], result2[:, 1], 'r-',linewidth=0.5)
    plt.savefig('output/1a.png')
    plt.close()

##for(b)
def plot(points , result , img):
    plt.imshow(img)
    plt.scatter(points[:, 0], points[:, 1],  s=5)
    plt.plot(result[:, 0], result[:, 1], 'r-' ,linewidth=0.5)
    plt.savefig('output/1b.png')
    plt.close()

def polynomial(points, t):
    return ((1-t)**3)*points[0]+3*t*((1-t)**2)*points[1]+3*(t**2)*(1-t)*points[2]+(t**3)*points[3]

def bezier_curve(points, T):
    result = []
    for i in range(13):
        for t in T:
            result.append(polynomial(points[3*i:3*(i+1)+1], t))
    return np.array(result)

def nearestNeighborInterpolation(image):
    # define height and width for the image
    h, w = image.shape[:2]

    # initialize new image
    image_NN = np.zeros(shape = (4 * h, 4 * w, 3), dtype='uint8')

    # calculate each pixel in new image
    for i in range(4 * h):
        for j in range(4 * w):
            image_NN[i][j] = image[int(i / 4)][int(j / 4)]

    return image_NN

def main():      
    # Load the image and points
    img = cv2.imread("./bg.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    points = np.loadtxt("./points.txt")

    ## 1.a
    t1 = np.arange(0.0, 1.1, 0.5)
    t2 = np.arange(0.0, 1.01, 0.01)

    result1 = bezier_curve(points, t1)
    result2 = bezier_curve(points, t2)
    subplot(points, result1, result2, img)
    
    # 2.a 
    img4 = nearestNeighborInterpolation(img)
    points4 = points * 4
    
    result = bezier_curve(points4, t2)
    plot(points4, result, img4)

if __name__ == "__main__":
    main()