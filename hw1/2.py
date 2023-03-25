import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def saveImage(image, fileName):
    outDir = 'out'
    try:
        os.mkdir(outDir)
    except:
        pass
    cv2.imwrite(os.path.join(outDir, fileName), image)

def nearestNeighborInterpolation(image):
    # define height and width for the image
    h, w = image.shape[:2]

    # initialize new image
    image_NN = np.zeros(shape = (4 * h, 4 * h, 3), dtype='uint8')

    # calculate each pixel in new image
    for i in range(4 * h):
        for j in range(4 * w):
            image_NN[i][j] = image[int(i / 4)][int(j / 4)]

    # save image
    saveImage(image_NN, 'bee_near.jpg')

    return image_NN

def biLinearInterpolation(image):
    # define height and width for the image
    h, w = image.shape[:2]

    # initialize new image
    image_BL = np.zeros(shape=(4 * h, 4 * h, 3))

    # calculate each pixel in new image
    for i in range(4 * h):
        for j in range(4 * w):
            # find position in original image
            x = (i + 1) / 4 - 1
            y = (j + 1) / 4 - 1

            # find referencing point
            xIndex = int(x)
            yIndex = int(y)

            # calculate bias
            u = x - xIndex
            v = y - yIndex

            # calculate by point * area
            if xIndex >= 0:
                if yIndex >= 0:
                    image_BL[i][j] += (1 - u) * (1 - v) * image[xIndex][yIndex]
                    if xIndex + 1 < h:
                        image_BL[i][j] += u * (1 - v) * image[xIndex + 1][yIndex]
                if yIndex + 1 < w:
                    image_BL[i][j] += (1 - u) * v * image[xIndex][yIndex + 1]
                    if xIndex + 1 < h:
                        image_BL[i][j] += u * v * image[xIndex + 1][yIndex + 1]

            # fix value
            for c in range(3):
                if image_BL[i][j][c] < 0:
                    image_BL[i][j][c] = 0

    image_BL = image_BL.astype('uint8')

    # save image
    saveImage(image_BL, 'bee_linear.jpg')

    return image_BL

def A(image):
    image_NN = nearestNeighborInterpolation(image.copy())

    return image_NN

def B(image):
    image_BL = biLinearInterpolation(image.copy())

    return image_BL

if __name__ == '__main__':
    # open input image
    imageFileName = 'img/bee.jpg'
    image = cv2.imread(imageFileName)

    # part A
    image_NN = A(image.copy())

    # part B
    image_BL = B(image.copy())

    # print size
    print(f'Size of original image: ({image.shape[0]}, {image.shape[1]})')
    print(f'Size of image with NN interpolation: ({image_NN.shape[0]}, {image_NN.shape[1]})')
    print(f'Size of image with BL interpolation: ({image_BL.shape[0]}, {image_BL.shape[1]})')

    # plot
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original')
    ax[1].imshow(cv2.cvtColor(image_NN, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Nearest Neighbor')
    ax[2].imshow(cv2.cvtColor(image_BL, cv2.COLOR_BGR2RGB))
    ax[2].set_title('Bilinear')
    plt.show()
