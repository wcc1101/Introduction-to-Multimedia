import cv2
import matplotlib.pyplot as plt
import utils

def A(image):
    image_NN = utils.nearestNeighborInterpolation(image.copy())

    return image_NN

def B(image):
    image_BL = utils.biLinearInterpolation(image.copy())

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
