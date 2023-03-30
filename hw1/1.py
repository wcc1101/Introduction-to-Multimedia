import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils

### A part ###
def A(image):
    # median cut
    image3 = utils.medianCut(image.copy(), 3)
    image6 = utils.medianCut(image.copy(), 6)

    # calculate MSE
    print(f'MSE of quantization with n=3: {np.mean((image3 - image) ** 2)}')
    print(f'MSE of quantization with n=6: {np.mean((image6 - image) ** 2)}')

    return image3, image6

### B part ###
def B(image, colorMap3, colorMap6):
    # error diffusion dithering
    image3 = utils.errorDiffusionDithering(image.copy(), colorMap3, 3)
    image6 = utils.errorDiffusionDithering(image.copy(), colorMap6, 6)

    # calculate MSE
    print(f'MSE of error diffusion dithering with n=3: {np.mean((image3 - image) ** 2)}')
    print(f'MSE of error diffusion dithering with n=6: {np.mean((image6 - image) ** 2)}')

    return image3, image6

if __name__ == '__main__':
    # open input image
    imageFileName = 'img/Lenna.jpg'
    image = cv2.imread(imageFileName)

    # part A
    image3, image6 = A(image.copy())

    # generate color map
    colorMap3 = set([tuple(j) for i in image3 for j in i])
    colorMap6 = set([tuple(j) for i in image6 for j in i])

    # build look-up table
    lookUpTable3 = utils.constructTable(image3)
    lookUpTable6 = utils.constructTable(image6)

    # bit reduce
    image3_reduce = utils.bitReduce(image3, lookUpTable3)
    image6_reduce = utils.bitReduce(image6, lookUpTable6)

    # part B
    image3_d, image6_d = B(image.copy(), colorMap3, colorMap6)

    # bit reduce
    image3_d_reduce = utils.bitReduce(image3_d, lookUpTable3)
    image6_d_reduce = utils.bitReduce(image6_d, lookUpTable6)

    # plot
    f, ax = plt.subplots(2, 3, subplot_kw={'xticks': [], 'yticks': []})
    ax[0][0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0][0].set_title('Original')
    ax[0][1].imshow(cv2.cvtColor(image3, cv2.COLOR_BGR2RGB))
    ax[0][1].set_title('Quantization (n=3)')
    ax[0][2].imshow(cv2.cvtColor(image3_d, cv2.COLOR_BGR2RGB))
    ax[0][2].set_title('Dithering (n=3)')
    ax[1][0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[1][0].set_title('Original')
    ax[1][1].imshow(cv2.cvtColor(image6, cv2.COLOR_BGR2RGB))
    ax[1][1].set_title('Quantization (n=6)')
    ax[1][2].imshow(cv2.cvtColor(image6_d, cv2.COLOR_BGR2RGB))
    ax[1][2].set_title('Dithering (n=6)')
    plt.show()
