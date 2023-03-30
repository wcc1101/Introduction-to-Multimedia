import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils

if __name__ == '__main__':
    # open input image
    imageFileName = 'img/lake.jpg'
    image_RGB = cv2.cvtColor(cv2.imread(imageFileName), cv2.COLOR_BGR2RGB)

    # Gamma correction
    image_RGB_gamma, y_channel, y_channel_gamma = utils.gammaCorrect(image_RGB.copy(), 3.0)

    # plot
    f, ax = plt.subplots(2, 2)

    ax[0][0].imshow(image_RGB)
    ax[0][0].set_title('Original image')

    ax[0][1].imshow(image_RGB_gamma)
    ax[0][1].set_title('Gamma correct image')

    ax[1][0].hist(y_channel)
    ax[1][0].set_xlim(0, 256)
    ax[1][0].set_title('Original image')

    ax[1][1].hist(y_channel_gamma)
    ax[1][1].set_xlim(0, 256)
    ax[1][1].set_title('Gamma correct image')

    plt.show()

    # save the images
    utils.saveImage(cv2.cvtColor(image_RGB_gamma, cv2.COLOR_RGB2BGR), 'gamma_img.jpg')
    utils.saveHist(y_channel, 'Y_hist.jpg')
    utils.saveHist(y_channel_gamma, 'Y_hist_gamma.jpg')
