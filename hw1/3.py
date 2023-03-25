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

def saveHist(hist, fileName):
    outDir = 'out'
    try:
        os.mkdir(outDir)
    except:
        pass
    plt.clf()
    plt.plot(hist)
    plt.title(fileName)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(outDir, fileName))

def RGB2YIQ(image):
    # define height and width for the image
    h, w = image.shape[:2]

    # initialize new image
    image_YIQ = np.zeros(shape=(h, w, 3), dtype='float64')

    # calculate each channel
    for i in range(h):
        for j in range(w):
            image_YIQ[i][j][0] += 0.299 * image[i][j][0] + 0.587 * image[i][j][1] + 0.114 * image[i][j][2]
            image_YIQ[i][j][1] += 0.596 * image[i][j][0] - 0.275 * image[i][j][1] - 0.321 * image[i][j][2]
            image_YIQ[i][j][2] += 0.212 * image[i][j][0] - 0.523 * image[i][j][1] + 0.311 * image[i][j][2]

    return image_YIQ

def YIQ2RGB(image):
    # define height and width for the image
    h, w = image.shape[:2]

    # initialize new image
    image_RGB = np.zeros(shape=(h, w, 3), dtype='float64')

    # calculate each channel
    for i in range(h):
        for j in range(w):
            image_RGB[i][j][0] += image[i][j][0] + 0.956 * image[i][j][1] + 0.621 * image[i][j][2]
            image_RGB[i][j][1] += image[i][j][0] - 0.272 * image[i][j][1] - 0.647 * image[i][j][2]
            image_RGB[i][j][2] += image[i][j][0] - 1.106 * image[i][j][1] + 1.703 * image[i][j][2]

    image_RGB = image_RGB.astype('uint8')

    return image_RGB

def gammaCorrect(image, gamma):
    # convert image to YIQ
    image_YIQ = RGB2YIQ(image.copy())
    y_channel = image_YIQ[:,:,0].copy()

    # standarize to 0-1
    image_YIQ[:,:,0] /= 255.0

    # gamma tranformation
    image_YIQ[:,:,0] = np.power(image_YIQ[:,:,0], gamma)

    # de-standarize
    image_YIQ[:,:,0] *= 255
    y_channel_gamma = image_YIQ[:,:,0].copy()

    # convert image to RGB
    image_RGB_gamma = YIQ2RGB(image_YIQ).astype('uint8')

    return image_RGB_gamma, y_channel, y_channel_gamma

if __name__ == '__main__':
    # open input image
    imageFileName = 'img/lake.jpg'
    image = cv2.imread(imageFileName)

    # convert image to RGB
    image_RGB = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

    # Gamma correction
    image_RGB_gamma, y_channel, y_channel_gamma = gammaCorrect(image_RGB.copy(), 3.0)

    # plot
    f, ax = plt.subplots(2, 2)

    ax[0][0].imshow(image_RGB)
    ax[0][0].set_title('Original image')

    ax[0][1].imshow(image_RGB_gamma)
    ax[0][1].set_title('Gamma correct image (Gamma=3)')

    hist, bins = np.histogram(y_channel.ravel(), 256, [0, 256])
    ax[1][0].plot(hist)
    ax[1][0].set_xlim(0, 256)
    ax[1][0].set_title('Hist of original image')
    ax[1][0].set_xlabel('Value')
    ax[1][0].set_ylabel('Frequency')

    hist_g, bins_g = np.histogram(y_channel_gamma.ravel(), 256, [0, 256])
    ax[1][1].plot(hist_g)
    ax[1][1].set_xlim(0, 256)
    ax[1][1].set_title('Hist of gamma correct image (Gamma=3)')
    ax[1][1].set_xlabel('Value')
    ax[1][1].set_ylabel('Frequency')

    plt.show()

    # save the images
    saveImage(cv2.cvtColor(image_RGB_gamma, cv2.COLOR_RGB2BGR), 'gamma_img.jpg')
    saveHist(hist, 'Y_hist.jpg')
    saveHist(hist_g, 'Y_hist_gamma.jpg')
