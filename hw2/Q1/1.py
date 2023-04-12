import cv2
import numpy as np

def YCbCr2RGB(image):
    # define height and width for the image
    h, w = image.shape[:2]

    # initialize new image
    imageRGB = np.zeros_like(image, dtype='float64')

    # calculate each channel
    for i in range(h):
        for j in range(w):
            # R = 1.164 * (Y - 16) + 1.596 * (Cr - 128)
            imageRGB[i, j, 0] += 1.164 * (image[i, j, 0] - 16) + 1.596 * (image[i, j, 2] - 128)
            # G = 1.164 * (Y - 16) - 0.392 * (Cb - 128) - 0.813 * (Cr - 128)
            imageRGB[i, j, 1] += 1.164 * (image[i, j, 0] - 16) - 0.392 * (image[i, j, 1] - 128) - 0.813 * (image[i, j, 2] - 128)
            # B = 1.164 * (Y - 16) + 2.017 * (Cb - 128)
            imageRGB[i, j, 2] += 1.164 * (image[i, j, 0] - 16) + 2.017 * (image[i, j, 1] - 128)

    imageRGB = imageRGB.astype('uint8')

    return imageRGB

def RGB2YCbCr(image):
    # define height and width for the image
    h, w = image.shape[:2]

    # initialize new image
    imageYCbCr = np.zeros_like(image, dtype='float64')

    # calculate each channel
    for i in range(h):
        for j in range(w):
            # Y = 0.257 * R + 0.564 * G + 0.098 * B + 16
            imageYCbCr[i, j, 0] += 0.257 * image[i, j, 0] + 0.564 * image[i, j, 1] + 0.098 * image[i, j, 2] + 16
            # Cb = -0.148 * R - 0.291 * G + 0.439 * B + 128
            imageYCbCr[i, j, 1] += -0.148 * image[i, j, 0] - 0.291 * image[i, j, 1] + 0.439 * image[i, j, 2] + 128
            # Cr = 0.439 * R - 0.368 * G - 0.071 * B + 128
            imageYCbCr[i, j, 2] += 0.439 * image[i, j, 0] - 0.368 * image[i, j, 1] - 0.071 * image[i, j, 2] + 128

    return imageYCbCr

def genDCTMat(n):
    dct = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == 0:
                dct[i, j] = 1 / np.sqrt(n)
            else:
                dct[i, j] = np.sqrt(2 / n) * np.cos(np.pi * i * (2 * j + 1) / (2 * n))
    return dct

def DCTProcess(image, dctMat):
    # D = A * I * A'
    return np.dot(np.dot(dctMat, image), np.transpose(dctMat))

def DCT2d(image):
    # define height and width for the image
    h, w = image.shape[:2]

    # initialize dcted image
    image_dct = np.zeros_like(image)

    # get dct matrix
    dctMat = genDCTMat(8)

    # divide into 8*8 blocks
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            image_dct[i:i+8, j:j+8] = DCTProcess(image[i:i+8, j:j+8], dctMat)
    
    return image_dct

def coeffRound(image, n):
    # define height and width for the image
    h, w = image.shape[:2]

    # initialize dcted image
    image_rounded = np.zeros_like(image)

    # divide into 8*8 blocks
    for iB in range(0, h, 8):
        for jB in range(0, w, 8):
            # keep upper-left n-by-n coefficients by setting the others to zero
            for i in range(n):
                for j in range(n):
                    image_rounded[iB + i, jB + j] = image[iB + i, jB + j]
    
    return image_rounded

def uniformQuantize(image, m):
    # define height and width for the image
    h, w = image.shape[:2]

    # define quantize table
    quantizeTable = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                            [12, 12, 14, 19, 26, 58, 60, 55],
                            [14, 13, 16, 24, 40, 57, 69, 56],
                            [14, 17, 22, 29, 51, 87, 80, 62],
                            [18, 22, 37, 56, 68, 109, 103, 77],
                            [24, 35, 55, 64, 81, 104, 113, 92],
                            [49, 64, 78, 87, 103, 121, 120, 101],
                            [72, 92, 95, 98, 112, 100, 103, 99]])
    
    # initialize quantized image
    image_quantized = np.zeros_like(image)

    # divide into 8*8 blocks
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            image_quantized[i:i+8, j:j+8] = np.round(image[i:i+8, j:j+8] / quantizeTable)
    
    return image_quantized

def DCTCompression(imageRGB, n, m):
    # step 1. convert to YCbCr
    imageYCbCr = RGB2YCbCr(imageRGB)
    # step 2. Shift pixel values by subtracting 128
    imageYCbCr = imageYCbCr - 128
    # step 3. divide to 8*8 blocks and perform 2D-DCT on each block
    imageY_dct = DCT2d(imageYCbCr[:, :, 0])
    print(imageY_dct[0:4, 0:4])
    # step 4. for each block, keep only lower-frequency coefficients
    imageY_dct_rounded = coeffRound(imageY_dct, n)
    print(imageY_dct_rounded[0:4, 0:4])
    # step 5. Apply uniform quantization with suitable quantization table
    imageY_dct_rounded_quantized = uniformQuantize(imageY_dct_rounded, m)
    print(imageY_dct_rounded_quantized[0:4, 0:4])
    # step 6. save each coefficient with m bits
    

    return imageRGB

if __name__ == '__main__':
    # get input
    catImageBGR = cv2.imread('cat.jpg')
    barImageBGR = cv2.imread('Barbara.jpg')
    # convert BGR to RGB
    catImageRGB = catImageBGR[:,:,::-1]
    barImageRGB = barImageBGR[:,:,::-1]

    
    catImage_n2m2 = DCTCompression(catImageRGB.copy(), 2, 2)