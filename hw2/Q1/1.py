import cv2
import numpy as np
import os

def saveImage(image, fileName):
    outDir = 'output'
    try:
        os.mkdir(outDir)
    except:
        pass
    cv2.imwrite(os.path.join(outDir, fileName), image)

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

    np.putmask(imageRGB, imageRGB > 255, 255)
    np.putmask(imageRGB, imageRGB < 0, 0)
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

def IDCTProcess(image, dctMat):
    # D' = A' * D * A
    return np.dot(np.dot(np.transpose(dctMat), image), dctMat)

def DCT2d(image, inverse = False):
    # define height and width for the image
    h, w = image.shape[:2]

    # initialize dcted image
    image_dct = np.zeros_like(image)

    # get dct matrix
    dctMat = genDCTMat(8)

    # divide into 8*8 blocks
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            if inverse:
                image_dct[i:i+8, j:j+8] = IDCTProcess(image[i:i+8, j:j+8], dctMat)
            else:
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

def uniformQuantize(image, m, t):
    # define height and width for the image
    h, w = image.shape[:2]

    # define quantize table
    if t == 'R':
        quantizeTable = np.array([[8, 6, 6, 7, 6, 5, 8, 7],
                                  [7, 7, 9, 9, 8, 10 ,12, 20],
                                  [13, 12, 11, 11, 12, 25, 18, 19],
                                  [15, 20, 29, 26, 31, 30, 29, 26], 
                                  [28, 28, 32, 36, 46, 39, 32, 34],
                                  [44, 35, 28, 28, 40, 55, 41, 44],
                                  [48, 49, 52, 52, 52, 31, 39, 57], 
                                  [61, 56, 50, 60, 46, 51, 52, 50]])
    elif t == 'L':
        quantizeTable = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                  [12, 12, 14, 19, 26, 58, 60, 55],
                                  [14, 13, 16, 24, 40, 57, 69, 56],
                                  [14, 17, 22, 29, 51, 87, 80, 62],
                                  [18, 22, 37, 56, 68, 109, 103, 77],
                                  [24, 35, 55, 64, 81, 104, 113, 92],
                                  [49, 64, 78, 87, 103, 121, 120, 101],
                                  [72, 92, 95, 98, 112, 100, 103, 99]])
    else:
        quantizeTable = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                                  [18, 21, 26, 66, 99, 99, 99, 99],
                                  [24, 26, 56, 99, 99, 99, 99, 99],
                                  [47, 66, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99]])
    
    # initialize quantized image
    image_quantized = np.zeros_like(image)

    # divide into 8*8 blocks
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            image_quantized[i:i+8, j:j+8] = np.round(image[i:i+8, j:j+8] / quantizeTable)

    # save with m bits
    maxValue = np.amax(image_quantized)
    minValue = np.amin(image_quantized)
    quant_step = (maxValue - minValue) / (2 ** m - 1)
    image_quantized = np.round((image_quantized - minValue) / quant_step)
    
    return image_quantized, quant_step, minValue

def unQuantize(image, quant_step, t, minValue):
    # define height and width for the image
    h, w = image.shape[:2]

    # define quantize table
    if t == 'R':
        quantizeTable = np.array([[8, 6, 6, 7, 6, 5, 8, 7],
                                  [7, 7, 9, 9, 8, 10 ,12, 20],
                                  [13, 12, 11, 11, 12, 25, 18, 19],
                                  [15, 20, 29, 26, 31, 30, 29, 26], 
                                  [28, 28, 32, 36, 46, 39, 32, 34],
                                  [44, 35, 28, 28, 40, 55, 41, 44],
                                  [48, 49, 52, 52, 52, 31, 39, 57], 
                                  [61, 56, 50, 60, 46, 51, 52, 50]])
    elif t == 'L':
        quantizeTable = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                  [12, 12, 14, 19, 26, 58, 60, 55],
                                  [14, 13, 16, 24, 40, 57, 69, 56],
                                  [14, 17, 22, 29, 51, 87, 80, 62],
                                  [18, 22, 37, 56, 68, 109, 103, 77],
                                  [24, 35, 55, 64, 81, 104, 113, 92],
                                  [49, 64, 78, 87, 103, 121, 120, 101],
                                  [72, 92, 95, 98, 112, 100, 103, 99]])
    else:
        quantizeTable = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                                  [18, 21, 26, 66, 99, 99, 99, 99],
                                  [24, 26, 56, 99, 99, 99, 99, 99],
                                  [47, 66, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99]])

    # unquantize "save with m bits"
    image = image * quant_step + minValue

    # unquantize "quantize table"
    # initialize unquantized image
    image_unquantized = np.zeros_like(image)

    # divide into 8*8 blocks
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            image_unquantized[i:i+8, j:j+8] = image[i:i+8, j:j+8] * quantizeTable

    return image_unquantized

def DCTCompression(image, n, m, t):
    # perform 2D-DCT on each block
    image_dct = DCT2d(image)

    # for each block, keep only lower-frequency coefficients
    image_dct_rounded = coeffRound(image_dct, n)

    # apply uniform quantization with suitable quantization table and save each coefficient with m bits
    image_dct_rounded_quantized, quant_step, minValue = uniformQuantize(image_dct_rounded, m, t)

    ###############################################
    # unquantize
    image_dct = unQuantize(image_dct_rounded_quantized, quant_step, t, minValue)

    # inverse-DCT
    image = DCT2d(image_dct, inverse=True)

    return image

def A(image, n, m):
    # shift pixel values by subtracting 128
    image_ = image.astype('float64')
    image_ = image_ - 128

    # RGB channel
    image_[:,:,0] = DCTCompression(image_[:,:,0], n, m, t='R')
    image_[:,:,1] = DCTCompression(image_[:,:,1], n, m, t='R')
    image_[:,:,2] = DCTCompression(image_[:,:,2], n, m, t='R')

    # shift pixel values back
    image_ = image_ + 128
    np.putmask(image_, image_ > 255, 255)
    np.putmask(image_, image_ < 0, 0)
    image_ = image_.astype('uint8')

    return image_

def B(image, n, m):
    # convert to YCbCr
    imageYCbCr = RGB2YCbCr(image)
    # apply 4:2:0
    imageYCbCr[1::2, :] = imageYCbCr[::2, :]
    imageYCbCr[:, 1::2] = imageYCbCr[:, ::2] 
    # shift pixel values by subtracting 128
    imageYCbCr = imageYCbCr - 128

    # Y channel
    imageYCbCr[:,:,0] = DCTCompression(imageYCbCr[:,:,0], n, m, t='L')
    # Cb channel
    imageYCbCr[:,:,1] = DCTCompression(imageYCbCr[:,:,1], n, m, t='C')
    # # Cr channel
    imageYCbCr[:,:,2] = DCTCompression(imageYCbCr[:,:,2], n, m, t='C')

    # shift pixel values back
    imageYCbCr = imageYCbCr + 128
    # convert to RGB
    imageRGB = YCbCr2RGB(imageYCbCr)

    return imageRGB

if __name__ == '__main__':
    # get input
    catImageBGR = cv2.imread('cat.jpg')
    barImageBGR = cv2.imread('Barbara.jpg')
    # convert BGR to RGB
    catImageRGB = catImageBGR[:,:,::-1]
    barImageRGB = barImageBGR[:,:,::-1]

    d = {'cat': catImageRGB, 'bar': barImageRGB}
    ### A part ###
    for file in d.keys():
        for n in [2, 4]:
            for m in [4, 8]:
                image = A(d[file], n, m)
                saveImage(image[:,:,::-1], f'{file}_n{n}m{m}_a.jpg')

    ### B part ###
    for file in d.keys():
        for n in [2, 4]:
            for m in [4, 8]:
                image = B(d[file], n, m)
                saveImage(image[:,:,::-1], f'{file}_n{n}m{m}_b.jpg')
