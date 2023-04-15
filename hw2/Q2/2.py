import scipy.io.wavfile as wavfile
import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

def saveSpectrum(signal, rate, fileName):
    outDir = 'output'
    try:
        os.mkdir(outDir)
    except:
        pass
    plt.clf()
    spectrum = abs(np.fft.fft(signal))
    freqs = np.fft.fftfreq(len(signal), 1.0 / rate)
    plt.figure(figsize=(15, 5))
    plt.plot(freqs, spectrum)
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.savefig(os.path.join(outDir, fileName))
    plt.clf()

def saveShape(signal, fileName):
    outDir = 'output'
    try:
        os.mkdir(outDir)
    except:
        pass
    plt.clf()
    plt.figure(figsize=(15, 5))
    plt.plot(signal)
    plt.savefig(os.path.join(outDir, fileName))
    plt.clf()

def blackman(N):
    # w(n)=0.42+0.5cos(2pin/n-1)+0.08cos(4pin/n-1)
    if N < 1:
        return np.array([], dtype=np.result_type(N, 0.0))
    elif N == 1:
        return np.ones(1, dtype=np.result_type(N, 0.0))
    
    n = np.arange(1 - N, N, 2)

    return 0.42 + 0.5*np.cos(2.0*np.pi*n/(N-1)) + 0.08*np.cos(4.0*np.pi*n/(N-1))

def constructFilter(freq_sample, windowSize, mode, *freq_cut):
    if mode == 'band-pass':
        (freq_cut1, freq_cut2) = freq_cut
        freq_cut1, freq_cut2 = freq_cut1 / freq_sample, freq_cut2 / freq_sample
    else:
        (freq_cut,) = freq_cut
        freq_cut = freq_cut / freq_sample

    middle = windowSize // 2

    fltr = np.zeros(windowSize)
    for n in range(- windowSize // 2, windowSize // 2 + 1):
        if n == 0:
            fltr[middle] = 1
        else:
            if mode == 'low-pass':
                fltr[n + middle] = np.sin(2 * np.pi * freq_cut * n) / (np.pi * n)
            elif mode == 'high-pass':
                fltr[n + middle] = -np.sin(2 * np.pi * freq_cut * n) / (np.pi * n)
            else:
                fltr[n + middle] = np.sin(2 * np.pi * freq_cut2 * n) / (np.pi * n) - np.sin(2 * np.pi * freq_cut1 * n) / (np.pi * n)

    if mode == 'low-pass':
        fltr[middle] = 2 * freq_cut
    elif mode == 'high-pass':
        fltr[middle] = 1 - 2 * freq_cut
    else:
        fltr[middle] = 2 * (freq_cut2 - freq_cut1)

    fltr *= blackman(windowSize)

    return fltr

def showF(audio, freq_sample):
    spectrum = abs(np.fft.fft(audio))
    freqs = np.fft.fftfreq(len(audio), 1.0 / freq_sample)
    plt.clf()
    plt.figure(figsize=(15, 5))
    plt.plot(freqs, spectrum)
    plt.show()

def convolution(audio, filter):
    # pad the filter to same length as audio
    filter_pad = np.zeros(len(audio))
    filter_pad[:len(filter)] = filter[::-1]

    # apply the filter to audio
    filtered_audio = np.fft.ifft(np.fft.fft(audio) * np.fft.fft(filter_pad)).real

    return filtered_audio

def subSample(signal, fs, newfs=2000):
    subRatio = fs // newfs

    newSignal = np.zeros(len(signal) // subRatio)
    for i in range(len(newSignal)):
        newSignal[i] = np.mean(signal[i*subRatio:(i+1)*subRatio])

    return newSignal

def echo(signal, mode='one'):
    delay = 12800

    newSignal = np.zeros_like(signal)

    newSignal[:delay] = signal[:delay]
    for i in range(delay, len(signal)):
        if mode == 'one':
            newSignal[i] = signal[i] + 0.8 * signal[i - delay]
        else:
            newSignal[i] = signal[i] + 0.8 * newSignal[i - delay]

    return newSignal

if __name__ == '__main__':
    # get input
    freq_sample, audio = wavfile.read('HW2_Mix.wav')
    
    # build filter
    filter1 = constructFilter(freq_sample, 10001, 'low-pass', 400)
    filter2 = constructFilter(freq_sample, 10001, 'high-pass', 800)
    filter3 = constructFilter(freq_sample, 10001, 'band-pass', 400, 800)

    # convolution
    filtered_audio1 = convolution(audio, filter1)
    filtered_audio2 = convolution(audio, filter2)
    filtered_audio3 = convolution(audio, filter3)

    # subsampling
    filtered_audio1_2k = subSample(filtered_audio1, freq_sample, 2000)
    filtered_audio2_2k = subSample(filtered_audio2, freq_sample, 2000)
    filtered_audio3_2k = subSample(filtered_audio3, freq_sample, 2000)

    #echo
    echo1 = echo(filtered_audio1, 'one')
    echo2 = echo(filtered_audio1, 'multiple')

    # save results
    saveSpectrum(audio, freq_sample, 'input.png')

    saveSpectrum(filter1, freq_sample, 'LowPass_spectrum.png')
    saveSpectrum(filter2, freq_sample, 'HighPass_spectrum.png')
    saveSpectrum(filter3, freq_sample, 'BandPass_spectrum.png')

    saveShape(filter1, 'LowPass_shape.png')
    saveShape(filter2, 'HighPass_shape.png')
    saveShape(filter3, 'BandPass_shape.png')

    saveSpectrum(filtered_audio1, freq_sample, 'output_by_LowPass.png')
    saveSpectrum(filtered_audio2, freq_sample, 'output_by_HighPass.png')
    saveSpectrum(filtered_audio3, freq_sample, 'output_by_BandPass.png')

    sf.write('output/LowPass_fc400_ws10001.wav', filtered_audio1, freq_sample)
    sf.write('output/HighPass_fc800_ws10001.wav', filtered_audio2, freq_sample)
    sf.write('output/BandPass_fc400-800_ws10001.wav', filtered_audio3, freq_sample)

    sf.write('output/LowPass_fc400_ws10001_2kHZ.wav', filtered_audio1_2k, 2000)
    sf.write('output/HighPass_fc400_ws10001_2kHZ.wav', filtered_audio2_2k, 2000)
    sf.write('output/BandPass_fc400-800_ws10001_2kHZ.wav', filtered_audio3_2k, 2000)

    sf.write('output/Echo_one.wav', echo1, freq_sample)
    sf.write('output/Echo_multiple.wav', echo2, freq_sample)