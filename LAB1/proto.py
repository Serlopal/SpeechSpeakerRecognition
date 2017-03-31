# DT2118, Lab 1 Feature Extraction
import tools
import math
import numpy as np
import scipy.signal
import scipy.fftpack
import matplotlib.pyplot as plt
from scipy.spatial import distance


def distances_global(input):
    word_mfcc = []
    for utte in input:
        word_mfcc.append(mfcc(utte['samples']))

    glob_mat = np.zeros([len(word_mfcc),len(word_mfcc)])
    for i,word in enumerate(word_mfcc):
        for j,word2 in enumerate(word_mfcc):
            glob_mat[i,j] = dtw(word, word2, distance.euclidean)
    return glob_mat


def correlation_mfcc(input,d):
    result = np.empty((0,d))
    for utterance in input:
        result = np.append(result, mfcc(utterance['samples']),axis=0)
    return result

# Function given by the exercise ----------------------------------

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22, cepstrum_flag=False):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    mspec = logMelSpectrum(spec, samplingrate)

    if cepstrum_flag is True:
        ceps = cepstrum(mspec, nceps)
        return tools.lifter(ceps, liftercoeff)
    else:
        return mspec

# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """


    N = math.floor(((len(samples)-winlen)/winshift)+1)
    x = np.zeros([N, winlen])
    for i in range(N):
        x[i] = samples[i*winshift:(i*winshift)+winlen]
    return x

def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    return scipy.signal.lfilter([1,-1*p], [1], input, 1)

def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windowed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    M = input.shape[1]
    window = scipy.signal.hamming(M, False)
    return np.multiply(input, window)

def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """

    return abs(scipy.fftpack.fft(input,nfft))**2


def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    nfft = input.shape[1]
    triangular_filters = tools.trfbank(samplingrate, nfft)
    # plt.plot(triangular_filters)
    # plt.show()
    return np.log(np.dot(input,triangular_filters.T))

def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    return scipy.fftpack.realtransforms.dct(input, norm='ortho')[:, :nceps]

def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path through AD

    Note that you only need to define the first output for this exercise.
    """
    N=x.shape[0]
    M=y.shape[0]
    locD = np.zeros([N,M])

    for i in range(N):
        for j in range(M):
            locD[i,j]=dist(x[i],y[j])


    AccD = np.zeros([N,M])
    AccD[:,0]=locD[:,0]
    AccD[0,:]=locD[0,:]

    for i in range(1,N):
        for j in range(1,M):
            AccD[i,j] = locD[i,j] + min(AccD[i-1,j],AccD[i-1,j-1],AccD[i,j-1])

    return AccD[N-1,M-1]