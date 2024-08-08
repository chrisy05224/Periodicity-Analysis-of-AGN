import scipy.signal as sig
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

def dft(flux, maxFreq):
    '''
    :param flux: The time series on which we are performing the Discrete Fourier Transform; flux should be an array.
    :param maxFreq: The maximum frequency that we could possibly sample, which is 0.5 divided by the minimum time bin converted into years.
    :return: The dominant period, False Alarm Probability of the dominant period, and a boolean for if the significance is denoted by the FAP or not.
    '''

    freqs, power = sig.welch(flux, fs=2 * maxFreq, nperseg=len(flux), scaling='spectrum')

    # If you wish to plot the periodogram generated, you may uncomment the below three lines of code.
    #plt.plot(freqs, power)
    #plt.xscale('log')
    #plt.show()

    domFreq = freqs[np.argmax(power)]
    domPeriod = 1 / domFreq

    fap = fisherDFT(flux, 2*maxFreq)

    return domPeriod, fap, True

def fisherDFT(flux, fs, numPermutations=20000):
    '''
    :param flux: The time series which we are performing the DFT on.
    :param fs: The sampling frequency of the DFT, which is 2 times the maximum frequency.
    :param numPermutations: The number of permutations of Fisher's method of randomization; default is 20000.
    :return: The False Alarm Probability calculated using Fisher's method of randomization.
    '''
    frequencies, ogPeriodogram = sig.welch(flux, fs, nperseg = len(flux), scaling = 'spectrum')

    ogPeakValue = ogPeriodogram[np.argmax(ogPeriodogram)]

    randomizedPeakValues = np.zeros(numPermutations)

    for i in range(numPermutations):
        randomizedFlux = np.random.permutation(flux)
        _, randomizedPeriodogram = sig.welch(randomizedFlux, fs, nperseg = len(randomizedFlux), scaling = 'spectrum')
        randomizedPeakValues[i] = np.max(randomizedPeriodogram)

    fap = 1 - stats.percentileofscore(randomizedPeakValues, ogPeakValue) / 100

    return fap