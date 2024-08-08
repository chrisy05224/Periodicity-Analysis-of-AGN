import numpy as np
from PyAstronomy.pyTiming import pyPDM
import matplotlib.pyplot as plt
import scipy.stats as stats

def pdm(times, flux, minFreq, maxFreq, estimateFrequency):
    '''
    :param times: The time values of the data.
    :param flux: The time series, or flux, with which we are calculating the period of.
    :param minFreq: The minimum possible frequency, which is the reciprocal of the maximum time interval in years, or the current year minus the year that data started to be collected.
    :param maxFreq: The maximum frequency that we could possibly sample, which is 0.5 divided by the minimum time bin converted into years.
    :param estimateFrequency: The estimated dominant frequency of the time series, which can be derived from the previous literature periods found of the specific AGN.
    :return: The dominant frequency of the time series, the False Alarm Probability, and a boolean for if the significance is denoted by the FAP or not.
    '''

    times = np.array(times)
    flux = np.array(flux)

    S = pyPDM.Scanner(minVal=minFreq, maxVal=maxFreq, dVal=0.001, mode="frequency")
    P = pyPDM.PyPDM(times, flux)
    freqs, theta = P.pdmEquiBinCover(10, 3, S)

    # We must find the indices of the relevant interval, because the overall plot is affected by noise.
    # If we do not do this, then the dominant frequency that we compute might be one due to noise, which is not what we want.
    lower = 0
    upper = 0
    for i in range(0, len(freqs)):
        if freqs[i] > estimateFrequency-0.1:
            lower = i
            break

    # finding maximum index
    for j in reversed(range(lower, len(freqs))):
        if freqs[j] < estimateFrequency+0.1:
            upper = j
            break

    relevantFrequencies = freqs[lower:upper+1]
    relevantThetas = theta[lower:upper+1]

    domFreq = relevantFrequencies[np.argmin(relevantThetas)]

    # If you wish to plot the periodogram, you may uncomment the below three lines of code.
    #plt.plot(freqs, theta)
    #plt.xscale('log')
    #plt.show()

    fap = fisherPDM(times, flux, minFreq, maxFreq)

    domPeriod = 1/domFreq
    return domPeriod, fap, True

def fisherPDM(times, flux, minFreq, maxFreq, numPermutations = 500):
    '''
    :param times: The time values of the data.
    :param flux: The time series, or flux, with which we are calculating the period of.
    :param minFreq: The minimum possible frequency, which is the reciprocal of the maximum time interval in years, or the current year minus the year that data started to be collected.
    :param maxFreq: The maximum frequency that we could possibly sample, which is 0.5 divided by the minimum time bin converted into years.
    :param numPermutations: The number of permutations of Fisher's method of randomization, which is default at 500.
    :return: The False Alarm Probability.
    '''

    originalS = pyPDM.Scanner(minVal=minFreq, maxVal=maxFreq, dVal=0.001, mode="frequency")
    originalP = pyPDM.PyPDM(times, flux)
    originalFreqs, originalThetas = originalP.pdmEquiBinCover(10, 3, originalS)
    minTheta = np.min(originalThetas)

    randomizedPeakValues = np.zeros(numPermutations)

    for i in range(numPermutations):
        randomizedFlux = np.random.permutation(flux)
        randomizedS = pyPDM.Scanner(minVal=minFreq, maxVal=maxFreq, dVal=0.001, mode="frequency")
        randomizedP = pyPDM.PyPDM(times, randomizedFlux)
        _, randomizedThetas = randomizedP.pdmEquiBinCover(10, 3, randomizedS)
        randomizedPeakValues[i] = np.min(randomizedThetas)

    fap = stats.percentileofscore(randomizedPeakValues, minTheta)

    return fap