import numpy as np
import math
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt


def lsp(times, flux, flux_error, minFreq, maxFreq, estimateFrequency):
    '''
    :param times: The time values of the data.
    :param flux: The time series, or flux, with which we are calculating the period of.
    :param flux_error: The flux error of the time series.
    :param minFreq: The minimum possible frequency, which is the reciprocal of the maximum time interval in years, or the current year minus the year that data started to be collected.
    :param maxFreq: The maximum frequency that we could possibly sample, which is 0.5 divided by the minimum time bin converted into years.
    :param estimateFrequency: The estimated dominant frequency of the time series, which can be derived from the previous literature periods found of the specific AGN.
    :return: The dominant period, the False Alarm Probability, and a boolean for if the significance is denoted by the FAP or not.
    '''

    ls = LombScargle(times, flux, flux_error)
    freqs = np.logspace(math.log(minFreq), math.log(maxFreq), 100000)
    power = ls.power(freqs)
    fap = LombScargle.false_alarm_probability(ls, power=power, method='bootstrap', minimum_frequency=minFreq, maximum_frequency=maxFreq)
    fal = list(LombScargle.false_alarm_level(ls, fap, method='bootstrap', minimum_frequency=minFreq, maximum_frequency=maxFreq))

    falseAlarmLevel = []
    for item in fal:
        newItem = float(item)
        falseAlarmLevel.append(newItem)

    # We must find the indices of the relevant interval, because the overall periodogram is affected by noise.
    # If we do not do this, then the dominant frequency that we compute might be one close to zero due to noise, which is not what we want.
    lower = 0
    upper = 0
    for i in range(0, len(freqs)):
        if freqs[i] > estimateFrequency - 0.1:
            lower = i
            break
    for j in reversed(range(lower, len(freqs))):
        if freqs[j] < estimateFrequency + 0.1:
            upper = j
            break

    relevantFrequencies = freqs[lower:upper + 1]
    relevantFALs = falseAlarmLevel[lower:upper + 1]

    # If you wish to plot the periodogram, you may uncomment the below five lines of code.
    #plt.plot(relevantFrequencies, relevantFALs)
    #plt.xlabel('Frequencies')
    #plt.ylabel('Power')
    #plt.xscale('log')
    #plt.show()

    domFreq = relevantFrequencies[np.argmax(relevantFALs)]

    falseAlarmProb = fap[np.where(freqs == domFreq)[0][0]]

    domPeriod = 1 / domFreq
    return domPeriod, falseAlarmProb, True