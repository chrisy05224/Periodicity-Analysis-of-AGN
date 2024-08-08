import numpy as np
import matplotlib.pyplot as plt
import emcee

def multiply(num, arr):
    # This method takes an array and multiplies each value by a scale factor.
    '''
    :param num: A float value
    :param arr: An array of numbers
    :return: A new array, which is the original array with each element being the original element multiplied by num.
    '''

    newArr = []
    for i in arr:
        newNum = num*i
        newArr.append(newNum)
    return newArr

def model(theta, times):
    # This method generates a model using a theta vector (a tuple of the amplitude, period, and shift) and time values.
    '''
    :param theta: Theta vector with amplitude, period, and vertical shift of the model.
    :param times: Time values.
    :return: A model using the time values and theta vector.
    '''

    a1, p1, T0 = theta
    return a1 * np.sin(multiply(2*np.pi/p1, times)) + T0

def lnlike(theta, x, y, yerr):
    # this model returns the likeliness of a theta vector's model when compared to original data.
    '''
    :param theta: Theta vector with amplitude, period, and vertical shift of the model.
    :param x: Original, raw time data.
    :param y: Original, raw time series, or flux.
    :param yerr: Flux errod of the original flux data.
    :return: The lnlike, or likeliness, of the model.
    '''

    lnlike = -0.5 * np.sum(((y - model(theta, x))/yerr) ** 2)
    return lnlike

def lnprior(theta):
    # This method checks if the theta vector's components are within its bounds, or priors.

    a1, p1, T0 = theta
    if 0.0 < a1 < 80.0 and 0.5 < p1 < 5. and 0. < T0 < 150.:
        return 0.0
    return -np.inf

def lnprob(theta, x, y, yerr):
    # this method runs the lnprior() method and if the theta vector is within the priors, and then returns the lnlike if it is.

    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

def main(p0,nwalkers,niter,ndim,lnprob,data):
    '''
    :param p0: The initial position of the walkers.
    :param nwalkers: Number of walkers in the MCMC chain.
    :param niter: Number of iterations to assign per walker.
    :param ndim: Number of dimensions in the parameter space.
    :param lnprob: The lnprob() method from earlier.
    :param data: The time series, or flux, data.
    :return: The sampler object, the position of the walkers, the final lnlike, and the final state of the walkers.
    '''

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

    # The burn-in phase of MCMC.
    p0, _, _ = sampler.run_mcmc(p0, 300)
    sampler.reset()

    # The production phase of MCMC.
    pos, prob, state = sampler.run_mcmc(p0, niter)

    return sampler, pos, prob, state

def mcmc(times, flux, flux_error, estimatePeriod):
    '''
    :param times: The time values of the time series.
    :param flux: The time series, or flux.
    :param flux_error: The error of the flux.
    :param estimatePeriod: The estimated period, which can be derived from the previous literature periods found of the specific AGN.
    :return: The dominant period, the lnlike of the fit, and a boolean on whether the significance is denoted by FAP or not.
    '''

    data = (times, flux, flux_error)
    nwalkers = 128
    niter = 20000
    initial = np.array([0.325e-7, estimatePeriod, 0.65e-7])
    ndim = len(initial)
    p0 = [np.array(initial) + 1e-11 * np.random.randn(ndim) for i in range(nwalkers)]

    sampler, pos, prob, state = main(p0, nwalkers, niter, ndim, lnprob, data)
    samples = sampler.flatchain

    theta_max = samples[np.argmax(sampler.flatlnprobability)]
    domPeriod = theta_max[1]

    # If you wish to plot the MCMC sinusoidal fit against the data, then you can uncomment the below three lines of code.
    #plt.plot(times, flux)
    #plt.plot(times, theta_max[0] * np.sin(multiply(2 * np.pi / domPeriod, times)) + theta_max[2], color = 'r')
    #plt.show()

    lnlike = mcmcSig(times, flux, theta_max, flux_error)

    return domPeriod, lnlike, False

def mcmcSig(times, flux, theta_max, error):
    '''
    :param times: The time data of the time series.
    :param flux: The time series, or flux.
    :param theta_max: The theta vector to plot the model with.
    :param error: The flux error of the data.
    :return: The likeliness, or lnlike, of the final model.
    '''

    domPeriod = theta_max[1]
    domAmp = theta_max[0]
    shift = theta_max[2]

    model = domAmp * np.sin(multiply(2 * np.pi / domPeriod, times)) + shift
    lnlike = -0.5 * np.sum(((flux - model)/error) ** 2)

    return lnlike