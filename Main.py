import json
import DFT as dft
import MCMC as mcmc
import LombScargleBootstrap as lsp
import PDM as pdm
import csv

literaturePeriods = dict()
literaturePeriods['PG1553'] = 2.2
literaturePeriods['PKS2155'] = 1.7
literaturePeriods['OJ014'] = 4.3

interval = 2024-2008
maxFreq = 0.5/(1.0/12)
minFreq = 1.0/(interval)

def processData():
    while True:
        agn = input('Specify an AGN: ')
        if agn == 'Show':
            for key, _ in literaturePeriods.items():
                print(key)
            continue
        if agn == 'Quit':
            quit()

        try:
            data = json.loads(open(agn + '_1monthbinned.json').read())
        except:
            print('Error!')
        break

    times = []
    flux = []
    flux_error = []

    for item in data['ts']:
        times.append(float(item[0]) / (3600.0 * 24 * 365))

    for item in data['flux']:
        flux.append(float(item[1]))

    for item in data['flux_error']:
        rang = (float(item[1]), float(item[2]))
        error = (rang[1] - rang[0]) / 2
        flux_error.append(error)

    if len(times) != len(flux):
        times = []
        flux = []
        flux_error = []
        with open(agn + '_1monthbinned.csv', newline='') as f:
            reader = csv.reader(f)
            data = list(reader)

        for i in range(1, len(data)):
            date = data[i][2]
            f = data[i][4]
            ferr = data[i][5]
            if ferr == '-':
                continue
            else:
                times.append(float(date) / (3600.0 * 24 * 365))
                flux.append(float(f))
                flux_error.append(float(ferr))

    return agn, times, flux, flux_error

agn, times, flux, flux_error = processData()

results = dict()

#results['Lomb-Scargle (Bootstrap)'] = lsp.lsp(times, flux, flux_error, minFreq, maxFreq, 1/literaturePeriods[agn])
#results['Discrete Fourier Transform'] = dft.dft(flux, maxFreq)
#results['Phase Dispersion Minimization'] = pdm.pdm(times, flux, minFreq, maxFreq, 1.0/literaturePeriods[agn])
results['Markov Chain Monte Carlo Sinusoidal Curve Fitting'] = mcmc.mcmc(times, flux, flux_error, literaturePeriods[agn])

print('Here are the dominant periods found by each method for ' + agn + ':')
for method, (result, fap, isItFAP) in results.items():
    label = ''
    if isItFAP:
        label = 'FAP'
    else:
        label = 'Likeliness'
    print(method + ': ' + str(round(result, 3)) + ' (' + label + ': ' + str(round(fap, 4)) + ')')
print('Literature: around ' + str(literaturePeriods[agn]))