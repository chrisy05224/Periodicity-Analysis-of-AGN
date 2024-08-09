import json
import matplotlib.pyplot as plt
import csv

def plotRawData(agn):
    # This is a method for if you want to plot the raw data of the AGN.
    data = json.loads(open(agn + '_1monthbinned.json').read())

    times = []
    flux = []
    flux_error = []

    for item in data['ts']:
        times.append(float(item[0]) / (3600.0 * 24 * 365))

    for item in data['flux']:
        flux.append(float(item[1]))

    if len(times) != len(flux):
        times = []
        flux = []
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

    plt.plot(times, flux)
    plt.xlabel('Time')
    plt.ylabel('Flux')
    plt.xscale('log')
    plt.show()

plotRawData('PG1553')