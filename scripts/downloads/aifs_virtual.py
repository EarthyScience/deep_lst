import dynamical_catalog

single = dynamical_catalog.open("ecmwf-aifs-single-forecast-virtual", chunks=None)
print(single)
