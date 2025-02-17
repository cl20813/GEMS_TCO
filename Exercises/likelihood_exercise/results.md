
# Full Likelihood V.S. Vecchia Approximated Likelihood

Data size: '''1,250 x 8(hours)''', Model: ```Matern($\sqrt{ ||x-vt||+\beta^2t^2}}$)```.

Conditioning number: 10 (5 on current space, 5 on one lag space)

parameters (sigmasq, range_latitude, range_longitude, advection, beta and nugget): [60, 8.25, 8.25, 0.5, 0.5, 5.0]
Full Likelihood: 24274.6368 (826 seconds). Vecchia Approximation: 24312.1206 (0.0101 seconds, this does not make sense)

parameters (sigmasq, range_latitude, range_longitude, advection, beta and nugget): [40, 5.25, 5.25, 0.5, 0.5, 0.5]
Full Likelihood: 24443.6162 (881 seconds). Vecchia Approximation: 24354.4574 (0.0101 seconds, this does not make sense)


