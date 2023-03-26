
import glob
import numpy as np
import pandas as ps

import h5py

import matplotlib.pyplot as plt

from triqs_tprf.ParameterCollection import ParameterCollection

filenames = [
'data_dimer_ntau_8192_maxorder_1_Nsamples_256_md5_3ec769d56f60d2d77b6883842c5ef185.h5',
'data_dimer_ntau_32768_maxorder_3_Nsamples_4096_md5_72c2152b0d31e6f8751c034569e10ba7.h5',
'data_dimer_ntau_32768_maxorder_4_Nsamples_4096_md5_991d9643db7885b6e65f46e3b1f98b7c.h5'
    ]

ds = []
for filename in filenames:
    print(f'--> Loading: {filename}')
    d = {}
    with h5py.File(filename, 'r') as a:
        for key, value in a['data'].attrs.items():
            d[key] = [value]
        for key, value in a['data'].items():
            d[key] = np.array(value)
    d = ParameterCollection(**d)
    ds.append(d)

plt.figure()
subp = [3, 1, 1]

plt.subplot(*subp); subp[-1] += 1
for d in ds:
    plt.plot(d.tau, d.P1, 'r', label=f'P1 order {d.maxorder}')
    plt.plot(d.tau, d.P2, 'g', label=f'P2 order {d.maxorder}')
    plt.plot(d.tau, d.P1_exact, 'c--')
    plt.plot(d.tau, d.P2_exact, 'm--')

plt.subplot(*subp); subp[-1] += 1
for d in ds:
    plt.plot(d.tau, d.P1 - d.P1_exact, label=f'P1 order {d.maxorder}')
    plt.plot(d.tau, d.P2 - d.P2_exact, label=f'P2 order {d.maxorder}')

plt.subplot(*subp); subp[-1] += 1
for d in ds:
    plt.plot(d.tau, np.abs(d.P1 - d.P1_exact), label=f'P1 order {d.maxorder}')
    plt.plot(d.tau, np.abs(d.P2 - d.P2_exact), label=f'P2 order {d.maxorder}')
plt.semilogy([], [])

plt.legend(loc='best')
plt.tight_layout()
plt.show()
