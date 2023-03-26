
import glob
import numpy as np
import pandas as ps

#from h5 import HDFArchive
import h5py

import matplotlib.pyplot as plt

filenames = glob.glob('data*h5')

df = ps.DataFrame()

for filename in filenames:
    print(f'--> Loading: {filename}')
    d = {}
    with h5py.File(filename, 'r') as a:
        for key, value in a['data'].attrs.items():
            d[key] = [value]
    #print(d)
    ndf = ps.DataFrame.from_dict(d)
    df = ps.concat([df, ndf])

print(df)

maxorders = np.sort(np.unique(df['maxorder']))

styles = {
    1. : 'o',
    3. : 's',
    4. : 'D',
    5. : '+',
    6. : 'x',
    }

ntaus = np.sort(np.unique(df['ntau']))

plt.figure(figsize=(10, 9))

colors = {}
for ntau in ntaus:
    colors[ntau] = plt.plot([], [])[0].get_color()


for maxorder in maxorders:
    idx = df['maxorder'] == maxorder
    df1 = df[idx]
    ntaus = np.sort(np.unique(df1['ntau']))
    style = styles[maxorder]
    for ntau in ntaus:
        color = colors[ntau]
        idx = df1['ntau'] == ntau
        df2 = df1[idx]
        ns = np.array(df2['N_samples'])
        diff = np.array(df2['diff'])
        sidx = np.argsort(ns)
        #print(sidx)
        plt.plot(ns[sidx], diff[sidx],
                 '-' + style, alpha=0.5, color=color,
                 label=f'order {maxorder} ntau {ntau}')

plt.loglog([], [])
plt.legend(loc='lower left', ncol=1)
plt.grid(True)
plt.xlabel('N samples')
plt.ylabel('$\max|\Delta P(\tau)|$')
#plt.axis('square')

plt.tight_layout()
plt.savefig('figure_dimer_P_tau_cf.pdf')
plt.show()
