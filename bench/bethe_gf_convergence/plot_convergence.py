
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from triqs_tprf.ParameterCollection import ParameterCollection

def read_and_sort(filenames):

    ps = []
    df = pd.DataFrame()

    for filename in filenames:
        print(f'--> Loading: {filename}')
        d = dict()
        with h5py.File(filename, 'r') as fd:
            for key, val in fd['data'].attrs.items():
                #print(key)
                d[key] = val
            for key, val in fd['data'].items():
                #print(key)
                d[key] = np.array(val)
                #print(d)

        #print(d.keys())

        #d['error'] = np.abs(np.interp(d['beta']*0.5, d['tau'], d['gf'] - d['gf_ref']))
        d['error'] = np.max(np.abs(d['gf'] - d['gf_ref']))
        d['order'] = np.max(d['orders'])

        p = ParameterCollection(**d)
        ps.append(p)

        d.pop('gf')
        d.pop('gf_ref')
        d.pop('tau')
        
        d.pop('orders')
        d.pop('orders_bare')
        d.pop('orders_gf')

        #print(d.keys())
        #for key, val in d.items():
        #    print(f'key: {key}, val type : {type(val)}')
        
        #ndf = pd.DataFrame.from_dict(d)
        ndf = pd.DataFrame([d])
        df = pd.concat([df, ndf])
            
        
    order = np.array([np.max(p.orders) for p in ps])
    #print(order)
    sidx = np.argsort(order)
    ps = np.array(ps)[sidx]

    print(df)
    #exit()
    
    return ps, df

# --

paths = [
    #'./n_pts_after_max_1/data*h5',
    #'./n_pts_after_max_NoLimit/data*h5',
    './*h5',
    ]
filenames = glob.glob(paths[0])
ps, df = read_and_sort(filenames)

plt.figure(figsize=(3.25*2, 5))

ntaus = np.sort(np.unique(df['ntau']))
orders = np.sort(np.unique(df['order']))

styles = ['o', 's', 'd', '>', '<', '^', 'x', '+']

colors = []
for ntau in ntaus:
    c = plt.plot([], [], label=f'ntau {ntau}')[0].get_color()
    colors.append(c)

for style, order in zip(styles, orders):
    plt.plot([], [], style, color='gray', label=f'order {order}')
    
for style, order in zip(styles, orders):
    for color, ntau in zip(colors, ntaus):
        dfc = df[ df['ntau'] == ntau ]
        dfc = dfc[ dfc['order'] == order ]
        dfc = dfc.sort_values(by='N_samples')
        
        plt.plot(
            dfc['N_samples'], dfc['error'],
            '-' + style, alpha=0.5, color=color,
            #label=f'ntau {ntau}, order {order}',
            )

plt.legend()
plt.loglog([], [])
plt.grid(True)
#plt.ylim(bottom=1e-4)

plt.show(); exit()

plt.figure(figsize=(3.25, 5))
subp = [2, 1, 1]

plt.subplot(*subp); subp[-1] += 1

for p in ps:
    plt.plot(p.tau, p.gf.imag, label=f'order {np.max(p.orders)}', alpha=0.75)
    
p = ps[0]
plt.plot(p.tau, p.gf_ref.imag, '--k', label='Analytic')
plt.ylim(bottom=0)
plt.legend(loc='best')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$G(\tau)$')
plt.grid(True)

plt.subplot(*subp); subp[-1] += 1

for path in paths:
    filenames = glob.glob(path)
    ps = read_and_sort(filenames)

    order = np.array([np.max(p.orders) for p in ps])
    err = np.array([np.max(np.abs(p.gf - p.gf_ref)) for p in ps])
    if ps[0].n_pts_after_max == 1:
        label = 'n_pts_after_max = 1'
    else:
        label = 'n_pts_after_max = unrestricted'
        
    plt.plot(order, err, '-o', alpha=0.75, label=label)

plt.xlabel('order')
plt.ylabel('GF error')
plt.legend(loc='best')
plt.semilogy([], [])
plt.grid(True)

plt.tight_layout()
plt.savefig('figure_bethe_gf_convergence.pdf')

plt.show()
