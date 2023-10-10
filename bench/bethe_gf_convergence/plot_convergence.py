
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from triqs_tprf.ParameterCollection import ParameterCollection


SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 8

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


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

        #d.pop('gf')
        #d.pop('gf_ref')
        #d.pop('tau')
        
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
    #'./*ntau_1024*h5',
    ]
#filenames = glob.glob(paths[0])

filenames = \
    glob.glob('data_order_0:1_ntau_256*.h5') + \
    glob.glob('data_order_0:2_ntau_128*.h5') + \
    glob.glob('data_order_0:3_ntau_256*.h5') + \
    glob.glob('data_order_0:4_ntau_256*.h5') + \
    glob.glob('data_order_0:5_ntau_1024*.h5') + \
    glob.glob('data_order_0:6_ntau_1024*.h5') + \
    glob.glob('data_order_0:7_ntau_1024*.h5') 
    
ps, df = read_and_sort(filenames)

plt.figure(figsize=(3.25, 3.5))

from matplotlib.gridspec import GridSpec
gs = GridSpec(
    2, 2,
    width_ratios=[1.1, 1.6],
    height_ratios=[1.2, 2.0],
    wspace=0.0, hspace=0.15,
    bottom=0.11, top=0.99,
    left=0.15, right=0.99,
    )

#subp = [2, 1, 1]
#plt.subplot(*subp); subp[-1] += 1
#subp = [2, 2, 3]

ax = plt.subplot(gs[1, 1])

plt.text(0.9, 0.9, "(c)", size=9, ha="center", va="center", transform=ax.transAxes)
    
ntaus = np.sort(np.unique(df['ntau']))
orders = np.sort(np.unique(df['order']))

styles = ['o', 's', 'd', '>', '<', '^', 'x', '+']
#alphas = [0.1]*6 + [0.5]
#alphas = [0.75]*len(styles)

alphas = np.linspace(0.2, 0.75, num=len(ntaus))
alphas[:] = 0.75

colors = []

from matplotlib.cm import get_cmap
cmap = get_cmap(name='brg_r')
#cmap = get_cmap(name='jet')
#cmap = get_cmap(name='rainbow')

if False:
    for idx, ntau in enumerate(ntaus):
        #c = plt.plot([], [], label=f'ntau {ntau}')[0].get_color()
        i = np.linspace(0, 1, num=len(ntaus))[idx]
        c = cmap(i)
        plt.plot([], [], color=c, label=f'ntau {ntau}')
        colors.append(c)
        
    for style, order in zip(styles, orders):
        plt.plot([], [], style, color='gray', label=f'order {order}')
else:
    for idx, order in enumerate(orders):
        #c = plt.plot([], [], label=f'ntau {ntau}')[0].get_color()
        i = np.linspace(0, 1, num=len(orders))[idx]
        c = cmap(i)
        #plt.plot([], [], color=c, label=f'order {order}')
        colors.append(c)

    if False:
        for style, alpha, ntau in zip(styles, alphas, ntaus):
            e = int(np.log2(ntau))
            plt.plot([], [], '-' + style, color='k',
                     #label=r'$N_\tau =' + f'{ntau}$',
                     label=r'$N_\tau = 2^{' + f'{e}' + r'}$',
                     markersize=4, alpha=alpha)

        
#for style, alpha, order in zip(styles, alphas, orders):
#    for color, ntau in zip(colors, ntaus):
for color, order in zip(colors, orders):
    for style, alpha, ntau in zip(styles, alphas, ntaus):
        dfc = df[ df['ntau'] == ntau ]
        dfc = dfc[ dfc['order'] == order ]
        dfc = dfc.sort_values(by='N_samples')
        
        plt.plot(
            dfc['N_samples'], dfc['error'],
            '-', alpha=alpha, color=color, markersize=4,
            )

        N = np.array(dfc['N_samples'])
        e = np.array(dfc['error'])
        #print(type(e))
        if len(e) > 0:
            plt.plot(
                N[-1], e[-1],
                #style,
                'o',
                alpha=alpha, color=color,
                #markersize=4,
                )


import matplotlib as mpl
gc = mpl.rcParams['grid.color']
opts = dict(color=gc, alpha=1.0, lw=1.0, zorder=-10)

x = np.logspace(1.5, 4.5, num=10)
plt.plot(x, 1/x, '--', **opts)

x = np.logspace(2.5, 5.5, num=10)
plt.plot(x, 100/x, '--', label='$\propto 1/N_{qMC}$', **opts)
            
plt.legend(
    fontsize=7, ncol=2, loc='lower left',
    labelspacing=0.1,
    columnspacing=0.75,
    )
plt.loglog([], [])
#plt.axis('equal')
#plt.xlim([1e1, 1e7])
plt.grid(True)
#plt.xticks([1e2, 1e4, 1e6])
plt.xticks([1e2, 1e3, 1e4, 1e5])

plt.tick_params(labelleft=False, left=False)

plt.xlabel('$N_{qMC}$')
plt.ylabel(r'$|G(\beta/2) - G_{exact}(\beta/2)|$')

# -- Best convergence plot

errors = []
for order in orders:
    dfc = df[ df['order'] == order ]
    dfc = dfc[ dfc['ntau'] == np.max(dfc['ntau']) ]
    dfc = dfc[ dfc['N_samples'] == np.max(dfc['N_samples']) ]
    errors.append(dfc['error'][0])

# -- BEGIN DATA

# A Tensor Train Continuous Time Solver for Quantum Impurity Models
# https://arxiv.org/abs/2303.11199
# figure 3

from io import StringIO
data = np.loadtxt(StringIO("""0.21660649819494537, 0.4016485041281371
2.0036101083032483, 0.23608963626703663
4.061371841155235, 0.12814223889440857
6.010830324909748, 0.06772815277441427
8.014440433212997, 0.033054513475732496
10.072202166064983, 0.014505692324279153
12.075812274368232, 0.005723903557562126
14.079422382671481, 0.0019258186341850783
16.028880866425993, 0.0005379838403443692
18.08664259927798, 0.00012151094113758904
20.03610108303249, 0.000024030855432419302
22.03971119133574, 0.0000037417685733024595
24.09747292418773, 4.349722373637536e-7
26.046931407942242, 6.089973244419447e-8"""), delimiter=',')
o, e = data.T
o = np.round(o)
ttd = dict(orders=o, errors=e)

# -- END DATA

#plt.figure()
#plt.subplot(*subp); subp[-1] += 1

plt.subplot(gs[1, 0], sharey=ax)

plt.text(-0.085, 0.9, "(b)", size=9, ha="center", va="center", transform=ax.transAxes)

l = plt.plot(orders, errors, '-k',
             #label='bold (qMC)',
             alpha=0.75)

color = l[0].get_color()
#plt.plot([], [], 'o-', label='Bold (qMC)', alpha=0.75, color=color)
plt.plot([], [], 'o-', label='Inchworm', alpha=0.75, color=color, markerfacecolor='gray', markeredgecolor='gray')

plt.plot([], [], '-s',
         label='Bare (TTD)', color='k', markerfacecolor='gray', markeredgecolor='gray', alpha=0.75)

n = 12
plt.plot(ttd['orders'][:n], ttd['errors'][:n], '-',
         #label='Bare (TTD)',
         color='k', alpha=0.75)

m = [0] + list(range(4, 12))
plt.plot(ttd['orders'][m], ttd['errors'][m], 's',
         color='gray', alpha=0.75)

for color, order in zip(colors, orders):
    idx = list(orders).index(order)
    plt.plot(orders[idx], errors[idx], 'o', color=color, alpha=0.75)

for color, order in zip(colors, orders):
    if order in set([2,4,6]):
        idx = list(ttd['orders']).index(order)
        plt.plot(ttd['orders'][idx], ttd['errors'][idx], 's',
                 color=color, alpha=0.75)


plt.legend(loc='lower left', fontsize=7)
plt.semilogy([], [])

#plt.ylim(bottom=1e-8)
#plt.ylim([1e-8, 1e0])

#plt.xlim([0, 30])
plt.xlim([0, 20])
plt.xticks([0, 5, 10, 15])
plt.ylim(bottom=1e-5)

plt.grid(True)
plt.xlabel('Order')
#plt.ylabel(r'$|G(\beta/2) - G_{exact}(\beta/2)|$')
plt.ylabel(r'$\max_{\tau}|G(\tau) - G_{exact}(\tau)|$', labelpad=-2)


#plt.subplot(*subp); subp[-1] += 1

plt.subplot(gs[0, :])

plt.text(-0.53, 1.18, "(a)", size=9, ha="center", va="center", transform=ax.transAxes)

for color, order in zip(colors, orders):
    dfc = df[ df['order'] == order ]
    dfc = dfc[ dfc['ntau'] == np.max(dfc['ntau']) ]
    dfc = dfc[ dfc['N_samples'] == np.max(dfc['N_samples']) ]

    d = dfc
    plt.plot(d['tau'][0], d['gf'][0].imag,
             color=color, alpha=0.75, label=f'Order {order}')

plt.plot(d['tau'][0], d['gf_ref'][0].imag, ':', color='y', alpha=1., label='Exact')

plt.legend(loc='best', fontsize=7, ncol=2)
plt.ylim(bottom=0)
plt.grid(True)
plt.xlabel(r'$\tau$', labelpad=-10)
plt.ylabel(r'$G(\tau)$')
#exit()

#plt.tight_layout()
plt.savefig('figure_gf_convergence.pdf')
#plt.show();
exit()












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
