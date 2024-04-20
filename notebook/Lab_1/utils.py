import numpy  as np
import pandas as pd
import copy 
from smt.sampling_methods import LHS

import matplotlib.pyplot as plt

from CFLib.euro_opt import np_euro_call
from CFLib.Heston   import Heston


def histo_params( x, title="None"):
    keys = list(x)
    LEN  = len(keys)
    fig, ax = plt.subplots(1,LEN, figsize=(12,4))
    if not title == None: fig.suptitle(title)
    for n in range(LEN):
        tag  = keys[n]
        lo   = np.min(x[tag])
        hi   = np.max(x[tag])
        bins = np.arange(lo, hi, (hi-lo)/100.)
        ax[n].hist(x[tag], density=True, facecolor='g', bins=bins)
        ax[n].set_title(tag)
        n += 1
    plt.subplots_adjust(left=.05, right=.95, bottom=.10, top=.80, wspace=.50)
    plt.show()

def lhs_sampling(rand, NUM, bounds=None):

    # input : rand   = RandomState object
    #         NUM    = n° of instances
    #         bounds = def. of the hypercube
    # output: X      = dataframe as below
    
    mInt = (1 << 15)
    MInt = (1 << 16)
    kw = list(bounds)

    limits = np.empty( shape=(0,2) )
    for k in kw: 
        limits = np.concatenate((limits, [bounds[k]]), axis=0)

    # define a sampler over the hypercube from 'limits' variable ...
    sampling = LHS(xlimits=limits, random_state=rand.randint(mInt,MInt))
    # ... and draw 'NUM' instances
    x = sampling(NUM)

    # rearrange the samples in a dataframe whose columns are 
    # identified by the keywords of input dictionary 'bounds'
    X = pd.DataFrame()
    for n in range(len(kw)):
        tag = kw[n]
        X[tag] = x[:,n]

    return X

def gen_BnS(NUM, x):
    
    # input : NUM = n° of instances
    #         x   = df containing option specifics T and K and BS vol 
    # output: y   = df of the type [x, BS price]

    S0     = np.full(NUM, 1.0, dtype = np.double)
    r      = np.full(NUM, 0.0, dtype = np.double)
    price  = np_euro_call(S0, r, x["T"], x["Sgma"], x["Strike"])
        
    y = pd.DataFrame(x)
    y["Price"] = price

    return y

def gen_Hes(NUM, x):
    
    # input : NUM = n° of instances
    #         x   = df containing option specifics T and K and Heston model parameters 
    # output: y   = df of the type [x, opt price]

    l = 40.

    y = copy.deepcopy(x)
    y["Price"] = np.full(NUM, 0.0, dtype=np.double)

    pCount = 0

    for m in range(NUM):

        So = 1.0
        K = y["Strike"][m]
        T = y["T"][m]

        model = Heston( lmbda = y["k"][m]
                      , nubar = y["theta"][m]
                      , eta   = y["sigma"][m]
                      , nu_o  = y["r0"][m]
                      , rho   = y["rho"][m]
                      )
        fwPut = model.HestonPut( So     = So
                               , Strike = K
                               , T      = T 
                               , Xc     = l*np.sqrt(T)
                               )

        if fwPut < max(K-So,0.): 
            pCount += 1
            continue

        y["Price"][m] = fwPut
        
    return y

def add_noise(rand, Xv, eps):
    
    # input : rand = RandomState object
    #         Xv   = df of the type [mdl params + T and K, Price ] 
    #         eps  = scale factor for gaussian noise 
    # output: X    = df of the type [mdl params + T and K, Price*]
    #                where Price* is polluted prices
                    
    X  = Xv.copy()
    
    xl = np.min(X["Price"])
    xh = np.max(X["Price"])
    
    # pollute prices by a normal random variable that scales with eps
    xi = rand.normal(loc = 0.0, scale = eps*(xh-xl), size=X.shape[0])
    X["Price"] += xi
    return X

def plot_nnpredict_45(y, y_hat, title, ax=None):

    if ax == None: return
    
    ax.plot(y, y, color='red')
    ax.plot(y, y_hat, ".")
    ax.set_title("%s" %(title))
    ax.set_xlabel("true")
    ax.set_ylabel("predicted")