import numpy as np
from scipy.stats import norm

def chf_heston(params, u, t):
        
    kappa = params[0]
    v_bar = params[1]
    nu    = params[2]
    v0    = params[3]
    rho   = params[4]
    
    beta  = kappa - rho*nu*1j*u
    alpha = -.5* u**2 -0.5*1j*u
    zeta  = 0.5*nu**2

    d = np.sqrt(beta**2-4*alpha*zeta)

    rm = (beta-d)/nu**2
    rp = (beta+d)/nu**2

    g = rm/rp

    C = rm*t - 2/nu**2*np.log((1-g*np.exp(-d*t))/(1-g))
    D = rm*(1-np.exp(-d*t)) / (1-g*np.exp(-d*t))

    phi = np.exp(v_bar*kappa*C + v0*D)
    
    return phi


def SINC_discFT(S,t,K,IR,DY,params,Xc,N,CP):
    
    n = np.arange(1, N/2 + 1, 2)
    wn = n/(2*Xc)

    k = np.log(K/S) - (IR-DY)*t

    sn = np.zeros((len(k), len(wn)))
    cs = np.zeros((len(k), len(wn)))

    for i in range(len(k)):
        sn[i,:] = np.sin(2*np.pi*k[i]*wn)
        cs[i,:] = np.cos(2*np.pi*k[i]*wn)

    f1 = chf_heston(params, 2*np.pi*wn   , t)
    f2 = chf_heston(params, 2*np.pi*wn-1j, t)

    ad1 = np.zeros(len(k))
    ad2 = np.zeros(len(k))

    for i in range(len(k)):
        ad1[i] = np.sum((sn[i,:]*np.real(f1) - cs[i,:]*np.imag(f1)) / n)
        ad2[i] = np.sum((sn[i,:]*np.real(f2) - cs[i,:]*np.imag(f2)) / n)

    conP = K*np.exp(-IR*t)*(0.5+(2/np.pi)*ad1)
    aonP = S*np.exp(-DY*t)*(0.5+(2/np.pi)*ad2)

    put = conP - aonP

    call = put+S*np.exp(-DY*t)-K*np.exp(-IR*t)

    if CP== 1:
        price = call
    elif CP==-1:
        price = put

    return price


def BS_closedForm(S, K, t, IR, DY, sigma, CP, tipo):
    
    x = np.log(S/K) + (IR-DY)*t
    sig = sigma*np.sqrt(t)
    d1 = x/sig + .5*sig
    d2 = d1 - sig
    pv = np.exp(-IR*t)

    put = -S*np.exp(-DY*t)*norm.cdf(-d1) + pv*K*norm.cdf(-d2)
    call = S*np.exp(-DY*t)*norm.cdf( d1) - pv*K*norm.cdf( d2)

    conP = pv*K*norm.cdf(-d2)
    conC = pv*K*norm.cdf( d2)

    aonP = np.exp(-DY*t)*S*norm.cdf(-d1)
    aonC = np.exp(-DY*t)*S*norm.cdf( d1)

    if CP*tipo== 1:
        out = call
    elif CP*tipo==-1:
        out = put
    elif CP*tipo== 2:
        out = conC
    elif CP*tipo==-2:
        out = conP
    elif CP*tipo== 3:
        out = aonC
    elif CP*tipo==-3:
        out = aonP

    return out


def BSImpliedVol(S, K, t, r, OptPrice, CP):
    
    if CP== 1:
        C = OptPrice
    elif CP==-1:
        P = OptPrice
        C = P + S - np.exp(-r*t) * K

    sigmaL = 1e-10
    CL = BS_closedForm(S, K, t, r, 0, sigmaL, 1, 1)

    sigmaH = 10
    CH = BS_closedForm(S, K, t, r, 0, sigmaH, 1, 1)

    while np.mean(sigmaH-sigmaL)>1e-10:
        sigma = (sigmaL+sigmaH) / 2
        CM = BS_closedForm(S, K, t, r, 0, sigma, 1, 1)

        CL = CL + (CM< C)*(CM-CL)
        sigmaL = sigmaL + (CM< C)*(sigma-sigmaL)

        CH = CH + (CM>=C)*(CM-CH)
        sigmaH = sigmaH + (CM>=C)*(sigma-sigmaH)

    return sigma
