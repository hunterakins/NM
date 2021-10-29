from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import scipy.interpolate as inter
from scipy.misc import derivative as deriv

'''
create gaussian disturbance on a profile given a weighted with depth distribution f(z)
f(z) specifies the variance of the disturbance at depth z
for example, if f(z) = delta(z-z0), then the disturbance only occurs at depth z0 and has variance 1

idea is to generate 100 samples or so according to this distribution
Then decompose the 100 samples into an EOF
then create a specific instance of the disturbance
represent it in my EOF basis
run my inversion to try and find the components in the EOF basis
'''

def gauss_bulge(zc, var, amp):
    return lambda z: amp*np.exp(-(z - zc)*(z-zc)/var)


def get_water_sspf(ssp, D):
    depths = np.linspace(0, D, len(ssp)) # depths for interpolate
    return interp1d(depths, ssp)
# given evenly spaced ssp through water column of depth d, give me interpolated fun
def get_sspf(ssp, D, cb):
    depths = np.linspace(0, D, len(ssp)) # depths for interpolate
    ssp_meanf = lambda z: np.piecewise(z, [z <= D, z>D], [interp1d(depths, ssp), cb])
    return ssp_meanf

def get_sspf_sec_deriv(ssp, D, cb):
    depths = np.linspace(0, D, len(ssp)) # depths for interpolate
    ssp_spline = inter.splrep(depths, ssp)
    ssp2_vals = np.array(inter.splev(depths, ssp_spline, der=2)) # second deriv
    ssp2_vals = ssp2_vals.reshape((len(ssp2_vals)))
    ssp2f = interp1d(depths, ssp2_vals)
    ssp_sec_deriv = lambda z: np.piecewise(z, [z <= D, z>D], [ssp2f, 0])
    return ssp_sec_deriv
    

def get_disturb(fz, domain):
    zweights = fz(domain)
    disturb = np.random.randn(1)
    deltac = zweights*disturb
    return deltac

def ssp_collection(num_runs, fz, domain):
    ssps = []
    zweights = fz(domain)
    ssps = np.zeros((len(domain), num_runs))
    for i in range(num_runs):
        deltac = get_disturb(fz, domain)
        ssps[:,i] = deltac
    return ssps


def eof(ssps):
    C = np.matmul(ssps, np.transpose(ssps)) # covariance
    w,v = np.linalg.eigh(C)
    inds = np.argsort(abs(w))[::-1] # sort them in largest to smallest order
    w = w[inds]
    w = w.reshape((len(w),1))
    v = v[:, inds]
#    E = np.transpose(w*(v.transpose()))
    return w, v, C
    
# get a list of specific instance of disturbance decomposed into eof basis
def get_disturb_e(fz, E, z, num_disturbs):
    deltac_list = []
    A_list = []
    for i in range(num_disturbs):
        deltac = get_disturb(fz, z)
        A = np.matmul(np.linalg.inv(E), deltac)
        deltac_list.append(deltac)
        A_list.append(A)
    return deltac_list, A_list






        
