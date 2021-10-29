import timing
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize, Bounds
from NM import normal_mode as nm
from NM import general as g
from NM import ssp_helpers as sh



if __name__ == '__main__':
    f = 40
#    f= 45
    zs,rs = 50,10000
    zr = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65, 70.0, 75, 80]
    rho1, rho2, rho3 = 1000, 1500, 5800
    D1 = 100
    D2 = 110
    ## SSP STUFF
    deltac = 1
    cb1, cb2 = 1600, 1800
    ssp = np.array([1500.0, 1490.0, 1480, 1480, 1479, 1478, 1480, 1481, 1482, 1483, 1485, 1486, 1489, 1492, 1492, 1499, 1503, 1503, 1503, 1504], dtype=np.float64)
    alpha1 = .5 # in db/lambda
    alpha2 = .06
    k1 = 2*np.pi*f/ cb1
    k2 = 2*np.pi*f/ cb2
    alpha1 = k1*alpha1/54.86 # convert dB/lambda to wavenumber atten
    alpha2 = k2*alpha2/54.86 
    sspf1 = sh.get_sspf(ssp, D1, cb1)
    sspf = lambda z: np.piecewise(z, [z < D2, z >= D2], [lambda z: sspf1(z), cb2])
    rhof1 = lambda z: np.piecewise(z, [z <= D1, z > D1], [rho1, rho2])
    alphaf1 = lambda z: np.piecewise(z, [z <= D1, z > D1], [0, alpha1])
    rhof = lambda z: np.piecewise(z, [z < D2, z >= D2], [lambda z: rhof1(z), rho3])
    alphaf = lambda z: np.piecewise(z, [z < D2, z >= D2], [lambda z: alphaf1(z), alpha2])
    med = g.Medium([D1,D2], rhof, cb1, sspf)
    p = nm.PekerisModel(f, med)
    mod_out = p.getNM()

    z = np.arange(0.0, 150, 1)
    r = np.arange(1.0, rs, 10)

    
    vals = mod_out.greens(50, z,r)
    mod_out.add_var_atten(alphaf)
    vals1 = mod_out.greens(50, z,r)


    vals = vals[:,5:-5]
    vals1 = vals1[:,5:-5]

    fig, axes = plt.subplots(2,1)

    plt.subplot(2,1,1)
    levels = np.linspace(np.min(vals), np.max(vals)/4, 20)
    plt.contourf(r[5:-5], z, abs(vals), levels=levels)
    plt.gca().invert_yaxis()


    plt.subplot(2,1,2)
    levels = np.linspace(np.min(vals1), np.max(vals1)/4, 20)
    plt.contourf(r[5:-5], z, abs(vals1), levels=levels)
    plt.gca().invert_yaxis()
    plt.show()
