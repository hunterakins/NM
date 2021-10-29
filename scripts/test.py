import timing
import numpy as np
import NM.general as g
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize, Bounds
from NM import normal_mode as nm
import NM.ssp_helpers as sh



if __name__ == '__main__':
    zs,rs = 50,2000
    zr = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65, 70.0, 75, 80]
    rho1, rho2 = 1000, 1500
    D = 100

    ## SSP STUFF
    deltac = 1
    cb = 1600
    cbguess = 1600# - deltac
    ssp = np.array([1500, 1500, 1500, 1500])
    ssp_guess = ssp + deltac
    sspf = sh.get_sspf(ssp, D, cb)
    rhof = lambda z: np.piecewise(z, [z <= D, z > D], [rho1, rho2])
    med = g.Medium([D], rhof, cb, sspf)
    f = 100
    p = nm.PekerisModel(f, med)
    mod_out = p.getNM()
#    mod_out.add_atten(10)
    
    gauss_env = sh.gauss_bulge(.1, 23, 40)
    
    pos_sspf = lambda z: sspf(z) + abs(gauss_env(z))
    neg_sspf = lambda z: sspf(z) - abs(gauss_env(z))

    
    pos_med = g.Medium([D], rhof, cb, pos_sspf)
    neg_med = g.Medium([D], rhof, cb, neg_sspf)

    pos_p = nm.PekerisModel(f, pos_med)
    neg_p = nm.PekerisModel(f, neg_med)
    pos_mod = pos_p.getNM()
    neg_mod = neg_p.getNM()
    
    z = np.arange(0.0, 150, 1)
    r = np.arange(1.0, rs, 10)

    
    vals = mod_out.greens(50, z,r)
    vals2 = np.flip(mod_out.greens(50, z, r), axis=1)
    vals = mod_out.line_kernel(vals, vals2, z)
    test_vals = mod_out.line_greens(50, zr, np.array([1000]))
    print(test_vals)
    plt.plot(test_vals.real)
    plt.plot(test_vals.imag)
    plt.show()
    thing = mod_out.line_greens(50, np.linspace(0,200, 201), np.linspace(0, 2000, 201)  )
    plt.contourf(abs(thing))
    plt.show()
    plt.plot((thing[:,-1]).imag)
    plt.show()
    
'''
    pos_vals = pos_mod.greens(50, z,r)
    pos_vals2 = np.flip(pos_mod.greens(50, z, r), axis=1)
    pos_vals = pos_mod.line_kernel(pos_vals, pos_vals2, z)

    neg_vals = neg_mod.greens(50, z,r)
    neg_vals2 = np.flip(neg_mod.greens(50, z, r), axis=1)
    neg_vals = neg_mod.line_kernel(neg_vals, neg_vals2, z)


    vals = vals[:,5:-5]
    pos_vals = pos_vals[:,5:-5]
    neg_vals = neg_vals[:,5:-5]

    fig, axes = plt.subplots(3,1)
    fig.suptitle("Example kernel for source and receiver at 50 m\n f = 1kHz 1 km range")

    plt.subplot(4,1,1)
    levels = np.linspace(np.min(vals), np.max(vals)/4, 20)
    plt.contourf(r[5:-5], z, vals, levels=levels)
    plt.gca().invert_yaxis()

    plt.subplot(4,1,2)
    levels = np.linspace(np.min(pos_vals), np.max(pos_vals)/4, 20)
    plt.contourf(r[5:-5], z, pos_vals, levels=levels)
    plt.gca().invert_yaxis()
    
    plt.subplot(4,1,3)
    levels = np.linspace(np.min(neg_vals), np.max(neg_vals)/4, 20)
    plt.contourf(r[5:-5], z, neg_vals, levels=levels)
    plt.gca().invert_yaxis()


    plt.subplot(4,1,4)
    dtdc = (pos_vals - neg_vals) / .1
    levels = np.linspace(np.min(dtdc),  np.max(dtdc), 20)
    plt.contourf(r[5:-5], z, dtdc, levels=levels)
    plt.gca().invert_yaxis()
    plt.show()

    print("RMS error: ", np.sqrt(np.sum(np.square(abs(dtdc))) / len(z) / len(r)))
'''
