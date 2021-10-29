import timing
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize, Bounds
from NM import normal_mode as nm
from NM import general as g
from NM import ssp_helpers as sh
import dill as dill


if __name__ == '__main__':
    f = 3000
#    f= 45
    zs,rs = 150,100000
    #zr = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65, 70.0, 75, 80]
    zr = [150]
    rho1, rho2 = 1000, 1500
    D = 5000
    ## SSP STUFF
    cb = 1800
    ztild = lambda z: 2*(z-1300)/1300
    c0 =1500
    eps =0.00737
    water_sspf = lambda z: c0*(1+eps*(ztild(z) - 1 + np.exp(-ztild(z))))
    sspf = lambda z: np.piecewise(z, [z<= D, z>D], [water_sspf, cb])

    plt.plot(sspf(np.linspace(1, 5000, 100)))
    plt.show()
    
    alpha1 = .5 # in db/lambda

    k = 2*np.pi*f/ cb

    alpha1 = k*alpha1/54.86 # convert dB/lambda to wavenumber atten

    #sspf = sh.get_sspf(ssp, D, cb)
    rhof = lambda z: np.piecewise(z, [z <= D, z > D], [rho1, rho2])
    alphaf = lambda z: np.piecewise(z, [z <= D, z > D], [0, alpha1])

    med = g.Medium([D], rhof, cb, sspf)
    p = nm.PekerisModel(f, med)
    run=False
    if run == True:
        mod_out = p.getNM()

        with open(str(f) + '_nm.dill', 'wb') as thing:
            dill.dump(mod_out, thing)


    with open(str(f) + '_nm.dill', 'rb') as thing:
        mod_out = dill.load(thing)
    kr = np.array(mod_out.kr_prop)
    #print(len(mod_out.kr_prop))
    plt.plot(kr)
    plt.show()


    dkdn = kr[1:] - kr[:-1]
    plt.plot(2*np.pi/dkdn.real)
    plt.show()
    

    z = np.arange(0.0, 5000, 20)
    r = np.arange(1.0, rs, 100)

    
    vals = mod_out.greens(50, z,r)
    mod_out.add_var_atten(alphaf)
    vals1 = mod_out.greens(50, z,r)


    vals = vals[:,5:-5]
    vals1 = vals1[:,5:-5]

    fig, axes = plt.subplots(2,1)

    plt.subplot(2,1,1)
    vals = abs(vals)
    vals1 = abs(vals1)
    levels = np.linspace(np.min(vals), np.max(vals)/4, 20)
    plt.contourf(r[5:-5], z, abs(vals), levels=levels)
    plt.gca().invert_yaxis()


    plt.subplot(2,1,2)
    levels = np.linspace(np.min(vals1), np.max(vals1)/4, 20)
    plt.contourf(r[5:-5], z, abs(vals1), levels=levels)
    plt.gca().invert_yaxis()
    plt.show()
