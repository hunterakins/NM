import timing
import numpy as np
import general as g
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize, Bounds
from NM import normal_mode as nm
import ssp_helpers as sh



if __name__ == '__main__':
    f= 150
    zs,rs = 36, 0
    rho1, rho2 = 1000, 1800
    D = 100
    ## SSP STUFF
    cw = 1500
    cb = 1800 
    rhof = lambda z: np.piecewise(z, [z <= D, z > D], [rho1, rho2])
    sspf = lambda z: np.piecewise(z, [z <= D, z > D], [cw, cb])
    med = g.Medium(D, rhof, cb, sspf)
    p = nm.PekerisModel(f, med)
    mod_out = p.getNM()
    r = np.arange(0, 3000, 10)
    z = np.arange(0,100, 1)
    tl = mod_out.TL(zs, z, r)
    plt.plot(r, tl[36,:])
    plt.gca().invert_yaxis()
    plt.show()
