from timeit import default_timer as timer
import numpy as np
from NM import normal_mode as nm
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import NM.ssp_helpers as sh
from matplotlib import rc


# single frequency source, multiple receiver positions
def get_err_vec(grec_guess, grec_true, source_params, zr, dom):
    z,r = dom.z, dom.r     
    zmin,dz,dr = dom.zmin,dom.dz,dom.dr
    receiver_inds = [int((x - zmin)/dz+1) for x in zr]
    error_vec = grec_true[receiver_inds,0] - grec_guess[receiver_inds,0]
    error_vec = error_vec.reshape((len(error_vec), 1))
    return error_vec

def get_full_err_vec(grec_guess, grec_true_list, source_params, zr, dom):
    tmp_err_vec = get_err_vec(grec_guess, grec_true_list[0], source_params, zr, dom)
    num_freqs = len(grec_true_list)
    err_dims = np.shape(tmp_err_vec)
    err_spacer = err_dims[0]
    full_err_dims = num_freqs*err_spacer, 1
    full_err_vec = np.zeros(full_err_dims, dtype=np.complex128)
    full_err_vec[0:err_spacer, 0] = tmp_err_vec
    for i in range(1, num_freqs):
        tmp_err_vec = get_err_vec(grec_guess, grec_true_list[i], source_params, zr, dom)
        full_err_vec[i*err_spacer:(i+1)*err_spacer] = tmp_err_vec
    return full_err_vec

def get_green(medium, source_params, dom, off_axis=False, range_ind=True):
    p = nm.PekerisModel(source_params.f, medium)
    mod = p.getNM() # generate normal mode solution for medium
    g = mod.line_greens(source_params.zs, dom.z, dom.r)
    if off_axis is  True:
        g = np.flip(g, axis=1) # place it remotely
    return mod, g

def plot_ssp(ssp, dom):
    plt.plot(ssp(dom.z),dom.z)
    plt.title("SSP for Pekeris waveguide")
    plt.xlabel("c (m/s)")
    plt.ylabel("Depth (m)")
    plt.gca().invert_yaxis()
#    plt.plot(ssp_meanf(dom.z))
    plt.show()

# kernel for single frequency
def twod_kern(mean_med, zr, source_params, dom, range_ind=True):
    # SET PARAMETERS FOR SOLUTION GRID
    z,r = dom.z, dom.r     
    # if range ind is true I can collapse the rows of the kernel into a single sum
    if range_ind == True:
        kern = np.zeros((len(zr), len(z)), dtype=np.complex128)
    # if not I have to keep the whole thing
    else:
        kern = np.zeros((len(zr), len(r)*len(z)), dtype=np.complex128)
    guess_model, grec_guess = get_green(mean_med, source_params, dom, off_axis=True)
    for zrr,i in zip(zr, range(len(zr))):
        gr=guess_model.line_greens(zrr, z, r) # get greens function as if source were at receiver lkocation
        tmpkern = guess_model.line_kernel(gr, grec_guess, z)
        if range_ind == True:
            tmpkern = np.sum(tmpkern, axis=1) # sum columns since no range dep.
            tmpkern = tmpkern.reshape(1, len(tmpkern))
        else:
            tmpkern = tmpkern.reshape((1, len(r)*len(z)))
        kern[i,:] = tmpkern
    return kern, grec_guess

# kernel for a vector of frequencies    
# return matrix whose ith row is kernel for ith freq.
# also return list of greens functions for the frequencies to calc. error vectors
def twod_full_kern(medium, zr, source_params, dom,  range_ind=True):
    z,r = dom.z, dom.r     
    f, rs, zs = source_params.f, source_params.rs, source_params.zs
    # run on first frequency to get dimensions of kernel
    f0 = f[0]    
    tmp_source_params = SourceParams(f0, rs, zs)
    tmpkern, tmpgrn = twod_kern(medium, zr, tmp_source_params, dom,range_ind=True)
    grns_list = [tmpgrn]
    dims = np.shape(tmpkern)
    # use dimensions to initialize matrices for the kernel
    fulldims = dims[0]*len(f), dims[1]
    fullkern= np.zeros(fulldims, dtype=np.complex128)
    spacer = dims[0]
    fullkern[0:spacer,:] = tmpkern
    for ff,i in zip(f[1:], range(1,len(f))):
        tmp_source_params = SourceParams(ff, rs, zs)
        tmpkern, tmpgrn = twod_kern(medium, zr, tmp_source_params, dom, range_ind=True)
        fullkern[i*spacer:(i+1)*spacer, :] = tmpkern
        grns_list.append(tmpgrn)
    return fullkern, grns_list

class SourceParams:
    def __init__(self, f, rs, zs):
        self.f =f 
        self.rs = rs
        self.zs = zs
        self.omega = 2*np.pi*self.f

    def disp(self):
        print("Freq, range, depth: ", self.f, self.rs, self.zs)

# basic pekeris waveguide
# ssp is a set of points evenly spaced through the water
# solid flui dinterface at D
# cb is sound speed in bottom (assumed constant)
# rho1 is water density, rho2 is solid density

def get_pek_sspf(ssp, D, cb):
    cwf = interp1d(np.linspace(0, D, len(ssp)), ssp)
    ssp_w = lambda z: cwf(z)
    sspf = lambda x: np.piecewise(x, [x<D, x>=D], [ssp_w, cb])
    return sspf

class Medium:
    def __init__(self, D, rhof, cb, sspf, c0=None):
        self.D = D
        self.cb = cb
        self.sspf = sspf
        self.rhof = rhof
        self.c0 = c0

class Domain:
    def __init__(self, zmin, zmax, dz, rmin, rmax, dr):
        self.zmin = zmin
        self.zmax = zmax
        self.dz = dz
        self.rmin = rmin
        self.rmax = rmax
        self.dr =dr
        self.z = np.arange(zmin, zmax+dz, dz)
        self.r = np.arange(rmin, rmax+dr, dr)

class GaussPertParams:
    def __init__(self, mean_depth, var, amp, num_profs):
        self.mean_depth = mean_depth
        self.var = var
        self.amp = amp
        self.num_profs = num_profs

    def get_gauss_bulge(self):
        return sh.gauss_bulge(self.mean_depth, self.var, self.amp)

    # generate a collection of perturbation drawn from distribution on domain 
    def get_collection(self, domain):
        bulge = self.get_gauss_bulge() 
        return sh.ssp_collection(self.num_profs, bulge, domain.z)

class GaussianModelInput:
    ''' model input that tests disturbances of gaussian envelope and gaussian distribution from mean
    In general, you can have sources with a list of frequencies
    and a list of receiver locales
    '''
    def __init__(self,source_params,zr, mediumparams, cbpert, dz, dr, gaussparams, num_perts):
        self.source_params = source_params  
        self.zr = zr 
        self.medium = mediumparams
        self.cbpert = cbpert
        zmin = 0
        zmax = 1.5*self.medium.D  # go to 1.5 times the depth as rule of thumb
        rmin = dr
        rmax = self.source_params.rs 
        self.domain = Domain(zmin, zmax, dz, rmin, rmax, dr)
        self.pert_params = gaussparams
        self.num_perts = num_perts
        start = timer()
        self.E, self.pert_list, self.coeff_list = get_perts(self.pert_params, self.num_perts, self.domain)
        end = timer()
        print("Time to get " + str(num_perts) + " perturbed ssps:",end - start)


class InvOutput:
    ''' store results of inversion:
    frequencies used, source position, receiver positions perturbation, 
    EOF matrix, representation of perturbation in EOF basis
    estimate of perturbation from inversion
    '''
    def __init__(self, source_params, zr, c_errs, dg, E, pert, coeffs, est_coeffs, est_err):
        self.source_params =source_params
        self.zr = zr
        self.c_errs = c_errs
        self.dg = dg
        self.E = E[:,0]
        self.pert = pert
        self.coeffs = coeffs[:4] # coefficients of the pert in the EOF basis
        self.est_coeffs = est_coeffs[:4]
        self.est_errs = est_err # estimated err from estimated coeff
#        self.full_kern = full_kern
        

def get_perts(pert_params, num_perts, dom):
    # collection of peturbations to do EOF with
    coll = pert_params.get_collection(dom)
    # get eof of the collection
    w, v,C = sh.eof(coll)
    E = v
    pert_list, coeff_list = sh.get_disturb_e(pert_params.get_gauss_bulge(), E, dom.z, num_perts) 
    return E, pert_list, coeff_list

         

def gaussian_inv_run(inp):
    NUM_COEFFS = 1 # number of EOF basis coefficients to keep
    dom = inp.domain # grid for evaluation
    mean_med = inp.medium # mean medium parameters
    source_params = inp.source_params
    ssp_meanf = mean_med.sspf
    mean_prof = ssp_meanf(dom.z)
    
    E, pert_list, coeff_list = inp.E, inp.pert_list, inp.coeff_list

    start = timer()
    # get kernel using background sound speed along with greens function for each frequency
    full_kern, grns_list = twod_full_kern(mean_med, inp.zr, source_params, dom, range_ind=True)
    end = timer()
    print("Kernel calculation time:", end - start) # Time in seconds, e.g. 5.38091952400282
    # run inversion for each perturbation in the list and for each freq.
    inv_results = []
    for pert, coeff in zip(pert_list, coeff_list):
        curr_kern = full_kern # create copy
        # interpolate it the perturation and apply it to the mean field to get a function for the ''true'' field
        pert_func = interp1d(dom.z, pert)
        ssp_truef = lambda z: ssp_meanf(z) + pert_func(z)
        # plot ssp
        # mean plus perturbation
        med_true = Medium(mean_med.D, mean_med.rhof, inp.cbpert, ssp_truef)
        # get real greens function
        tmp_source_params = SourceParams(source_params.f[0], source_params.rs, source_params.zs)
        mod_true, g_true = get_green(med_true, source_params, dom, off_axis=True)
         
        # compute error vector
        errs = get_err_vec(grns_list[0], g_true, source_params, inp.zr, dom)
        z, r =dom.z, dom.r
    
        dc = pert_func(z).reshape((len(pert_func(z)), 1))
        da = dom.dz*dom.dr
        dg = da*np.matmul(curr_kern, dc)
    #dg = da*np.sum(curr_kern, axis=1)*.1
        curr_kern = np.concatenate((curr_kern.real, curr_kern.imag))
        eof_basis_kern = da*np.matmul(curr_kern, E)
        eof_basis_kern = eof_basis_kern[:,0:NUM_COEFFS] # only need first couple of columns if it's truly sparse
        # inversion a la Wunsch DIASEP p. 2.95
        tinv = np.linalg.inv(np.matmul(np.transpose(eof_basis_kern), eof_basis_kern))
        errs = np.concatenate((errs.real, errs.imag))
        est_coeffs = np.matmul(tinv, np.matmul(np.transpose(eof_basis_kern), errs))
        est_errs = np.matmul(eof_basis_kern, est_coeffs)
        length = np.shape(errs)[0]//2
        c_errs = errs[:length] + complex(0,1)*errs[length:]
        saved_params = source_params
        results = InvOutput(saved_params, inp.zr, c_errs, dg, E, pert, coeff, est_coeffs, est_errs)
        inv_results.append(results)
    return inv_results

