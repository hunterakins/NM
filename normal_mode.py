import sys 
from timeit import default_timer as timer
import numpy as np
import scipy.linalg as lin
from scipy.interpolate import interp1d
from scipy.integrate import quad
from matplotlib import pyplot as plt

def mode_wavelength(kr, k):
    kz = np.sqrt(k*k - kr*kr)
    return 2 * np.pi / kz

class NMModel:
    '''
    contains all the information that characterizes a normal mode model:
    - medium characteristics
    - normal mode eigenvectors
    - normal mode eigenvalues (kr_prop[m] = krm -- range propagating wave numbers)
    '''

    def __init__(self, kr_prop, mode_vec, modef, medium, omega):
        self.kr_prop = kr_prop
        self.mode_vec = mode_vec # original eigenvector
        self.modef = modef # interpolated on the range of the model
        self.omega = omega
        self.medium = medium


    def TL_term(self, r, z, m, zs):
        func = self.modef[m] # get mth eigenfunctoin
        f_inter = self.modef # turn it into a function of z
        weight = f_inter(zs)
        hank = np.exp(complex(0,1) * self.kr_prop[m] * r) / np.sqrt(self.kr_prop[m]) 
        return pow(abs(weight*f_inter(z)*hank), 2) / np.sqrt(r)


    def TL(self, zs, z_range, r):
        thing = np.zeros((len(z_range), len(r)), dtype=np.complex128)
        funcs = self.modef
        sample = funcs[0]
        rho2 = self.medium.rhof(self.medium.D+1)
        sum_total = np.zeros((len(z_range), len(r)), dtype=np.complex128)
        for m in range(len(self.kr_prop)): # for each mode
            func = self.modef[m] # get the wavefunction
            krm = self.kr_prop[m] # get the mode numbers
            hank = np.exp(complex(0,1) * krm * r) / np.sqrt(krm)  # get the hankel component
            weight = func(zs) # get our weighting value
            range_vec = np.matrix(weight * hank) # this is the same for all values of z
            zvec = np.matrix(func(z_range)).transpose() # turn into column vec 
            val = zvec * range_vec 
            sum_total += val
        return -20*np.log10(1/rho2 *np.sqrt(2*np.pi / r)*abs(sum_total))

    def greens(self, zs, z_range, r):
        i = complex(0,1)
        thing = np.zeros((len(z_range), len(r)), dtype=np.complex128)
        funcs = self.modef
        sample = funcs[0]
        sum_total = np.zeros((len(z_range), len(r)), dtype=np.complex128)
        for m in range(len(self.kr_prop)): # for each mode
            func = self.modef[m] # get the wavefunction
            krm = self.kr_prop[m] # get the mode numbers
            hank = np.exp(i * krm * r) / np.lib.scimath.sqrt(krm)  # get the hankel component
            weight = func(zs) # get our weighting value
            range_vec = np.matrix(weight * hank) # this is the same for all values of z
            zvec = np.matrix(func(z_range)).transpose() # turn into column vec 
            val = zvec * range_vec 
            sum_total += val
        weight = np.nan_to_num(np.exp(-i*np.pi/4) * i / self.medium.rhof(zs) / np.sqrt(8*np.pi*r))
        #weight = np.nan_to_num(np.exp(-i*np.pi/4) * i / np.sqrt(8*np.pi*r))
        p = weight*sum_total
        return p

    def off_axis_greens(self, zs, z_range, r, rs, theta):
        i = complex(0,1)
        thing = np.zeros((len(z_range), len(r),len(theta)), dtype=np.complex128)
        funcs = self.modef
        sample = funcs[0]
        full_field = np.zeros((len(z_range), len(r), len(theta)), dtype=np.complex128)
#        rdiff = r*r + rs*rs - 2*rs*r*np.cos(theta) # law of cosines
        for j in range(len(theta)):
            ang = theta[j]
            rdiff = r*r + rs*rs - 2*rs*r*np.cos(ang) # law of cosines
            sum_total = np.zeros((len(z_range), len(r)), dtype=np.complex128)
            for m in range(len(self.kr_prop)): # for each mode
                func = self.modef[m] # get the wavefunction
                krm = self.kr_prop[m] # get the mode numbers
                hank = np.exp(i * krm * rdiff) / np.lib.scimath.sqrt(krm)  # get the hankel component
                weight = func(zs) # get our weighting value
                range_vec = np.matrix(weight * hank) # this is the same for all values of z
                zvec = np.matrix(func(z_range)).transpose() # turn into column vec 
                val = zvec * range_vec 
                sum_total += val
            weight = np.nan_to_num(np.exp(-i*np.pi/4) * i / self.medium.rhof(zs) / np.sqrt(8*np.pi*rdiff))
            p = weight*sum_total
            full_field[:,:,j] = p
        if len(theta) == 1:
            return full_field[:,:,0]
        return full_field
        
    def kernel(self, zr, zs, z_range, r, rs, theta):
        i = complex(0,1)
        thing = np.zeros((len(z_range), len(r),len(theta)), dtype=np.complex128)
        funcs = self.modef
        sample = funcs[0]
        full_field = np.zeros((len(z_range), len(r), len(theta)), dtype=np.complex128)
#        rdiff = r*r + rs*rs - 2*rs*r*np.cos(theta) # law of cosines
        cf = self.medium.sspf(z_range)
        cf = np.reshape(cf, (len(cf),1))
        cf = np.power(cf, -3)
        for j in range(len(theta)):
            ang = theta[j]
            rdiff = np.sqrt(r*r + rs*rs - 2*rs*r*np.cos(ang)) # law of cosines
            sum_total0 = np.zeros((len(z_range), len(r)), dtype=np.complex128)
            sum_total1 = np.zeros((len(z_range), len(r)), dtype=np.complex128)
            for m in range(len(self.kr_prop)): # for each mode
                func = self.modef[m] # get the wavefunction
                krm = self.kr_prop[m] # get the mode numbers
                hank0 = np.exp(i * krm * r) / np.lib.scimath.sqrt(krm)  # get the hankel component
                hank1 = np.exp(i * krm * rdiff) / np.lib.scimath.sqrt(krm)  # get the hankel component
                func = self.modef # turn it into a function of z
                weight0 = func(zr) # get our weighting value
                weight1 = func(zs) # get our weighting value
                range_vec0 = np.matrix(weight0 * hank0) # this is the same for all values of z
                range_vec1 = np.matrix(weight1 * hank1) # this is the same for all values of z
                zvec = np.matrix(func(z_range)).transpose() # turn into column vec 
                val0 = zvec * range_vec0 
                val1 = zvec * range_vec1 
                sum_total0 += val0
                sum_total1 += val1
            weight0 = np.nan_to_num(np.exp(-i*np.pi/4) * i / self.medium.rhof(zr) / np.sqrt(8*np.pi*r))
            weight1 = np.nan_to_num(np.exp(-i*np.pi/4) * i / self.medium.rhof(zs) / np.sqrt(8*np.pi*rdiff))
            p0 = weight0*sum_total0
            p1 = weight1*sum_total1
            p0 = np.multiply(cf, p0) # 1/c^3
            full_field[:,:,j] = np.multiply(p0, p1)
        full_field = full_field * - 2*pow(self.omega,2)
        return full_field

    #green's function for line source
    def line_greens(self, zs, z_range, r):
        i = complex(0,1)
        thing = np.zeros((len(z_range), len(r)), dtype=np.complex128)
        funcs = self.modef
        sample = funcs[0]
        sum_total = np.zeros((len(z_range), len(r)), dtype=np.complex128)
        for m in range(len(self.kr_prop)): # for each mode
            func = self.modef[m] # get the wavefunction
            krm = self.kr_prop[m] # get the mode numbers
            r_func = i/self.medium.rhof(zs)/2*np.exp(i * krm * abs(r)) / krm
            func = self.modef[m] 
            weight = func(zs) # get our weighting value
            range_vec = np.matrix(weight * r_func) # this is the same for all values of z
            zvec = np.matrix(func(z_range)).transpose() # turn into column vec 
            val = zvec * range_vec 
            sum_total += val
        p = sum_total
        return p

    def line_kernel(self, gr, gs,z_range):
        kern =np.multiply(gr, gs)
        kern = -2*self.omega*self.omega*kern
        cf = self.medium.sspf(z_range)
        cf = np.reshape(cf, (len(cf),1))
        cf = np.power(cf, -3)
        kern = kern*cf
        return kern

    def add_atten(self, alpha): # add attenuation to the modal wavenumbers
        # alpha is atten. coeff in the bottom in units of ?
        new_krm = [0]*len(self.kr_prop)
        for krm, psim, i in zip(self.kr_prop, self.modef, range(len(self.kr_prop))):
            num = np.square(psim(self.medium.D))*alpha*self.omega
            gamma_m = np.lib.scimath.sqrt(krm*krm - np.square(self.omega / self.medium.cb))
            den = 2*krm*gamma_m *self.medium.cb*self.medium.rhof(self.medium.D[-1]+1)
            delta_alpham = num / den
            krm_pert = np.sqrt(krm*krm + 2*complex(0,1)*delta_alpham*krm)
            new_krm[i] = krm_pert
        self.kr_prop = new_krm

    def add_var_atten(self, alphf): # add attenuation given as a function of z
        new_krm = [0]*len(self.kr_prop)
        D = self.medium.D # interface depths
        for krm, psim, i in zip(self.kr_prop, self.modef, range(len(self.kr_prop))):
            integrand = lambda z: np.square(psim(z))*self.omega*alphf(z)/(self.medium.sspf(z)*self.medium.rhof(z)) # 5.175 / 2i
            delta_alpham = quad(integrand, 1,D[0])[0] # water column contrib
            for j in range(len(D)-1): # for each layer, approximate the integral
                dz = .1
                dom = np.arange(D[j], D[j+1], dz)
                grid_integrand = integrand(dom)
                trap_heights = (grid_integrand[1:] +grid_integrand[:-1])/2
                integral = sum(trap_heights)*dz
                delta_alpham += integral
            ''' last contribution is an integral of expon. decay to infinity '''
            alpha = alphf(D[-1]+.1) # alpha of last layer
            num = np.square(psim(D[-1]))*alpha*self.omega
            gamma_m = np.lib.scimath.sqrt(krm*krm - np.square(self.omega / self.medium.cb))
            den = 2*gamma_m *self.medium.sspf(D[-1]+1)*self.medium.rhof(D[-1]+1)
            delta_alpham += num / den # add last layer contrib
            krm_pert = np.sqrt(krm*krm + 2*complex(0,1)*delta_alpham) 
            new_krm[i] = krm_pert
        self.kr_prop = new_krm
        

    def ug(self, m, omega): # group vel. of mth mode
        krm, mode = self.kr_prop[m], self.modef[m]
        integrand = lambda z: np.square(mode(z))/self.medium.rhof(z)/np.square(self.medium.sspf(z))
        integral = quad(integrand, 0, 1.5*self.medium.D)[0] # the first element is the vlaue, second is the error
        u_inv = integral* omega/krm
        return 1/u_inv

    def plot_ug(self, m):
        om  = np.linspace(self.omega - 20, self.omega+20, 100)
        f = om / 2 / np.pi
        plt.plot(f, self.ug(m, om))
        plt.show()
        
    def plot_first_last(self):
        vecs = self.mode_vec
        plt.plot(vecs[0])
        plt.plot(vecs[-1])
        plt.show()

class PekerisModel:
    def __init__(self, freq, medium):
        self.medium = medium
        self.freq = freq
        self.cw = medium.sspf # function that gives speed profile in water
        self.cb = medium.cb # bottom speed
        self.D  = medium.D
        z_tmp = np.linspace(1, self.D[0], 100) # up to first solid interface
        self.cmin = np.min(self.cw(z_tmp)) # used to bound the modes
        self.cmax = np.max(self.cw(z_tmp)) # used to bound the modes
        self.cinter = [self.cw(x) for x in self.D]
        self.rhow = medium.rhof(self.D[0]//2) # pekeris will have a piecewiserho func 
        self.rhob = medium.rhof(self.D[0]*1.5)
        self.omega = 2 * np.pi * np.array(freq)
        #self.k1 = self.omega /self.cmin # don't seem to use this
        #self.k2 = self.omega /self.cb
        self.z = None # grid of points extends D/2 into the lower fluid layer
        self.h = None
        self.kr_prop = None # modes...populate later
        self.eigenf = None # depth functions psi(z)
        self.summer = False 

    # get bounds of propagating modes
    def kr_bounds(self):
        f = self.freq
        cb =self.cb
        omega =self.omega
        kmin, kmax = omega / cb, omega / self.cmin
        if self.D[0] > 4000: #deep water just use max speed
            kmin = omega/self.cmax
        if kmax < kmin:
            raise ValueError("kmax is bigger than kmin, so bottom speed must be smaller than top speed")
        return kmin, kmax

    # return a numpy array with a reasonably sampled grid for highest mode
    # ten points per wavelength
    def discretize_depth(self):
        krmin = self.kr_bounds()[0]
        k = self.omega / self.cmin
        wavelength = mode_wavelength(krmin, k)
        dz = wavelength / 10
        N = int(self.D[0] // dz)
        z = np.arange(0, self.D[0], dz)
        for layer_ind in range(len(self.D)-1): # loop through layers to make sure they appear as interfaces
            layer_depth = self.D[layer_ind]
            next_layer_depth = self.D[layer_ind+1]
            zlayer = np.arange(layer_depth, next_layer_depth, dz) # z in the bottom
            z = np.concatenate((z, zlayer), axis=0)
        # last layer
        zlast = np.arange(self.D[-1], 2*self.D[-1]+dz, dz)
        if self.D[-1] > 4000: # deep water? Idk just trying hti sout
            zlast = np.arange(self.D[-1], 1.1*self.D[-1]+dz, dz)
        z = np.concatenate((z, zlast), axis=0)
        self.z = z
        self.h = dz
        return z,N
  
    def ea_numerov(self, j):
        depth = self.z[j]
        om = self.omega
        c, rho = self.medium.sspf(depth), self.medium.rhof(depth)
        denom = self.h * rho
        numer = (1 + pow(self.h, 2) * 1/12 * pow(om, 2) / pow(c,2))    
        ea = numer / denom
        return ea
 
    def da_inter_numerov(self, j):
        depth, h = self.z[j], self.h
        if depth not in self.D:
            raise ValueError("depth not on interface")
        om = self.omega  
        cu = self.medium.sspf(depth-.1)
        cb = self.medium.sspf(depth+.1)
        rhou = self.medium.rhof(depth-.1)
        rhob = self.medium.rhof(depth+.1)
        numer1, denom1 = -1 + h*h*5/12*om*om/cu/cu, h*rhou
        numer2, denom2 = -1 + h*h*5/12*om*om/cb/cb, h*rhob
        term1, term2 = numer1 /denom1, numer2/denom2
        da = term1 + term2
        return da

    def da_numerov(self, j):
        inter_depth = self.D
        depth, h = self.z[j], self.h
        om = self.omega
        if depth in inter_depth:
            da = self.da_inter_numerov(j)
            print("inter depth",depth)
            print("da", da)
        else:
            c, rho = self.medium.sspf(depth), self.medium.rhof(depth)
            numer = -2 + h*h * 10 /12 * om*om / (c*c)
            denom = h * self.medium.rhof(depth)
            da = numer / denom
        return da

    def eb_numerov(self, j):
        depth = self.z[j] 
        rho = self.medium.rhof(depth)
        return self.h/12 / rho 

    def db_numerov(self, j):
        depth = self.z[j]
        h = self.h
        if depth in self.D:
            db = 5/12 * (h / self.rhow + h / self.rhob)
        else:
            db = 10/12 * h / self.medium.rhof(depth)
        return db

    def Amat(self):
        z = self.z
        N = len(z)
        A = np.matrix(np.zeros((N-1,N-1))) # throw out surface point
        for i in range(1, N):
            ind = i -1
            A[ind,ind] = self.da_numerov(i)
            if i < (N-1): # not on the last point
                if (z[i] in self.D): # if I'm at the interface make sure I'm setting the derivative right
                    A[ind+1, ind] = self.ea_numerov(i+1)
                    A[ind, ind+1] = self.ea_numerov(i+1) 
                else:
                    A[ind+1, ind] = self.ea_numerov(i)
                    A[ind, ind+1] = self.ea_numerov(i) 
        return A

    def Bmat(self):
        z = self.z
        N = len(z)
        B = np.matrix(np.zeros((N-1,N-1))) # throw out surface point
        for i in range(1, N):
            ind = i -1 # I threw out the first row and column, so...matrix and depth indices differ by one
            B[ind,ind] = self.db_numerov(i)
            if i < (N-1): # not on the last point
                if (z[i] in self.D): # if I'm at the interface make sure I'm setting the derivative right
                    B[ind+1, ind] = self.eb_numerov(i+1)
                    B[ind, ind+1] = self.eb_numerov(i+1) 
                else:
                    B[ind+1, ind] = self.eb_numerov(i)
                    B[ind, ind+1] = self.eb_numerov(i) 
        return B

    def get_eigs(self):
        A, B = self.Amat(), self.Bmat()
        lam, w = lin.eigh(A,B)
        return lam, w

    def get_modes(self):
        print('discretizing depth')
        self.discretize_depth()
        lam, w = self.get_eigs()
        kr_min, kr_max = self.kr_bounds() 
        mode_inds = list(filter(lambda i: ((lam[i] > 0) and (np.sqrt(lam[i]) >= kr_min) and (np.sqrt(lam[i]) <= kr_max)), range(len(lam)) ))
        kr = [np.sqrt(lam[i]) for i in mode_inds]
        mode_vecs = [w[:,i] for i in mode_inds]
        mode_vecs = [np.insert(x, 0, 0) for x in mode_vecs] # prepend 0
        # normalize
        for i in range(len(mode_vecs)):
            mode_vec = mode_vecs[i]
            squares = np.square(mode_vec)
            terms = [squares[i] / self.medium.rhof(self.z[i]) for i in range(len(squares))]
            I = sum((np.array(terms[1:]) +np.array(terms[:-1]))/2) * self.z[-1] / len(squares) # trapezoid sum
            I -= (terms[0] / 2 + terms[-1] /2)
            krm = kr[i]
            gamma = np.sqrt(krm*krm - pow((self.omega / self.cb),2))
            Nm = I
            mode_vec = mode_vec /  Nm
            mode_vecs[i] = mode_vec
        self.eigenf = mode_vecs
        self.kr_prop = kr
        modefs = []
        for vec in mode_vecs:
            modefs.append(interp1d(self.z, vec)) # turn it into a function of z
        return kr, mode_vecs, modefs

    def getNM(self):
        krm, modevec, modef = self.get_modes()
        med = self.medium
        nm = NMModel(krm, modevec, modef, med, self.omega)
        return nm
        

