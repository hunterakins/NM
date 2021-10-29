import timing
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize, Bounds
from NM import normal_mode as nm
from NM import general as g
from NM import ssp_helpers as sh
from cz_detector.mode_ray import bf_thing
import dill as dill
from env.env.envs import factory
from heigenray.heigenray import get_eigenrays, plot_eigenrays, get_rcv_angs



from matplotlib import rc

rc('text', usetex=True)
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'



def reverse_model_kr(model):
    model.kr_prop.reverse()
    model.mode_vec.reverse()
    model.modef.reverse()
    
def restrict_modes(model, zs):
    modef = model.modef
    num_modes= len(modef)
    strengths = np.zeros(num_modes)
    for i in range(num_modes):
        strengths[i] = modef[i](zs)
    strengths /= np.max(strengths)
    i=0
    while strengths[i] < 0.05:
        i+=1
    return i, np.array(model.kr_prop[i:])

def plot_mode_strength(model, zs, i):
    """
    Plot the excitation strength vs mode number for a source depth zs
    """
    modef = model.modef
    num_modes= len(modef)
    strengths = np.zeros(num_modes)
    for i in range(num_modes):
        strengths[i] = modef[i](zs)
    plt.figure()
    plt.plot(range(i, i+num_modes), abs(strengths))
    plt.xlabel('Mode number')
    plt.savefig('/home/hunter/research/cz_detector/notes/pics/python_strength.png')
    return

def plot_kr(kr, i):
    plt.figure()
    plt.plot(range(i, i+len(kr)), kr)
    plt.ylabel('$k_{n}$')
    plt.xlabel('Mode number')
    plt.savefig('/home/hunter/research/cz_detector/notes/pics/python_kn.png')

def plot_D(kr, i):
    dkdn = kr[1:] - kr[:-1]
    D = -2*np.pi/dkdn
    plt.figure()
    plt.plot(range(i, i+D.size), D)
    plt.ylabel('D')
    plt.xlabel('Mode number')
    plt.savefig('/home/hunter/research/cz_detector/notes/pics/python_D.png')

    plt.figure()
    plt.plot(range(i, i+D.size), dkdn)
    plt.ylabel('$\\frac{\partial k}{\partial n}$')
    plt.xlabel('Mode number')
    plt.savefig('/home/hunter/research/cz_detector/notes/pics/python_dkdn.png')
    return D

def plot_X(kr, i):
    d2kdn2 = kr[2:] + kr[:-2] - 2*kr[1:-1]
    X = 2*np.pi/d2kdn2
    plt.figure()
    plt.plot(range(i, i+X.size), X)
    plt.ylabel('X')
    plt.xlabel('Mode number')
    plt.savefig('/home/hunter/research/cz_detector/notes/pics/python_X.png')

    plt.figure()
    plt.plot(range(i, i+X.size), d2kdn2)
    plt.ylabel('$\\frac{\partial^{2} k}{\partial n^{2}}$')
    plt.xlabel('Mode number')
    plt.savefig('/home/hunter/research/cz_detector/notes/pics/python_d2kdn2.png')
    return X

def phase_approx(mod_out, zs):
    """
    Look at the Taylor expansion for mode phase
    as well as the number of modes in a group at the 
    cycle distance D """
    i, kr=restrict_modes(mod_out, zs)
    dkdn = kr[1:] - kr[:-1]
    D = -2*np.pi/dkdn
    n0_offset = 800
    D0 = D[n0_offset]
    n0 = i+n0_offset
    d2kdn2 = kr[2:] + kr[:-2] - 2*kr[1:-1]
    X = 2*np.pi/d2kdn2
    X0 = X[n0_offset] # evaluate d2k
    delta_n_max = int(np.sqrt(X[n0_offset] / 2/D[n0_offset]))
    print('delta_n_max', delta_n_max)
    sub_kr = kr[n0_offset:n0_offset+delta_n_max]
    phases = np.array(sub_kr)*D0
    approx_phases = np.zeros(delta_n_max)
    phi0 = sub_kr[0]*D0
    phases -= phi0
    for i in range(delta_n_max):
        phi_approx = 2*np.pi*i + 2*np.pi*(0.5*i*i/X0)*D0
        approx_phases[i] = phi_approx
    plt.figure()
    print(n0)
    plt.xlabel('$\Delta n \ (n_{0} = ' + str(n0) + ')$')
    plt.ylabel('Phase (radians)')
    plt.plot(phases%(2*np.pi))
    plt.plot(approx_phases%(2*np.pi))
    plt.savefig('/home/hunter/research/cz_detector/notes/pics/phase_approx_' + str(n0_offset) +'.png')
    plt.show()

def mode_ray_degrade_range(mod_out, zs):
    n0, kr = restrict_modes(mod_out, zs)

    dkdn = kr[1:] - kr[:-1]
    D = -2*np.pi/dkdn
    d2kdn2 = kr[2:] + kr[:-2] - 2*kr[1:-1]
    X = 2*np.pi/d2kdn2
    X0 = X[0] # evaluate d2k
    delta_n_cz = int(np.sqrt(X[0] / 2/D[0]))
    delta_kr = kr[delta_n_cz] - kr[0]
    B =  delta_kr/4/np.pi
    coh_range = 1/B
    print('coh range', coh_range)
    print('delta_n_cz', delta_n_cz)
    i0 = int(delta_n_cz // 2)
    kr_group = kr[:delta_n_cz]
    r0 = D[i0]
    rvals = np.arange(r0, r0 + 1200, 100)
    plt.figure()
    for r in rvals:
        phi0 = kr_group[0]*r
        phases = kr_group * r - phi0
        phases = phases % (2*np.pi)
        big_inds = (phases >= np.pi)
        phases[big_inds] = phases[big_inds] - 2*np.pi
        plt.scatter([r]*phases.size, phases, color='b')
    plt.ylabel('Phase (radians)')
    plt.xlabel('Range (m)')
    plt.savefig('/home/hunter/research/cz_detector/notes/pics/phase_range_dep.png')
    plt.show()

def get_mode_start_center_ind(delta_n_cz, num_modes):
    """
    Get the index of the start of each group
    """
    start_inds = [0]
    center_inds = [int(delta_n_cz[0]/2)]
    while start_inds[-1] + delta_n_cz[start_inds[-1]] < num_modes:
        delta_n_tmp = delta_n_cz[start_inds[-1]]
        start_inds.append(start_inds[-1] + int(delta_n_tmp))
        center_inds.append(start_inds[-1] + int(delta_n_tmp/2))
    return start_inds, center_inds

def mode_group_dominanace(mod_out, zs):
    """
    Divide modes into groups based on n_cz
    For each group, get D of middle mode
    Use rule of thumb that each mode group contributes
    to 400 meters on each side
    Make a plot of each group, where the plot scatters
    group center mode number vs D and then adds a box
    up and down
    """
    
    n0, kr = restrict_modes(mod_out, zs)

    dkdn = kr[1:] - kr[:-1]
    D = -2*np.pi/dkdn
    d2kdn2 = kr[2:] + kr[:-2] - 2*kr[1:-1]
    X = 2*np.pi/d2kdn2
    X0 = X[0] # evaluate d2k
    delta_n_cz = np.sqrt(X/2/D[1:])
    num_modes = kr.size
    start_inds, center_inds = get_mode_start_center_ind(delta_n_cz, num_modes)
    print(start_inds, center_inds)
    delta_n_cz_size = delta_n_cz[start_inds]
    D_vals = D[center_inds]
    num_groups = len(start_inds)
    dom = range(num_groups)
    plt.figure()
    plt.errorbar(dom, D_vals, yerr=[400]*num_groups, fmt='none')
    plt.xlabel('Mode group number')
    plt.ylabel('Range of influence (m)')
    plt.savefig('/home/hunter/research/cz_detector/notes/pics/mode_influence.png')
    plt.show()
    
def mode_bundles(model, zs):
    """
    Modes should be reversed in advance
    """

    zr = np.linspace(120, 180, 100)
    
    rho1, rho2 = 1000, 1500
    D = 5000
    ## SSP STUFF
    cb = 1800
    ztild = lambda z: 2*(z-1300)/1300
    c0 =1500
    eps =0.00737
    water_sspf = lambda z: c0*(1+eps*(ztild(z) - 1 + np.exp(-ztild(z))))
    sspf = lambda z: np.piecewise(z, [z<= D, z>D], [water_sspf, cb])
    
    alpha1 = .5 # in db/lambda

    k = 2*np.pi*f/ cb

    alpha1 = k*alpha1/54.86 # convert dB/lambda to wavenumber atten
    
    rhof = lambda z: np.piecewise(z, [z <= D, z > D], [rho1, rho2])
    alphaf = lambda z: np.piecewise(z, [z <= D, z > D], [0, alpha1])
   
    #z = np.arange(20.0, 5000, 20)
    #r = np.arange(1000.0, rs, 100)
      
    med=g.Medium([D], rhof, cb, sspf)

    n0, kr = restrict_modes(mod_out, zs)

    dkdn = kr[1:] - kr[:-1]
    D_center = -2*np.pi/dkdn
    d2kdn2 = kr[2:] + kr[:-2] - 2*kr[1:-1]
    X = 2*np.pi/d2kdn2
    delta_n_cz = np.sqrt(X/2/D_center[1:])
    num_modes = kr.size
    start_inds, center_inds = get_mode_start_center_ind(delta_n_cz, num_modes)
    print(start_inds, center_inds)
    delta_n_cz_size = delta_n_cz[start_inds]

    D_vals = D_center[center_inds]
    num_groups = len(start_inds)
    dom = range(num_groups)

    fig1, ax = plt.subplots(1,1)
    #fig2, axes = plt.subplots(num_groups-4,1, sharey=True, sharex=True)

    fig3 , ax3=plt.subplots(1,1)

    cmap = matplotlib.cm.get_cmap('Spectral')
    for count in range(num_groups-4):
        #n = 80 # deduced from sqrt(X/2D)
        D_center = D_vals[count]
        inds = slice(n0+start_inds[count], n0+start_inds[count+1])

        mini_model = nm.NMModel(model.kr_prop[inds], model.mode_vec[inds], model.modef[inds], med, model.omega)
        kr_group = np.array(model.kr_prop[inds])

        modes = model.modef[inds]

        r = np.arange(D_center - 1000, D_center + 1000, 100)
        vals = mini_model.greens(zs, [zs],r)
        ax.plot(r, abs(vals[0,:]), color=cmap(count/num_groups))

        vals = mini_model.greens(zs, zr,r)
        lamb =1500 / f 
        k = 2*np.pi*f/sspf(zs)
        #print(kr_group/k)
        kr_angles = 180/np.pi*np.arccos(kr_group/ k)
        ax3.scatter([count]*kr_angles.size, kr_angles, color=cmap(count/(num_groups-4)))

        beams, angles = bf_thing(vals, zr, k)
        angle_inds = slice(4, -4)
        print(beams.shape, r.shape, zr.shape)
        tmp_fig, ax2 = plt.subplots(1,1)
        for range_count in range(r.size):
            ax2.plot(angles[angle_inds]*180/np.pi, np.square(abs(beams[angle_inds,range_count])), color=cmap(range_count/r.size))
        
        ax2.set_xlabel('Beam angle (deg)')
        ax2.set_ylabel('Power')
        ax2.legend([str(int(x)) + ' m' for x in r], loc='upper right')
        tmp_fig.savefig('/home/hunter/research/cz_detector/notes/pics/mode_ray_beams_' + str(count)+'.png')

        r = np.arange(100, 65000, 100)
        z = np.arange(10, 5000, 10)
        vals = mini_model.greens(zs, z,r)
        fig = plt.figure()
        plt.pcolormesh(r, z, 10*np.log10(abs(vals)/np.max(abs(vals))),vmin=-30, vmax=0, shading='auto')
        plt.colorbar()
        plt.xlabel('Range (m)')
        plt.ylabel('Depth (m)')
        plt.gca().invert_yaxis()
        fig.set_size_inches(6,4)
        print('D_center, count', count, D_center)
        fig.savefig('/home/hunter/research/presentations/qual_exam/pics/mode_bundle_'+ str(count)+'.png')
        

    #fig2.text(0.5, 0.01,  'Beam angle (deg)', ha='center')
    #fig2.text(0.01, 0.5,  'Power', va='center', rotation='vertical')
    ax.set_xlabel('Range (m)') 
    ax.set_ylabel('$|p(r, 150)|$')
    ax.legend(['Group ' + str(i+1) for i in range(num_groups-1)])
    fig1.savefig('/home/hunter/research/cz_detector/notes/pics/mode_ray_range.png')
    ax3.set_xlabel('Mode group number')
    ax3.set_ylabel('Ray angle (deg)')
    fig3.savefig('/home/hunter/research/cz_detector/notes/pics/mode_ray_angles.png')
    

    
    plt.show()

def coherence_within_cz(model, zs):
    """
    Generate some synthetic data using all the modes
    Then use mode-ray theory to look at an FFT centered on one of the 
    mode ray convergence zone locations and see how the coherent gain
    drops off as the FFT gets longer
    """
    return

def cz_pressure(model, zs):
    zr = [zs]
    
    rho1, rho2 = 1000, 1500
    D = 5000
    ## SSP STUFF
    cb = 1800
    ztild = lambda z: 2*(z-1300)/1300
    c0 =1500
    eps =0.00737
    water_sspf = lambda z: c0*(1+eps*(ztild(z) - 1 + np.exp(-ztild(z))))
    sspf = lambda z: np.piecewise(z, [z<= D, z>D], [water_sspf, cb])
    
    alpha1 = .5 # in db/lambda

    k = 2*np.pi*f/ cb

    alpha1 = k*alpha1/54.86 # convert dB/lambda to wavenumber atten
    
    rhof = lambda z: np.piecewise(z, [z <= D, z > D], [rho1, rho2])
    alphaf = lambda z: np.piecewise(z, [z <= D, z > D], [0, alpha1])
   
    med=g.Medium([D], rhof, cb, sspf)
    n0, kr = restrict_modes(model, zs)
    inds = slice(0, -1)
    model = nm.NMModel(model.kr_prop[inds], model.mode_vec[inds], model.modef[inds], med, model.omega)
    r = np.arange(59000, 66000, 10)
    vals = model.greens(zs, zr,r)
    plt.figure()
    plt.plot(r/1000, abs(vals[0,...]))
    plt.xlabel('Range (km)')
    plt.ylabel('$|p(r, z_{s})|^{2}$')
    plt.savefig('/home/hunter/research/cz_detector/notes/pics/I_cz.png')
    plt.show()
    
def single_mode_frequency(model, zs):
    """
    How broadband is a mode-ray?
    Modes should be reversed in advance
    """

    
    rho1, rho2 = 1000, 1500
    D = 5000
    ## SSP STUFF
    cb = 1800
    ztild = lambda z: 2*(z-1300)/1300
    c0 =1500
    eps =0.00737
    water_sspf = lambda z: c0*(1+eps*(ztild(z) - 1 + np.exp(-ztild(z))))
    sspf = lambda z: np.piecewise(z, [z<= D, z>D], [water_sspf, cb])
    
    alpha1 = .5 # in db/lambda

    k = 2*np.pi*f/ cb

    alpha1 = k*alpha1/54.86 # convert dB/lambda to wavenumber atten
    
    rhof = lambda z: np.piecewise(z, [z <= D, z > D], [rho1, rho2])
    alphaf = lambda z: np.piecewise(z, [z <= D, z > D], [0, alpha1])
   
    #z = np.arange(20.0, 5000, 20)
    #r = np.arange(1000.0, rs, 100)
      
    med=g.Medium([D], rhof, cb, sspf)

    n0, kr = restrict_modes(mod_out, zs)

    dkdn = kr[1:] - kr[:-1]
    D_center = -2*np.pi/dkdn
    d2kdn2 = kr[2:] + kr[:-2] - 2*kr[1:-1]
    X = 2*np.pi/d2kdn2
    delta_n_cz = np.sqrt(X/2/D_center[1:])
    num_modes = kr.size
    start_inds, center_inds = get_mode_start_center_ind(delta_n_cz, num_modes)
    print(start_inds, center_inds)
    delta_n_cz_size = delta_n_cz[start_inds]

    D_vals = D_center[center_inds]
    num_groups = len(start_inds)
    dom = range(num_groups)

    cmap = matplotlib.cm.get_cmap('Spectral')
    plt.figure()
    min_D = D_vals[0]
    max_D = D_vals[-1]
    for count in range(num_groups-1):
        #n = 80 # deduced from sqrt(X/2D)
        D_center = D_vals[count]
        inds = slice(n0+start_inds[count], n0+start_inds[count+1])

        mini_model = nm.NMModel(model.kr_prop[inds], model.mode_vec[inds], model.modef[inds], med, model.omega)
        kr_group = np.array(model.kr_prop[inds])

        modes = model.modef[inds]
        dr = .1
        r = np.arange(D_center - 2000, D_center + 2000, dr)
        vals = mini_model.greens(zs, [zs],r)
        if count == 0:
            norm = np.sqrt(np.var(vals))
        vals /= norm
        vals = np.squeeze(vals)
        f_vals = 1/vals.size* np.fft.fft(vals)
        freqs = np.fft.fftfreq(vals.size, dr)
        plt.plot(freqs, abs(f_vals), color=cmap(count/num_groups))
        
    plt.xlim([1.93, 1.96])
    plt.xlabel('Freq (m$^{-1}$)')
    plt.ylabel('$|p|$')
    print('count', count)
    #plt.savefig('/home/hunter/research/cz_detector/notes/pics/f_rep_' + str(count) + '.png')

    inds = slice(n0,n0+start_inds[count+1])
    mini_model = nm.NMModel(model.kr_prop[inds], model.mode_vec[inds], model.modef[inds], med, model.omega)
    kr_group = np.array(model.kr_prop[inds])

    modes = model.modef[inds]
    dr = .1
    r = np.arange(min_D - 1000, max_D + 1000, dr)
    vals = mini_model.greens(zs, [zs],r)
    vals /= np.sqrt(np.var(vals))
    vals = np.squeeze(vals)
    f_vals = 1/vals.size* np.fft.fft(vals)
    freqs = np.fft.fftfreq(vals.size, dr)
    plt.plot(freqs, abs(f_vals), color='k')
    plt.legend(['Mode ' + str(x) for x in range(num_groups-1)] + ['Full zone'])
    #plt.xlim([1.93, 1.96])
    #plt.xlabel('Freq (m$^{-1}$)')
    #plt.ylabel('$|p|$')
    plt.savefig('/home/hunter/research/cz_detector/notes/pics/full_zone.png')
    plt.show()

def power_leak(model, zs):
    """
    Compare time domain signal energy to peak power
    """

    
    rho1, rho2 = 1000, 1500
    D = 5000
    ## SSP STUFF
    cb = 1800
    ztild = lambda z: 2*(z-1300)/1300
    c0 =1500
    eps =0.00737
    water_sspf = lambda z: c0*(1+eps*(ztild(z) - 1 + np.exp(-ztild(z))))
    sspf = lambda z: np.piecewise(z, [z<= D, z>D], [water_sspf, cb])
    
    alpha1 = .5 # in db/lambda

    k = 2*np.pi*f/ cb

    alpha1 = k*alpha1/54.86 # convert dB/lambda to wavenumber atten
    
    rhof = lambda z: np.piecewise(z, [z <= D, z > D], [rho1, rho2])
    alphaf = lambda z: np.piecewise(z, [z <= D, z > D], [0, alpha1])
   
    #z = np.arange(20.0, 5000, 20)
    #r = np.arange(1000.0, rs, 100)
      
    med=g.Medium([D], rhof, cb, sspf)

    n0, kr = restrict_modes(mod_out, zs)

    dkdn = kr[1:] - kr[:-1]
    D_center = -2*np.pi/dkdn
    d2kdn2 = kr[2:] + kr[:-2] - 2*kr[1:-1]
    X = 2*np.pi/d2kdn2
    delta_n_cz = np.sqrt(X/2/D_center[1:])
    num_modes = kr.size
    start_inds, center_inds = get_mode_start_center_ind(delta_n_cz, num_modes)
    print(start_inds, center_inds)
    delta_n_cz_size = delta_n_cz[start_inds]

    D_vals = D_center[center_inds]
    num_groups = len(start_inds)
    dom = range(num_groups)

    cmap = matplotlib.cm.get_cmap('Spectral')
    plt.figure()
    min_D = D_vals[0]
    max_D = D_vals[-1]
    inds = slice(n0, -1)

    mini_model = nm.NMModel(model.kr_prop[inds], model.mode_vec[inds], model.modef[inds], med, model.omega)
    full_r = np.arange(min_D-2500, max_D +2500, 10)
    full_vals = mini_model.greens(zs, [zs],full_r)   
    full_vals=np.squeeze(full_vals)
    kr_group = np.array(model.kr_prop[inds])

    modes = model.modef[inds]
    dr=  1
    integration_ranges = np.arange(100, 2000, 100)
    start_locations = D_vals[:]
    peak_pows = np.zeros((start_locations.size, integration_ranges.size))
    E = np.zeros((start_locations.size, integration_ranges.size))
    N_zero_pad = 2**10
    for i, D_c in enumerate(start_locations):
        for j, integration_range in enumerate(integration_ranges):
            print(integration_range)
            r = np.arange(D_c - integration_range/2, D_c + integration_range/2, dr)
            vals = mini_model.greens(zs, [zs],r)   
            vals = np.squeeze(vals)
            E_normed = np.sum(np.square(abs(vals)))/vals.size
            E[i,j] = E_normed
            f_vals = 1/vals.size* np.fft.fft(vals, n=N_zero_pad)
            freqs = np.fft.fftfreq(N_zero_pad, dr)
            #plt.plot(freqs, abs(f_vals))
            #plt.show()
            max_pow = np.max(np.square(abs(f_vals)))
            peak_pows[i,j] = max_pow
        fig, axes = plt.subplots(2,1)
        axes[0].plot(integration_ranges, E[i,:],color='r')
        axes[0].plot(integration_ranges, peak_pows[i,:],color='k')
        axes[0].set_xlabel('Integration length (m)')
        axes[0].legend(['Energy', ' Peak pow'])
        axes[1].plot(full_r, abs(full_vals), color='k')
        axes[1].scatter(D_c, abs(full_vals[np.argmin(abs(full_r - D_c))]), color='b', marker='+', s=24)
        axes[1].plot(full_r[abs(full_r - D_c) < 1000], abs(full_vals[abs(full_r - D_c) < 1000]), color='b')
        axes[1].plot(full_r[abs(full_r - D_c) < 50], abs(full_vals[abs(full_r - D_c) < 50]), color='r')
        axes[1].legend(['$|p|$ over full CZ', 'Data included in longest integration', 'Data for shortest integration'])
        axes[1].set_xlabel('Range (m)')
        plt.tight_layout()
        plt.savefig('/home/hunter/research/cz_detector/notes/pics/energy_dc'  + str(i) + '.png')
        plt.close(fig)
            
    
    #plt.savefig('/home/hunter/research/cz_detector/notes/pics/f_rep_' + str(count) + '.png')

    inds = slice(n0,n0+start_inds[count+1])
    mini_model = nm.NMModel(model.kr_prop[inds], model.mode_vec[inds], model.modef[inds], med, model.omega)
    kr_group = np.array(model.kr_prop[inds])

    modes = model.modef[inds]
    dr = .1
    r = np.arange(min_D - 1000, max_D + 1000, dr)
    vals = mini_model.greens(zs, [zs],r)
    vals /= np.sqrt(np.var(vals))
    vals = np.squeeze(vals)
    f_vals = 1/vals.size* np.fft.fft(vals)
    freqs = np.fft.fftfreq(vals.size, dr)
    plt.plot(freqs, abs(f_vals), color='k')
    plt.legend(['Mode ' + str(x) for x in range(num_groups-1)] + ['Full zone'])
    #plt.xlim([1.93, 1.96])
    #plt.xlabel('Freq (m$^{-1}$)')
    #plt.ylabel('$|p|$')
    plt.tight_layout()
    plt.savefig('/home/hunter/research/cz_detector/notes/pics/full_zone_channel.png')
    plt.show()

def comp_to_eigenray(zs, zr):
    plot_type='eigenray'
    
    folder ='at_files/'
    fname = 'eigen_test'
    
    builder = factory.create('deepwater')
    env = builder()

    launch_angles = [-20, 20]
    num_interactions = 2
    min_dist = 5
    dz = 1
    zmax = 5000
    dr = 1
    for rmax in np.arange(59.5 ,66, .5):
        env.add_field_params(dz, zmax, dr, rmax*1e3)
        env.add_source_params(3000, np.array([zs]), np.array([zr]))
        rays=get_eigenrays(env, folder, fname, launch_angles, min_dist, num_interactions)
        rcv_angles = get_rcv_angs(rays)
        print('rcv angles', rcv_angles)
        fig,ax1= plot_eigenrays(env,rays)
        ax1.set_xlabel('Range (m)')
        ax1.set_ylabel('Depth (z)')
        
        plt.savefig('pics/' + str(round(rmax, 3)) + '.png')


def cz_eigenzone_comparison(zs, zr):
    """
    Look at eigenrays at two points in the CZ
    """
    plot_type='eigenray'
    
    folder ='at_files/'
    fname = 'eigen_test'
    
    builder = factory.create('deepwater')
    env = builder()

    launch_angles = [-20, 20]
    num_interactions = 1
    min_dist = 2
    dz = 1
    zmax = 5000
    dr = 1
    for rmax in [60.5, 62.5]:
        env.add_field_params(dz, zmax, dr, rmax*1e3)
        env.add_source_params(3000, np.array([zs]), np.array([zr]))
        rays=get_eigenrays(env, folder, fname, launch_angles, min_dist, num_interactions)
        rcv_angles = get_rcv_angs(rays)
        print('rcv angles', rcv_angles)
        if rmax == 60.5:
            fig,ax1= plot_eigenrays(env,rays,color='k')
        if rmax == 62.5:
            for ray in rays:
                num_top_bnc = ray.num_top_bnc
                num_bot_bnc = ray.num_bot_bnc
                x = ray.xy[0,:]
                y = ray.xy[1,:]
                if num_top_bnc == 0 and num_bot_bnc==0: 
                    ax1.plot(x,y, color='b')
    ax1.set_ylim([env.dz, env.zmax])
    ax1.invert_yaxis()
    ax1.set_xlim([env.dr, env.rmax])
    fig.set_size_inches(8,4)
    plt.savefig('/home/hunter/research/presentations/qual_exam/pics/cz_range_comp.png', dpi=500)

if __name__ == '__main__':

    #cz_eigenzone_comparison(150, 150)
    #comp_to_eigenray(150, 150)

    f = 3000
    zs = 150

    with open(str(f) + '_nm.dill', 'rb') as thing:
        mod_out = dill.load(thing)


    reverse_model_kr(mod_out)
    
    mode_bundles(mod_out, 150)
    
    sys.exit(0)
    power_leak(mod_out, zs)
    mode_bundles(mod_out, zs)
    single_mode_frequency(mod_out, zs)
    cz_pressure(mod_out, zs)
    

    mode_group_dominanace(mod_out, zs)
    sys.exit(0)

    phase_approx(mod_out, zs)
    mode_ray_degrade_range(mod_out, zs)


    n0, kr = restrict_modes(mod_out, zs)
    dkdn = kr[1:] - kr[:-1]
    D = -2*np.pi/dkdn
    d2kdn2 = kr[2:] + kr[:-2] - 2*kr[1:-1]
    X = 2*np.pi/d2kdn2
    n = np.sqrt(X[:1000]/2/D[1:1001])
    plt.figure()
    plt.plot(range(n0, n0+n.size), n)
    plt.ylabel('$\Delta n_{cz}$')
    plt.xlabel('Mode number')
    plt.savefig('/home/hunter/research/cz_detector/notes/pics/n_cz_channel.png')
    plt.show()
    
