import pickle
import numpy as np
import timing
import NM.general as g
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize, Bounds
import NM.ssp_helpers as sh
import NM.diagnostics as d

if __name__ == '__main__':
    fs = np.linspace(300, 500, 1)
    rsrs = np.linspace(1000, 1000, 1)
    zszs = np.linspace(20, 80, 1)
    amps = np.linspace(60, 75, 4)
    print(amps)
    diag = input("Generate diagnostics?")
    if diag == 'y':
        print('caught it')
        d.generate_diagnostics(fs, rsrs, zszs)
    else:
        res = d.load_results(fs, rsrs)
        res1 = res[0]
        print((res1.pert[400]))
    #    print(np.max(res1.pert))
    #    d.view_results(res1)
        for subres in res:
            plt.plot(subres.E[:])
        plt.show()
        g.compare_pert(res1.E, res1.coeffs, res1.est_coeffs)
        a = [x.coeffs[0] for x in res]
        b = [x.est_coeffs[0] for x in res]
        line = np.linspace(np.min(a), np.max(a), 20)
        plt.plot(line, line)
        plt.scatter(a,b)
        plt.title("Leading coefficient of perturbation vs. estimated coefficient")
        plt.xlabel("True coefficient")
        plt.ylabel("Estimated coefficient")
        plt.show()
        a = [.12*x.coeffs[0] for x in res]
        b = [.12*x.est_coeffs[0] for x in res]
        line = np.linspace(np.min(a), np.max(a), 20)
        plt.plot(line, line)
        plt.scatter(a,b)

        plt.title("Comparing estimated and true amplitude")
        plt.xlabel("True amplitude (m/s)")
        plt.ylabel("Estimated amplitude (m/s)")
        plt.show()

