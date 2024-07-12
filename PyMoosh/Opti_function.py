# pm est la version non locale de pymoosh, pym, c'est la version normale de pymoosh pour differential_evolution (comme on a pas tout converti, c'est encore séparé sur nos pc)
import Pymoosh_non_local_datas as pm 
import PyMoosh as pym
import numpy as np
import matplotlib.pyplot as plt 

def Optimized(budget, Xmin,Xmax, wl_ref, R_ref, mat1, mat2, stack, thickness, theta,  polarization = 1):
    #plot the reference curve
    plt.plot(wl_ref, R_ref, label = "Ref curve")
    plt.xlabel('wavelength')
    plt.ylabel('Reflectance')
  
    #definition of the cost function
    def cost(X):
        base, scale = X[5], X[6]

      #we changed values of X for ones we can use (turn [chi_b, w_p, gamma, Re(beta²), Im(beta²)/w] into [chi_b, chi_f, w_p, beta])
        def convert(wavelength):
            w = 2 * np.pi * 299792458 / (wavelength * 10**(-9))
            chi_f = -X[1]**2 / ( w* (w + 1j * X[2]))
            return X[0], chi_f, X[1], np.sqrt(X[3]-X[4]*w*1j)
          
        #defind the structure with non-local material
        materials = [mat1, mat2, pm.Material(convert, "NL")]  
        structure = pm.Structure(materials, stack, thickness)
        R = []

        #for each wavelength in the reference list, we calculate the reflectance 
        for i in wl_r :
            R.append((pm.coefficient(structure, i, theta, 1)[2] + base) * scale)
        R = np.array(R, dtype = complex)
        
        #now calculate the difference between our list and the reference one (also the differential)
        V = np.abs(R_r - R)
        dV = np.abs(np.diff(R_r) - np.diff(R))
        cost = (sum(V) / len(V) + 20*sum(dV) / len(dV))
        return cost
    
    #find the optimized set up for the given budget 
    Best, Convergency = pym.differential_evolution(cost, budget, Xmin, Xmax)    
    print("\n Cost for best is:", cost(Best))
    
    #plot the optimized structure with the reference one to compare by using spectrum (we use convert again but with the optimized list of values)
    def convert(wavelength):
            w = 2 * np.pi * 299792458 / (wavelength * 10**(-9))
            chi_f = -Best[1]**2 / ( w* (w + 1j * Best[2]))
            return Best[0], chi_f, Best[1], np.sqrt(Best[3]-Best[4]*w*1j)
    mat = [mat1, mat2, pm.Material(convert, "NL")]
    struct = pm.Structure(mat, stack, thickness)
    
    spec = pm.spectrum(struct,theta, polarization, wl_ref[0], wl_ref[len(wl_ref)-1], 10000)
    scale = Best[6]
    plt.plot(spec[0], (np.array(spec[3]) + np.array([Best[5]]*len(spec[3])))*scale, label = "Fit curve")
    
    print("best =", Best)
    plt.show()
    return Best
