# encoding utf-8
import numpy as np
from scipy.special import wofz
import json
from refractiveindex import RefractiveIndexMaterial

class Material :

    """
        Types of material (default) : / format:                    / SpecialType:

            - simple_perm             / scalar(complex)            / 'Default'
            - magnetic                / list-tuple-array(complex)  / 'Default'
            - CustomFunction          / function                   / 'Default'
            - BrendelBormann          / string                     / 'Default'

        Types of material (special) : / format:                    / SpecialType:

            - ExpData                 / string                     / 'ExpData'
            - RefractiveIndexInfo     / list(shelf, book, page)    / 'RII'
            - Anisotropic             / list(shelf, book, page)    / 'ANI'
            - NonLocalMaterial        / list-tuple-array(complex)  / 'NL'
            - NonLocalCustomFunction  / string                     / 'NL'
    """
    
    def __init__(self, mat, SpecialType = "Default", verbose = False) :
        
        self.beta = 0
        self.mat = mat
        self.SpecialType = SpecialType
        if SpecialType == "Default" :
            if mat.__class__.__name__ == 'function' :
                self.type = "CustomFunction"
                self.permittivity_function = mat
                self.name = "CustomFunction : " + str(mat)
                if verbose :
                    print("Custom dispersive material. Epsilon =", mat.__name__, "(wavelength in nm)")
            elif not hasattr(mat, '__iter__') :
            # no func / not iterable --> single value, convert to complex by default
                self.type = "simple_perm"
                self.name = "SimplePermittivity :" + str(mat)
                self.permittivity = complex(mat)
                self.permeability = 1.
                if verbose :
                    print("Simple, non dispersive: epsilon = ", self.permittivity)

            elif isinstance(mat, list) or isinstance(mat, tuple) or isinstance(mat, np.ndarray) :
            # iterable : if list or similar : magnetic
                self.type = "magnetic"
                self.permittivity = mat[0]
                self.permeability = mat[1]
                self.name = "Permittivity :" + str(mat[0]) + "Permeability :" + str(mat[1])

                if verbose :
                    print("Magnetic, non dispersive : epsilon = ", mat[0], " mu = ", mat[1])

                if len(mat) > 2 :
                    print(f'Warning: Magnetic material should have 2 values (epsilon, mu), but {len(mat)} were given.')

            elif isinstance(mat,str) :
            # iterable: string --> database material
            # from file in shipped database
                import pkgutil
                f = pkgutil.get_data(__name__, "data/material_data.json")
                f_str = f.decode(encoding = 'utf8')
                database = json.loads(f_str)

                if mat in database :
                    material_data = database[mat]
                    model = material_data["model"]

                    if model == "BrendelBormann" :
                        self.type = "BrendelBormann"
                        self.name = "BrendelBormann model : " + str(mat)
                        self.f0 = material_data["f0"]
                        self.Gamma0 = material_data["Gamma0"]
                        self.omega_p = material_data["omega_p"]
                        self.f = np.array(material_data["f"])
                        self.gamma = np.array(material_data["Gamma"])
                        self.omega = np.array(material_data["omega"])
                        self.sigma = np.array(material_data["sigma"])

                    elif model == "CustomFunction" :
                        self.type = "CustomDatabaseFunction"
                        self.name = "CustomDatabaseFunction : " + str(mat)
                        permittivity = material_data["function"]
                        self.permittivity_function = authorized[permittivity]

                    else:
                        print(model, " not an existing model (yet).")
                        #sys.exit()

                    if verbose :
                        print("Database material :", self.name)
                else :
                    print(mat, "Unknown material (for the moment)")
                    print("Known materials:\n", existing_materials())
                    sys.exit()

            else :
                print(f"Warning : Given data is not in the right format for a 'Default' SpecialType. You should check the data format or specify a SpecialType. You can refer to the following table :")
                print(self.__doc__)

        elif SpecialType == "RII" :
            if len(mat) != 3 :
                print(f'Warning: Material from RefractiveIndex Database should have 3 values (shelf, book, page), but {len(mat)} were given.')
            self.type = "RefractiveIndexInfo"
            self.SpecialType = SpecialType
            self.name = "MaterialRefractiveIndexDatabase: " + str(mat)
            shelf, book, page = mat[0], mat[1], mat[2]
            self.path = "shelf : {}, book : {}, page : {}".format(shelf, book, page)
            material = RefractiveIndexMaterial(shelf, book, page) # create object
            self.material = material
            if verbose :
                print("Hello there ;) \n Material from Refractiveindex Database")

        elif SpecialType == "ExpData" :
            import pkgutil
            f = pkgutil.get_data(__name__, "data/material_data.json")
            f_str = f.decode(encoding='utf8')
            database = json.loads(f_str)

            if mat in database :
                material_data = database[mat]
                model = material_data["model"]

                if model == "ExpData" :
                    self.type = "ExpData"
                    self.name = "ExpData : " + str(mat)
                    self.SpecialType = SpecialType
                    wl = np.array(material_data["wavelength_list"], dtype = float)
                    epsilon = np.array(material_data["permittivities"], dtype = complex)
                    if "permittivities_imag" in material_data:
                        epsilon = epsilon + 1j * np.array(material_data["permittivities_imag"])

                    self.wavelength_list = wl
                    self.permittivities  = epsilon
                else :
                    print(f'Warning: Used model should be "ExpData", but {model} were given.')

        elif SpecialType == "ANI" :
            if len(mat) != 3 :
                print(f'Warning: Material from RefractiveIndex Database should have 3 values (shelf, book, page), but {len(mat)} were given.')
            self.type = "Anisotropic"
            self.SpecialType = SpecialType
            shelf, book, page = mat[0], mat[1], mat[2]
            self.path = "shelf: {}, book: {}, page: {}".format(shelf, book, page) # not necessary ?
            material_list = wrapper_anisotropy(shelf, book, page) # A list of three materials
            self.material_list = material_list
            self.material_x = material_list[0]
            self.material_y = material_list[1]
            self.material_z = material_list[2]
            self.name = "Anisotropic material from Refractiveindex.info : " + str(mat)

            if verbose :
                print("Material from Refractiveindex Database")
        # Here, non-local type have constant permeability
        elif SpecialType == "NL" :
            self.SpecialType = SpecialType
            if  mat.__class__.__name__ == "function" :
                self.type = "NonLocalCustomFunction"
                self.name = "NonLocalCustomFunction : " + str(mat)
                self.beta = mat(500)[3]
                if len(self.mat(500)) == 5 : 
                    self.permeability = self.mat(500)[4]
                else :
                    self.permeability = 1

                if verbose :
                    print("Custom dispersive material. Chi_b, chi_f, w_p, beta, Mu = ", str(mat) + str(mat.permeability), "(wavelength in nm)")
            
            elif isinstance(mat, list) or isinstance(mat, tuple) or isinstance(mat, np.ndarray) :
                if len(mat) == 4 or len(mat) == 5 :
                    self.type = "NonLocalMaterial"
                    self.chi_b, self.chi_f, self.w_p, self.beta = self.mat[0], self.mat[1], self.mat[2], self.mat[3]

                    if len(mat) == 4 :
                        self.permeability = 1.0

                    else :
                        self.permeability = mat[4]
                    self.name = "NonLocalMaterial :" + str(mat)

                    if verbose :
                        print(f"NonLocalMaterial : [chi_b = {self.chi_b}, chi_f = {self.chi_f}, w_p = {self.w_p}, beta = {self.beta}, mu = {self.permeability}] SpecialType = {self.SpecialType}")
            
            else :
                print(mat, "Unknown material (for the moment)")

        elif SpecialType == "Unspecified" :
            self.SpecialType = SpecialType
            print(SpecialType, "Unknown type of material (for the moment)")
            sys.exit()

        else :
            print(f'Warning : Unknown SpecialType : {SpecialType}')

    def __str__(self) :
        return self.name
    def __repr__(self):
        return self.mat

    def get_values_nl(self, wavelength = 500) :

        if self.type == "NonLocalCustomFunction" :
            self.beta = self.mat(wavelength)[3]
            self.w_p = self.mat(wavelength)[2]
            self.chi_b = self.mat(wavelength)[0]
            self.chi_f = self.mat(wavelength)[1]
        
        elif self.type == "NonLocalMaterial" :
            self.beta = self.mat[3]
            self.w_p = self.mat[2]
            self.chi_b = self.mat[0]
            self.chi_f = self.mat[1]

        return self.chi_b, self.chi_f, self.w_p, self.beta

    def get_permittivity(self, wavelength) :
        if self.type == "simple_perm" :
            return self.permittivity
        
        elif self.type == "magnetic" :
            return self.permittivity
        
        elif self.type == "CustomFunction" :
            return self.permittivity_function(wavelength)
        
        elif self.type == "BrendelBormann" :
            w = 6.62606957 * 10**(-25) * 299792458 / 1.602176565 * 10 **(-19) / wavelength
            a = np.sqrt(w * (w + 1j * self.gamma))
            x = (a - self.omega) / (np.sqrt(2) * self.sigma)
            y = (a + self.omega) / (np.sqrt(2) * self.sigma)
            # Polarizability due to bound electrons
            chi_b = np.sum(1j * np.sqrt(np.pi) * self.f * self.omega_p ** 2 /
                        (2 * np.sqrt(2) * a * self.sigma) * (wofz(x) + wofz(y)))
            # Equivalent polarizability linked to free electrons (Drude model)
            chi_f = -self.omega_p ** 2 * self.f0 / (w * (w + 1j * self.Gamma0))
            epsilon = 1 + chi_f + chi_b
            return epsilon
        
        elif self.type == "RefractiveIndexInfo" :
            try :
                k = self.material.get_extinction_coefficient(wavelength)
                return self.material.get_epsilon(wavelength)
            
            except :
                n = self.material.get_refractive_index(wavelength)
                return n**2
            
        elif self.type == "ExpData" :
            return np.interp(wavelength, self.wavelength_list, self.permittivities)
        
        elif self.type == "Anisotropic" :
            print("Warning: Functions for anisotropic materials generaly requires more information than isotropic ones. You probably want to use 'get_permittivity_ani()' function.")
        
        elif self.type == "NonLocalMaterial" :
            return 1 + self.chi_b + self.chi_f
        
        elif self.type == "NonLocalCustomFunction" :
            return 1 + self.mat(wavelength)[0] + self.mat(wavelength)[1]

    
    def get_permeability(self,wavelength, verbose = False) :
        if self.type == "simple_perm" or self.type == "magnetic" or self.type == "NonLocalMaterial" or self.type == "NonLocalCustomFunction" :
            return self.permeability
        
        elif self.type == "RefractiveIndexInfo" :
            if verbose :
                print('Warning : Magnetic parameters from RefractiveIndex Database are not implemented. Default permeability is set to 1.0.')
            return np.ones(wavelength.size)
        
        elif self.type == "Anisotropic" :
            if verbose :
                print('Warning : Magnetic parameters from RefractiveIndex Database are not implemented. Default permeability is set to 1.0.')
            return [1.0, 1.0, 1.0] # We should extend it to an array


        
def existing_materials():
    import pkgutil
    f = pkgutil.get_data(__name__, "data/material_data.json")
    f_str = f.decode(encoding='utf8')
    database = json.loads(f_str)
    for entree in database:
        if "info" in database[entree]:
            print(entree,"::",database[entree]["info"])
        else :
            print(entree)

# Sometimes materials can be defined not by a well known model
# like Cauchy or Sellmeier or Lorentz, but have specific formula.
# That may be convenient.

def permittivity_glass(wl):
    #epsilon=2.978645+0.008777808/(wl**2*1e-6-0.010609)+84.06224/(wl**2*1e-6-96)
    epsilon = (1.5130 - 3.169e-9*wl**2 + 3.962e3/wl**2)**2
    return epsilon

# Declare authorized functions in the database. Add the functions listed above.

authorized = {"permittivity_glass":permittivity_glass}

def wrapper_anisotropy(shelf, book, page):
    if page.endswith("-o") or page.endswith("-e"):
        if page.endswith("-e"):
            page_e, page_o = page, page.replace("-e", "-o")
        elif page.endswith("-o"):
            page_e, page_o = page.replace("-o", "-e"), page

        # create ordinary and extraordinary object.
        material_o = RefractiveIndexMaterial(shelf, book, page_o)
        material_e = RefractiveIndexMaterial(shelf, book, page_e)
        return [material_o, material_o, material_e]
    
    elif page.endswith("-alpha") or page.endswith("-beta") or page.endswith("-gamma"):
        if page.endswith("-alpha"):
            page_a, page_b, page_c = page, page.replace("-alpha", "-beta"), page.replace("-alpha", "-gamma")
        elif page.endswith("-beta"):
            page_a, page_b, page_c = page.replace("-beta", "-alpha"), page, page.replace("-beta", "-gamma")
        elif page.endswith("-gamma"):
            page_a, page_b, page_c = page.replace("-gamma", "-alpha"), page.replace("-gamma", "-beta"), page
        
        # create ordinary and extraordinary object.
        material_alpha = RefractiveIndexMaterial(shelf, book, page_a)
        material_beta = RefractiveIndexMaterial(shelf, book, page_b)
        material_gamma = RefractiveIndexMaterial(shelf, book, page_c)
        return [material_alpha, material_beta, material_gamma]
    
    else:
        # there may better way to do it.
        try:
            page_e, page_o = page + "-e", page + "-o"
            material_o = RefractiveIndexMaterial(shelf, book, page_o)
            material_e = RefractiveIndexMaterial(shelf, book, page_e)
            return [material_o, material_o, material_e]
        except:
            try:
                page_a, page_b, page_c = page + "-alpha", page + "-beta", page + "-gamma"
                print(page_a)
                material_alpha = RefractiveIndexMaterial(shelf, book, page_a)
                material_beta = RefractiveIndexMaterial(shelf, book, page_b)
                material_gamma = RefractiveIndexMaterial(shelf, book, page_c)
                return [material_alpha, material_beta, material_gamma]
            except:
                print(f'Warning: Given material is not known to be anisotropic in the Refractiveindex.info database. You should try to remove "ANI" keyword in material definition or to spellcheck the given path.')
