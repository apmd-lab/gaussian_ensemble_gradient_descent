import numpy as np
import os
directory = os.path.dirname(os.path.realpath(__file__))

class ref_index:
    def __init__(self, filename, target_wvl=None):
        """ target_wvl: the wavelength points to be interpolated to """

        if os.path.exists(directory[:-5] + "/material_data/" + filename + '.txt'):
            with open(directory[:-5] + "/material_data/" + filename + '.txt', 'r') as f:
                labels = f.readline()
                line_count = 0
                for line in f:
                    line_count += 1
                f.seek(0)
                f.readline()
                self.raw = np.zeros((line_count, 3))
                count = 0
                for line in f:
                    temp_array = np.array([line.split()])
                    self.raw[count,:] = np.asarray(temp_array, dtype=np.float64)
                    count += 1
        elif os.path.exists(directory[:-5] + "/material_data/" + filename + '.npz'):
            data = np.load(directory[:-5] + "/material_data/" + filename + '.npz')
            self.raw = np.zeros((data['lam'].size, 3))
            self.raw[:,0] = data['lam']
            self.raw[:,1] = data['n']
            self.raw[:,2] = data['k']
        if target_wvl is not None:
            new_n = np.interp(target_wvl, self.raw[:,0], self.raw[:,1])
            new_k = np.interp(target_wvl, self.raw[:,0], self.raw[:,2])
            self.new = np.stack((target_wvl, new_n, new_k), 1)
                
    def three_column(self):

        return self.new
    
    def two_column(self):
        
        temp_data = self.new.astype(complex)
        temp_data[:,1] = temp_data[:,1] + 1j*temp_data[:,2]
        return temp_data[:,:2]
    
    def wavelength(self):
        
        return self.raw[:,0]
    
    def n(self):
        
        return self.new[:,1]
    
    def k(self):
        
        return self.new[:,2]
    
    def n_k(self):
        
        temp_data = self.new.astype(complex)
        temp_data[:,1] = temp_data[:,1] + 1j*temp_data[:,2]
        return temp_data[:,1]
    
    def raw_n_k(self):
        temp_data = self.raw[:,1].astype(complex)
        temp_data = temp_data + 1j*self.raw[:,2]
        return self.raw[:,0], temp_data

def load_all(target_wavelength, function, material=np.array(['all'])):
    raw_wavelength = {}
    mat_dict = {}
    mat_directory = directory[:-5] + "/material_data/"
    if material[0] == 'all':
        mat_files = os.listdir(mat_directory)
        for filename in mat_files:
            wav = None
            if filename[-4:] == '.txt':
                if function == 'three_column':
                    RI = ref_index(filename[:-4], target_wavelength).three_column()
                elif function == 'two_column':
                    RI = ref_index(filename[:-4], target_wavelength).two_column()
                elif function == 'n':
                    RI = ref_index(filename[:-4], target_wavelength).n()
                elif function == 'k':
                    RI = ref_index(filename[:-4], target_wavelength).k()
                elif function == 'n_k':
                    RI = ref_index(filename[:-4], target_wavelength).n_k()
                elif function == 'raw_n_k':
                    wav, RI = ref_index(filename[:-4]).raw_n_k()
                else:
                    raise ValueError('Invalid Function')
                mat_dict[filename[:-4]] = RI
                raw_wavelength[filename[:-4]] = wav
    else:
        for mat in material:
            if mat != 'sample':
                wav = None
                if function == 'three_column':
                    RI = ref_index(mat, target_wavelength).three_column()
                elif function == 'two_column':
                    RI = ref_index(mat, target_wavelength).two_column()
                elif function == 'n':
                    RI = ref_index(mat, target_wavelength).n()
                elif function == 'k':
                    RI = ref_index(mat, target_wavelength).k()
                elif function == 'n_k':
                    RI = ref_index(mat, target_wavelength).n_k()
                elif function == 'raw_n_k':
                    wav, RI = ref_index(mat).raw_n_k()
                else:
                    raise ValueError('Invalid Function')
                mat_dict[mat] = RI
                raw_wavelength[mat] = wav
    return raw_wavelength, mat_dict