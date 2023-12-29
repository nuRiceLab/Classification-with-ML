import numpy as np
import tensorflow as tf
from tensorflow import keras
import zlib
import glob



class DataGenerator(keras.utils.Sequence):
    def __init__(self, files, batch_size, dim, n_channels):
        self.files = files
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels

    # __normalize__ method removed
    
    def get_info(self, file):
        with open(file, 'rb') as info_file:
            info = info_file.readlines()
            truth = {}
            truth['NuPDG'] = int(info[7].strip())
            truth['NuEnergy'] = float(info[1])
            truth['LepEnergy'] = float(info[2])
            truth['Interaction'] = int(info[0].strip()) % 4
            truth['NProton'] = int(info[8].strip())
            truth['NPion'] = int(info[9].strip())
            truth['NPizero'] = int(info[10].strip())  
            truth['NNeutron'] = int(info[11].strip())
            truth['is_antineutrino'] = int(int(info[7].strip())<0)
            
        return truth
    
    def get_pixels_map(self, file_name):
        cells = self.dim[0]
        planes = self.dim[1]
        views = self.n_channels
        file = open(file_name, 'rb').read()
        pixels_map = np.frombuffer(zlib.decompress(file), dtype=np.uint8)
        pixels_map = pixels_map.reshape(views, planes, cells)
        return pixels_map
    
    def get_data_and_labels(self, files):
        data_maps = []
        num_class_tasks = 6
        data_labels = np.zeros((len(files),num_class_tasks), dtype=np.int32)
        # data_labels = np.array()
        for i, file in enumerate(files):
            truth_info = self.get_info(file)
            pdg = np.abs(truth_info['NuPDG'])
            if pdg == 1:
                interaction_label = 0 #NC
            elif pdg == 12:
                interaction_label = 1 #CC nu_e
            elif pdg == 14:
                interaction_label = 2 #CC nu_mu
            elif pdg == 16:
                interaction_label = 3 #CC nu_tau (not included in our E range but still here) 

            proton_label = np.clip(truth_info['NProton'], None, 3) # 0,1,2,or N Protons 
            pion_label = np.clip(truth_info['NPion'], None, 3) # 0,1,2,or N Pions 
            pizero_label = np.clip(truth_info['NPizero'], None, 3) # 0,1,2,or N Pizeros
            neutron_label = np.clip(truth_info['NNeutron'], None, 3) # 0,1,2,or N Neutrons 
            anti_label = truth_info['is_antineutrino']

            truth_labels = [interaction_label, 
                            proton_label, 
                            pion_label, pizero_label, neutron_label, anti_label
                           ]
            data_labels[i] = truth_labels
            #image loading part 
            image = file[:-5] + '.gz'
            data_maps.append(self.get_pixels_map(image))
        data_label_dict = {"flavour":np.array(data_labels[:,0]), 
                           "protons": data_labels[:,1], 
                           "pions": data_labels[:,2], "pizeros": data_labels[:,3], "neutrons": data_labels[:,4], "is_antineutrino": data_labels[:,5]
                          }
        return data_maps, data_label_dict
    
    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = list(range(index * self.batch_size, (index + 1) * self.batch_size))

        files_temp = [self.files[i] for i in indexes]

        maps_temp, labels_temp = self.get_data_and_labels(files_temp)
        maps_z_view = np.asarray(maps_temp)[:, 2:]
        maps_v_view = np.asarray(maps_temp)[:, 1:2]
        maps_u_view = np.asarray(maps_temp)[:, 0:1]

        train_temp = []
        if self.n_channels == 1:
            for i in range(len(maps_z_view)):
                train_temp.append(maps_z_view[i][0])

        elif self.n_channels == 3:
            for i in range(len(maps_z_view)):
                train_temp.append(np.dstack((maps_u_view[i][0], maps_v_view[i][0], maps_z_view[i][0])))
        train_temp = np.array(train_temp).reshape([self.batch_size, self.dim[0], self.dim[1], self.n_channels])
        
        # labels_temp is a dictionary containing multiple output labels
        X, y = self.__data_generation(train_temp, labels_temp)
        return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.files))
    
    def __data_generation(self, train_temp, labels_temp):
        # Convert the labels_temp dictionary to a list of tensors
        y_list = [labels_temp[key] for key in labels_temp.keys()]
        return train_temp, y_list






