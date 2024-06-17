# import pprint, pickle
# # import numpy as np
# # parameter_values = []
# # with open(f'simulation_parameters_1718614141.581343.txt', 'r') as f:
# #     for line in f:
# #         # a = f.split(':')[1].split('\n')[0]
# #         parameter_values.append(line.split(':')[1].split('\n')[0])
# #     self.seed = parameter_values[0]
# #     self.height = parameter_values[1]
# #     self.width = parameter_values[2]
# #     self.number_births = parameter_values[3]
# #     self.number_deaths = parameter_values[4]
# #     self.k = parameter_values[5]
# #     self.tau = parameter_values[6]
# #     self.gamma = parameter_values[7]
# #     self.D = parameter_values[8]
# #     self.h = parameter_values[9]
# #     self.lam = parameter_values[10]
# #     self.phi_c = parameter_values[11]
# #     self.num_iterations = parameter_values[12]

# # # ECM_file = open('ecm_layers_data_1718614141.581343.pkl', 'rb')
# # # nutrient_file = open('nutrient_field_data_1718614141.581343.pkl', 'rb')
# # # N_T_file = open('n_t_data_1718614141.581343.pkl', 'rb')
# # birth_file = open('Births_data_1718614141.5947258.pkl', 'rb')
# # # death_file = open('deaths_data_1718614141.581343.pkl', 'rb')

# # # ECM_file_pkl = np.array(pickle.load(ECM_file))
# # # nutrient_file_pkl = np.array(pickle.load(nutrient_file))
# # # N_T_file_pkl = np.array(pickle.load(N_T_file)
# # print(pickle.load(birth_file))
# # # death_file_pkl = np.array(pickle.load(death_file))

# # # nutrient_file.close()
# # # nutrient_file.close()
# # # N_T_file.close()
# # birth_file.close()
# # # death_file.close()

# pkl_file = open('Births_data_1718614141.5947258.pkl', 'rb')

# data1 = pickle.load(pkl_file)
# pprint.pprint(data1)

# pkl_file.close()

import pickle
import numpy as np
# pkl_file = open('Nutrient_layers_data_1718614141.5947258.pkl', 'rb')
with open('Nutrient_layers_data_1718614141.5947258.pkl','rb') as f:
    unpickled_array = pickle.load(f)
    print('Array shape: '+str(unpickled_array.shape))
    print('Data type: '+str(type(unpickled_array)))
# data1 = np.array(pickle.load(pkl_file))
# print(data1)

# pkl_file.close()