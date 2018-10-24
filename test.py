import scipy.io
import numpy as np

dataset_list = ['banknote_authentication', 'image', 'blood_transfusion', 'breast_cancer', 'cancer', 'cleverland_heart', 'diabetis',
                    'flare_solar', 'german', 'heart', 'hepatitis', 'housing', 'liver', 'parkinsons', 'pima', 'sonar', 'thyroid'  ,
                    'titanic', 'wdbc', 'wholesale_customers', 'glass', 'ionosphere']
data_set = scipy.io.loadmat('data/wdbc.mat')
# print(data)
data = data_set['A']
lables = data_set['d']

data_arr = np.array(data)
print(data_arr.shape)
lables_arr = np.array(lables)
# len_lables_arr = np.sum(lables_arr == 1)
# print(len_lables_arr)

all_data = np.concatenate((data_arr, lables_arr), axis=1)

# all_p_data = np.extract((all_data[:, -1] == 1), all_data)
# print(all_p_data.size)
all_p_data = [example for example in all_data if all_data.any(all_data[:, -1] == 1)]
print(all_p_data)
