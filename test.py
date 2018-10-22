import scipy.io

dataset_list = ['banknote_authentication', 'image', 'blood_transfusion', 'breast_cancer', 'cancer', 'cleverland_heart', 'diabetis',
                    'flare_solar', 'german', 'heart', 'hepatitis', 'housing', 'liver', 'parkinsons', 'pima', 'sonar', 'thyroid'  ,
                    'titanic', 'wdbc', 'wholesale_customers', 'glass', 'ionosphere']
data = scipy.io.loadmat('data/banknote_authentication.mat')
print(data)