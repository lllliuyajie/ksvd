import scipy.io as sio
import numpy as np
import sklearn.preprocessing as skp
from sklearn.model_selection import train_test_split
import OMP

# cancer数据最后一列是标签
def extract_data():
    data_list = ['cancer', 'Glass', 'wdbc']
    len_dataset = len(data_list)
    p_list = []
    n_list = []

    #  读取其中一个数据集
    one_dataset = sio.loadmat('data/'+data_list[0]+'.mat')   # 字典形式
    data = one_dataset['A']
    lables = one_dataset['d']

    # 将字典转成 array
    data_array = np.array(data)
    lables_array = np.array(lables)
    all_data = np.concatenate((data_array, lables_array), axis=1)
    print('数据集维度: '+str(all_data.shape))
    for row in all_data:
        # 取出成列表形式
        if row[-1] == 1:
            p_list.append(row)
        elif row[-1] == -1:
            n_list.append(row)
        else:
            continue

    # 计算正类和负类分别占多少     Ture 可以当作1 false当作1
    p_arr = np.array(p_list)
    print('正类样例维度 ：'+str(p_arr.shape))
    n_arr = np.array(n_list)
    print('负类样例维度： '+str(n_arr.shape))
    len_p_lables = np.sum(lables_array == 1)
    print('正类样例的个数： '+str(len_p_lables))
    len_n_lables = np.sum(lables_array == -1)
    print('负类样例的个数： ' + str(len_n_lables))

    # 正类样本和负类样本
    p_data = p_arr[:, :-1]
    p_lable = np.array(p_arr[:, -1]).reshape((len_p_lables, 1))

    n_data = n_arr[:, :-1]
    n_lable = np.array(n_arr[:, -1]).reshape((len_n_lables, 1))

    # 数据预处理  标准化 归一化
    p_scale_data = skp.scale(p_data)
    p_all_data = np.concatenate((p_scale_data, p_lable), axis=1)  # 0按列，1按行

    n_scale_data = skp.scale(n_data)
    n_all_data = np.concatenate((n_scale_data, n_lable), axis=1)

    # 取出训练集（70%的训练集，0.05的异常点(11, 12)）和测试集
    # p_data_train, p_data_test, p_lable_train, p_lable_test = train_test_split(p_scale_data, p_lable, train_size=0.7)
    p_data_train, p_data_test = train_test_split(p_all_data, train_size=0.7)
    n_data_train, n_data_test = train_test_split(n_all_data, train_size=0.05, test_size=0.05)

    nall_train = np.concatenate((p_data_train, n_data_train), axis=0)
    all_train = np.random.permutation(nall_train)   # 组合训练集，并且打乱
    print(all_train.shape)
    nall_test = np.concatenate((p_data_test, n_data_test), axis=0)
    all_test = np.random.permutation(nall_test)     # 组合测试集并且打乱

    # len_data_tr = int(round(len_p_lables * 0.7))
    # print(len_data_tr)
    # len_abnormal_data = int(round (len_n_lables * 0.05))
    # print(len_abnormal_data)

    return all_train, all_test


if __name__ == '__main__':
    train_set, test_set = extract_data()
    OMP.omp()