import os
import numpy as np
from RVFL import RVFL_train
import scipy.io

directory = '.'
file_list = os.listdir(directory)

for file_name in file_list:
    if file_name.endswith(".mat"):
        file_path = os.path.join(directory, file_name)
        print(directory)
        print(file_name)
        file_data = scipy.io.loadmat(file_path)
        
        file_data1 = file_data['X1']
        file_data2 = file_data['X2']
        label = file_data['y']

        m, n = file_data1.shape
        for i in range(m):
            if label[i] == -1:
                label[i] = 0

        file_data1= np.hstack((file_data1, label))
        file_data2= np.hstack((file_data2, label))
        np.random.seed(0)
        indices = np.random.permutation(m)
        #View A
        file_data1 = file_data1[indices]
        A_train=file_data1[0:int(m*(1-0.30))]
        A_test=file_data1[int(m * (1-0.30)):]
        
        #View 2
        file_data2 = file_data2[indices]
        B_train=file_data2[0:int(m*(1-0.30))]
        B_test=file_data2[int(m * (1-0.30)):] 

        m, n = A_train.shape
        # View A Normalize
        x1 = A_train[:, 0:n-1]
        y1 = A_train[:, n-1]
        xtest0 = A_test[:,0:n-1]
        ytest0 = A_test[:,n-1]
        # Normalize the data training and testing
        me = np.tile(np.mean(x1, axis=0), (x1.shape[0], 1))
        st = np.tile(np.std(x1, axis=0), (x1.shape[0], 1))
        tme = np.tile(np.mean(x1, axis=0), (xtest0.shape[0], 1))
        tst = np.tile(np.std(x1, axis=0), (xtest0.shape[0], 1))
        x1 = (x1 - me) / st
        xtest0 = (xtest0 - tme) / tst

        A_test = np.hstack((xtest0, ytest0.reshape(ytest0.shape[0], 1)))
        A_train = np.hstack((x1, y1.reshape(y1.shape[0], 1)))

        # View B Normalize
        m, n = B_train.shape
        x1 = B_train[:, 0:n-1]
        y1 = B_train[:, n-1]
        xtest0 = B_test[:,0:n-1]
        ytest0 = B_test[:,n-1]

        # Normalize the data training and testing
        me = np.tile(np.mean(x1, axis=0), (x1.shape[0], 1))
        st = np.tile(np.std(x1, axis=0), (x1.shape[0], 1))
        tme = np.tile(np.mean(x1, axis=0), (xtest0.shape[0], 1))
        tst = np.tile(np.std(x1, axis=0), (xtest0.shape[0], 1))
        x1 = (x1 - me) / st
        xtest0 = (xtest0 - tme) / tst

        B_test = np.hstack((xtest0, ytest0.reshape(ytest0.shape[0], 1)))
        B_train = np.hstack((x1, y1.reshape(y1.shape[0], 1)))

        c1 = 0.00001
        c2 = 1000
        rho =0.01
        N = 23
        Act = 2
       
        Eval, Test_time = RVFL_train(A_train, B_train, A_test, B_test, c1,c1, c2,rho, N,Act)
        print("Testing Acc:",Eval[0])

        
