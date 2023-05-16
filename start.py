from kernel_methods import KernelSVC
import numpy as np
import pandas as pd

kernel_X = np.load("coeffs/kernel_train_5.npy")
alpha = np.load("coeffs/dual_coeff_5.npy")
support = np.load("coeffs/support_5.npy")
kernel_test = np.load("coeffs/kernel_test_5.npy")

np.savez_compressed("coeffs/kernel_train_5.npz", kernel_X)

# svm = KernelSVC(C = 1, kernel = None, epsilon=1e-3, precomputed=kernel_X, class_weight = "balanced")
# svm.alpha = alpha
# svm.support_indices = support

# preds = -svm.predict(kernel_test)[:,0]

# kernel_X = np.load("coeffs/kernel_train_all.npy")
# alpha = np.load("coeffs/dual_coeff_all_1.npy")
# support = np.load("coeffs/support_all_1.npy")
# kernel_test = np.load("coeffs/kernel_test_all.npy")

# svm = KernelSVC(C = 1, kernel = None, epsilon=1e-3, precomputed=kernel_X, class_weight = "balanced")
# svm.alpha = alpha
# svm.support_indices = support

# preds -= .6*svm.predict(kernel_test)[:,0]

# dataframe = pd.DataFrame({"Predicted": preds}) 
# dataframe.index += 1


# dataframe.to_csv('final_result.csv',index_label='Id')
