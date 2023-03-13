import numpy as np


def random_classes_except_current(y_test, n_cls):
    y_test_new = np.zeros_like(y_test)
    for i_img in range(y_test.shape[0]):
        # lst_classes = list(range(n_cls))
        # lst_classes.remove(y_test[i_img])
        # y_test_new[i_img] = np.random.choice(lst_classes)
        if y_test[i_img] + 1 > n_cls:
            y_test_new[i_img] = 0
        else:
            y_test_new[i_img] = y_test[i_img] + 1
    return y_test_new
