import numpy as np
import gc
import psutil


def save_data():
    anwser = np.loadtxt(open("D://topcoder//trainingAnswers.csv", "rb"), delimiter=",", skiprows=1)
    mat = None
    for i in range(100001, 109701):
        my_matrix = np.loadtxt(open("D://topcoder//training//" + str(i) + ".csv", "rb"), delimiter=",", skiprows=0)
        np.reshape(my_matrix, (-1, 2))
        my_matrix = my_matrix[0:8000, :]
        my_matrix = np.reshape(my_matrix, (1, 8000, 2))
        if mat is None:
            mat = my_matrix
        else:
            mat = np.concatenate((mat, my_matrix), axis=0)
        # print(i)
        # info = psutil.virtual_memory()
        # print(u'内存占比：', info.percent)
        del my_matrix
        gc.collect()

    np.savez('train.npz', train_x=mat, train_y=anwser)


def main():
    save_data()


if __name__ == "__main__":
    main()
