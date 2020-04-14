import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_data(do_suffle=False):
    data = pd.read_csv('./HousingData.csv')
    columns = data.columns.to_list()
    data = data.dropna().to_numpy()
    if do_suffle:
        ind = np.arange(len(data))
        #print(ind)
        np.random.shuffle(ind)
        data = data[ind]
    #print((len(data),1))
    bias = np.ones((len(data),1))
    tr_size = int(len(data)*3./4.)
    target = data[:,-1]
    input_data = data[:,:-1]
    input_data = np.concatenate((input_data,bias), 1) # 0 행단위 1 열단위

    tr_x = input_data[:tr_size]
    te_x = input_data[tr_size:]
    tr_t = target[:tr_size]
    te_t = target[tr_size:]

    return tr_x, tr_t, te_x, te_t, columns

def mse(y,t):
    return np.sum(np.square(y - t))/float(len(y))

def do_linear_regression(tr_x,tr_t,te_x,te_t,title=''):
    print('[*] do linear regression!(%s)'%title)
    print('[-] fit model...')
    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(tr_x.T,tr_x)),tr_x.T),tr_t)
    print("[-] optimal parameters : ", [p for p in w.tolist()])

    tr_y = np.matmul(tr_x,w)
    #print(len(tr_y))
    te_y = np.matmul(te_x,w)
    tr_mse = mse(tr_y,tr_t)
    te_mse = mse(te_y,te_t)

    print('[-] MSE(train) : ', tr_mse)
    print('[-] MSE(test) : ', te_mse)
    print()
    return tr_mse, te_mse

def main():
    tr_x, tr_t, te_x, te_t, columns = load_data(True)

    tr_mses = list()
    te_mses = list()
    ticks = list()

    tr_mse,te_mse = do_linear_regression(tr_x, tr_t, te_x, te_t,'using all features')
    tr_mses.append(tr_mse)
    te_mses.append(te_mse)
    ticks.append('using add features')

    for i in range (len(columns[:-1])):
        for j in range(len(columns[:-1])):
            tr_x_ex = np.delete(tr_x, i if i == j else [i,j],1)
            te_x_ex = np.delete(te_x, i if i == j else [i,j],1)

            tr_mse,te_mse = do_linear_regression(tr_x_ex, tr_t, te_x_ex, te_t,'remove %s col' %(columns[i] if i == j else [columns[i], columns[j]]))
            tr_mses.append(tr_mse)
            te_mses.append(te_mse)
            ticks.append(str(columns[i] if i==j else [columns[i],columns[j]]))
    fig = plt.figure()
    plt.plot(np.arange(len(tr_mses)),tr_mses,c='b', label='train mse')
    plt.plot(np.arange(len(te_mses)),te_mses,c='r', label='test mse')
    plt.legend()
    plt.xticks(np.arange(len(tr_mses)),ticks,rotation=45)
    fig.show()
    plt.show()

if __name__ == '__main__':
    main()