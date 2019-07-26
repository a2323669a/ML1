import pandas
import numpy as np

df_data = pandas.read_csv("./data/PM2.5/train.csv",encoding='Big5')

df_data = df_data[df_data['測項'] == 'PM2.5']

df_data = df_data.iloc[:,3:]

data_pm = df_data.values.astype(np.float).reshape((-1,))

epochs = 10000
w = [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
b = 0.
lr = 10

data_x = []
data_y = []

for i in range(len(data_pm) - 10):
    data_x.append(data_pm[i:i + 9])
    data_y.append([data_pm[i + 9]])

data_x = np.asarray(data_x)
data_y = np.asarray(data_y)

def shuffle(vali_ratio = 0.2):
    seed = np.random.randint(0,1e6)

    np.random.seed(seed)
    np.random.shuffle(data_x)
    np.random.seed(seed)
    np.random.shuffle(data_y)

    end = int(len(data_y) * (1 - vali_ratio))

    return data_x[:end],data_y[:end],data_x[end:],data_y[end:]

def accuarcy(vali_x,vali_y):
    prod_y = np.sum(np.multiply(w,vali_x),axis=1) + b
    pred_y = (prod_y).astype(np.int)
    return np.mean(np.equal(pred_y,vali_y))

def loss_img(vali_x,vali_y):
    prod_y = np.sum(np.multiply(w, vali_x), axis=1) + b
    pred_y = (prod_y).astype(np.int).reshape([-1,1])

    return np.abs(pred_y - vali_y).reshape((-1,)).tolist()

def loss(xs,ys):
    t_sum = np.sum(np.multiply(w,xs),axis=1).reshape([-1,1])#must reshape from (n,) to [n,1]
    k = (ys - t_sum - b)
    return np.mean( k ** 2)

def loss_mean(xs,ys):
    t_sum = np.sum(np.multiply(w, xs), axis=1).reshape([-1, 1])  # must reshape from (n,) to [n,1]
    k = (ys - t_sum - b)
    return np.mean(abs(k))

w_g_square_sum = np.zeros((9,))
b_g_square_sum = 0

for epoch in range(epochs):
    #normal

    xs,ys,vali_xs,vali_ys = shuffle()
    xs = data_x
    ys = data_y

    t_k = ys - np.sum(np.multiply(w,xs),axis=1).reshape([-1,1]) - b #otherwise will [n,1] - (n,),result is [n,n]

    g_k = np.multiply(t_k,-xs)

    w_g = 2 * np.sum(g_k,axis=0)

    b_g = 2 * np.sum(t_k * -1.)

    w_g_square_sum = (w_g_square_sum + w_g ** 2)
    b_g_square_sum += b_g ** 2



    w1 = w - (lr) * (w_g / np.sqrt(w_g_square_sum))
    b1 = b - (lr) * (b_g / np.sqrt(b_g_square_sum))

    w = w1
    b = b1

    print("loss={:.5f}".format(loss_mean(vali_xs, vali_ys)))

print(w)
print(b)
