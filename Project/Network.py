# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 22:19:42 2021

@author: ilya
"""
import numpy as np
from tqdm import tqdm
import gzip
import pickle
from numba import jit

def convolution(image, filt, bias, s=1):
    '''
    Свертка фильтра с изображением с отступом s
    '''
    (n_f, n_c_f, f, _) = filt.shape
    n_c, in_dim, _ = image.shape
    
    out_dim = int((in_dim - f)/s) + 1
    
    out = np.zeros((n_f,out_dim,out_dim))
    
    for curr_f in range(n_f):
        curr_y = out_y = 0
        while curr_y + f <= in_dim:
            curr_x = out_x = 0
            while curr_x + f <= in_dim:
                out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:,curr_y:curr_y+f, curr_x:curr_x+f]) + bias[curr_f]
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        
    return out

@jit(nopython = True)
def maxpool(image, f=2, s=2):
    '''
    Сжатие картинки maxpooling'ом размера f с отступом s
    '''
    n_c, h_prev, w_prev = image.shape
    
    # размеры на выходе
    h = int((h_prev - f)/s)+1 
    w = int((w_prev - f)/s)+1

    downsampled = np.zeros((n_c, h, w)) 
    
    for i in range(n_c):
        curr_y = out_y = 0
        while curr_y + f <= h_prev:
            curr_x = out_x = 0
            while curr_x + f <= w_prev:
                downsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y+f, curr_x:curr_x+f])
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    return downsampled

@jit(nopython = True)
def softmax(raw_preds):
    '''
    Функция активации softmax
    '''
    out = np.exp(raw_preds)
    return out/np.sum(out)

@jit(nopython = True)
def categoricalCrossEntropy(probs, label):
    '''
    Вычисление функции потерь
    '''
    return -np.sum(label * np.log(probs))

def extract_data(filename, num_images, IMAGE_WIDTH):
    '''
    Преобразование файла с выборкой к массиву
    '''
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_WIDTH*IMAGE_WIDTH)
        return data


def extract_labels(filename, num_images):
    '''
    Преобразование файла с таргетом к массиву
    '''
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def initializeFilter(size, scale = 1.0):
    '''
    Инициализация фильтра
    '''
    stddev = scale/np.sqrt(np.prod(size))
    return np.random.normal(loc = 0, scale = stddev, size = size)

def initializeWeight(size):
    '''
    Инициализация весов
    '''
    return np.random.standard_normal(size=size) * 0.01

def convolutionBackward(dconv_prev, conv_in, filt, s):
    '''
    Backpropagation через сверточный слой
    '''
    (n_f, n_c, f, _) = filt.shape
    (_, orig_dim, _) = conv_in.shape
    dout = np.zeros(conv_in.shape) 
    dfilt = np.zeros(filt.shape)
    dbias = np.zeros((n_f,1))
    for curr_f in range(n_f):
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                dfilt[curr_f] += dconv_prev[curr_f, out_y, out_x] * conv_in[:, curr_y:curr_y+f, curr_x:curr_x+f]
                dout[:, curr_y:curr_y+f, curr_x:curr_x+f] += dconv_prev[curr_f, out_y, out_x] * filt[curr_f]
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        dbias[curr_f] = np.sum(dconv_prev[curr_f])
    return dout, dfilt, dbias


def nanargmax(arr):
    '''
    Возвращает индекс наибольшего числа в массиве.
    '''
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs 
    
def maxpoolBackward(dpool, orig, f, s):
    '''
    Backpropagation через maxpooling слой.
    '''
    (n_c, orig_dim, _) = orig.shape
    
    dout = np.zeros(orig.shape)
    
    for curr_c in range(n_c):
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                (a, b) = nanargmax(orig[curr_c, curr_y:curr_y+f, curr_x:curr_x+f])
                dout[curr_c, curr_y+a, curr_x+b] = dpool[curr_c, out_y, out_x]
                
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        
    return dout

def conv(image, label, params, conv_s, pool_f, pool_s):
    
    [f1, f2, w3, w4, b1, b2, b3, b4] = params 
    
    conv1 = convolution(image, f1, b1, conv_s)
    conv1[conv1<=0] = 0
    
    conv2 = convolution(conv1, f2, b2, conv_s)
    conv2[conv2<=0] = 0
    
    pooled = maxpool(conv2, pool_f, pool_s)
    
    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2 * dim2 * dim2, 1))
    
    z = w3.dot(fc) + b3
    z[z<=0] = 0
    
    out = w4.dot(z) + b4
     
    probs = softmax(out)
    
    loss = categoricalCrossEntropy(probs, label)
        
    dout = probs - label
    dw4 = dout.dot(z.T)
    db4 = np.sum(dout, axis = 1).reshape(b4.shape)
    
    dz = w4.T.dot(dout)
    dz[z<=0] = 0
    dw3 = dz.dot(fc.T)
    db3 = np.sum(dz, axis = 1).reshape(b3.shape)
    
    dfc = w3.T.dot(dz)
    dpool = dfc.reshape(pooled.shape)
    
    dconv2 = maxpoolBackward(dpool, conv2, pool_f, pool_s)
    dconv2[conv2<=0] = 0
    
    dconv1, df2, db2 = convolutionBackward(dconv2, conv1, f2, conv_s)
    dconv1[conv1<=0] = 0
    
    dimage, df1, db1 = convolutionBackward(dconv1, image, f1, conv_s)
    
    grads = [df1, df2, dw3, dw4, db1, db2, db3, db4] 
    
    return grads, loss

def adamGD(batch, num_classes, lr, dim, n_c, beta1, beta2, params, cost):
    '''
    Обновляет параметры согласно методу AdamGrad
    '''
    [f1, f2, w3, w4, b1, b2, b3, b4] = params
    
    X = batch[:,0:-1]
    X = X.reshape(len(batch), n_c, dim, dim)
    Y = batch[:,-1]
    
    cost_ = 0
    batch_size = len(batch)
    
    df1 = np.zeros(f1.shape)
    df2 = np.zeros(f2.shape)
    dw3 = np.zeros(w3.shape)
    dw4 = np.zeros(w4.shape)
    db1 = np.zeros(b1.shape)
    db2 = np.zeros(b2.shape)
    db3 = np.zeros(b3.shape)
    db4 = np.zeros(b4.shape)
    
    v1 = np.zeros(f1.shape)
    v2 = np.zeros(f2.shape)
    v3 = np.zeros(w3.shape)
    v4 = np.zeros(w4.shape)
    bv1 = np.zeros(b1.shape)
    bv2 = np.zeros(b2.shape)
    bv3 = np.zeros(b3.shape)
    bv4 = np.zeros(b4.shape)
    
    s1 = np.zeros(f1.shape)
    s2 = np.zeros(f2.shape)
    s3 = np.zeros(w3.shape)
    s4 = np.zeros(w4.shape)
    bs1 = np.zeros(b1.shape)
    bs2 = np.zeros(b2.shape)
    bs3 = np.zeros(b3.shape)
    bs4 = np.zeros(b4.shape)
    
    for i in range(batch_size):
        
        x = X[i]
        y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1)
        
        grads, loss = conv(x, y, params, 1, 2, 2)
        [df1_, df2_, dw3_, dw4_, db1_, db2_, db3_, db4_] = grads
        
        df1+=df1_
        db1+=db1_
        df2+=df2_
        db2+=db2_
        dw3+=dw3_
        db3+=db3_
        dw4+=dw4_
        db4+=db4_

        cost_+= loss

        
    v1 = beta1*v1 + (1-beta1)*df1/batch_size
    s1 = beta2*s1 + (1-beta2)*(df1/batch_size)**2
    f1 -= lr * v1/np.sqrt(s1+1e-7)
    
    bv1 = beta1*bv1 + (1-beta1)*db1/batch_size
    bs1 = beta2*bs1 + (1-beta2)*(db1/batch_size)**2
    b1 -= lr * bv1/np.sqrt(bs1+1e-7)
   
    v2 = beta1*v2 + (1-beta1)*df2/batch_size
    s2 = beta2*s2 + (1-beta2)*(df2/batch_size)**2
    f2 -= lr * v2/np.sqrt(s2+1e-7)
                       
    bv2 = beta1*bv2 + (1-beta1) * db2/batch_size
    bs2 = beta2*bs2 + (1-beta2)*(db2/batch_size)**2
    b2 -= lr * bv2/np.sqrt(bs2+1e-7)
    
    v3 = beta1*v3 + (1-beta1) * dw3/batch_size
    s3 = beta2*s3 + (1-beta2)*(dw3/batch_size)**2
    w3 -= lr * v3/np.sqrt(s3+1e-7)
    
    bv3 = beta1*bv3 + (1-beta1) * db3/batch_size
    bs3 = beta2*bs3 + (1-beta2)*(db3/batch_size)**2
    b3 -= lr * bv3/np.sqrt(bs3+1e-7)
    
    v4 = beta1*v4 + (1-beta1) * dw4/batch_size
    s4 = beta2*s4 + (1-beta2)*(dw4/batch_size)**2
    w4 -= lr * v4 / np.sqrt(s4+1e-7)
    
    bv4 = beta1*bv4 + (1-beta1)*db4/batch_size
    bs4 = beta2*bs4 + (1-beta2)*(db4/batch_size)**2
    b4 -= lr * bv4 / np.sqrt(bs4+1e-7)
    

    cost_ = cost_/batch_size
    cost.append(cost_)

    params = [f1, f2, w3, w4, b1, b2, b3, b4]
    
    return params, cost

def train(num_classes = 10, lr = 0.01, beta1 = 0.95, beta2 = 0.99, img_dim = 28, img_depth = 1, f = 5, num_filt1 = 8, num_filt2 = 8, batch_size = 32, num_epochs = 3, save_path = 'params.pkl'):

    X = extract_data('train-images-idx3-ubyte.gz', 60000, img_dim)
    y_dash = extract_labels('train-labels-idx1-ubyte.gz', 60000).reshape(60000,1)
    X-= int(np.mean(X))
    X/= int(np.std(X))
    train_data = np.hstack((X,y_dash))

    np.random.shuffle(train_data)

    f1, f2, w3, w4 = (num_filt1 ,img_depth,f,f), (num_filt2 ,num_filt1,f,f), (128,800), (10, 128)
    f1 = initializeFilter(f1)
    f2 = initializeFilter(f2)
    w3 = initializeWeight(w3)
    w4 = initializeWeight(w4)

    b1 = np.zeros((f1.shape[0],1))
    b2 = np.zeros((f2.shape[0],1))
    b3 = np.zeros((w3.shape[0],1))
    b4 = np.zeros((w4.shape[0],1))

    params = [f1, f2, w3, w4, b1, b2, b3, b4]

    cost = []

    for epoch in range(num_epochs):
        np.random.shuffle(train_data)
        batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]

        t = tqdm(batches)
        for x,batch in enumerate(t):
            params, cost = adamGD(batch, num_classes, lr, img_dim, img_depth, beta1, beta2, params, cost)
            t.set_description("Loss: %.2f" % (cost[-1]))
            

    with open(save_path, 'wb') as file:
        pickle.dump(params, file)
        
    return cost

def predict(image, f1, f2, w3, w4, b1, b2, b3, b4, conv_s = 1, pool_f = 2, pool_s = 2):
    '''
    Прогнозирует таргеты и вероятности для подаваемой выборки
    '''
    conv1 = convolution(image, f1, b1, conv_s)
    conv1[conv1<=0] = 0
    
    conv2 = convolution(conv1, f2, b2, conv_s)
    conv2[conv2<=0] = 0 
    
    pooled = maxpool(conv2, pool_f, pool_s)
    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2 * dim2 * dim2, 1))
    
    z = w3.dot(fc) + b3
    z[z<=0] = 0
    
    out = w4.dot(z) + b4
    probs = softmax(out)
    
    return np.argmax(probs), np.max(probs)