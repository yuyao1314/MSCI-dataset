from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Concatenate, GRU, Dropout, Activation, Reshape, \
    AveragePooling2D
from keras import applications
from keras.optimizers import SGD, adam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import *
# from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import glob, os
from keras.utils import np_utils, multi_gpu_model
from keras.callbacks import TensorBoard
from sklearn.metrics import classification_report
from collections import Counter
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import math
import csv
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import matplotlib.pyplot as plt
from scipy import interp



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def load_cnn_model():
    acetic_model = load_model('/home/som/lab/Documents/yuyao/123/yu/acetic_model.h5')
    green_model=load_model('/home/som/lab/Documents/yuyao/123/yu/green.h5')
    iodine_model=load_model('/home/som/lab/Documents/yuyao/123/yu/iodine_model.h5')
    return acetic_model,green_model,iodine_model

def load_lstm_model():
    lstm_model=load_model('/home/som/lab/Documents/yuyao/123/yu/GRU.h5')
    return lstm_model

def predict_lstm(model,predict_data):
    predict_data = predict_data.reshape(predict_data.shape[0] // 5, 5, predict_data.shape[1])
    scores = model.predict(predict_data, batch_size=16)
    return scores


def predict_cnn(model,predict_data):
    scores = model.predict(predict_data, batch_size=16)
    return scores


def train_model(train_data, train_labels, test_data, test_labels):
    model = Sequential()

    model.add(Conv2D(4, (1, 1), padding='same',
                     input_shape=train_data.shape[1:]))
    model.add(Activation('relu'))
    model.add(Flatten())
    # model.add(Dense(32))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    # 召回率不要
    history=model.fit(train_data, train_labels, validation_data=(test_data, test_labels), batch_size=32, epochs=40, verbose=1)

    model.save('.../C-GCNN.h5')
    scores = model.evaluate(test_data, test_labels, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    sscores = model.predict(test_data, batch_size=16)
    return sscores,history


def read_record(save_path):
    r = open(save_path, 'r')
    f = r.readlines()
    records = []
    for i in f:
        records.append(i.strip('\n').split('\t'))
    return records


def load_acetic_data(datapath, datatxt):
    records = read_record(datatxt)
    label = []
    acetic = []
    for i in range(0, len(records)):
        k = records[i][2]
        g = records[i][0]
        my_img = load_img(datapath + '/' + records[i][0] + '/' + records[i][1] + '/' + records[i][3])
        interimage = img_to_array(my_img, data_format='channels_last')
        label.append(int(g))
        if k == '0':
            acetic.append(interimage)
        elif k == '1':
            acetic.append(interimage)
        elif k == '2':
            acetic.append(interimage)
        elif k == '3':
            acetic.append(interimage)
        elif k == '4':
            acetic.append(interimage)

    label = label[0::7]
    print(label)
    print(len(label))
    acetic = np.array(acetic)
    print(acetic.shape)
    return acetic, label

def load_green_data(datapath, datatxt):
    records = read_record(datatxt)
    acetic = []
    for i in range(0, len(records)):
        k = records[i][2]
        g = records[i][0]
        my_img = load_img(datapath + '/' + records[i][0] + '/' + records[i][1] + '/' + records[i][3])
        interimage = img_to_array(my_img, data_format='channels_last')

        if k == '5':
            acetic.append(interimage)

    acetic = np.array(acetic)
    print(acetic.shape)
    return acetic

def load_iodine_data(datapath, datatxt):
    records = read_record(datatxt)
    acetic = []
    for i in range(0, len(records)):
        k = records[i][2]
        g = records[i][0]
        my_img = load_img(datapath + '/' + records[i][0] + '/' + records[i][1] + '/' + records[i][3])
        interimage = img_to_array(my_img, data_format='channels_last')

        if k == '6':
            acetic.append(interimage)


    acetic = np.array(acetic)
    print(acetic.shape)
    return acetic

def label_preprocess(label):
    label=np_utils.to_categorical(label, 4)
    return label

def img_preprocess(img):
    img = img.astype('float32')
    img /= 255
    mean1 = img.mean(axis=0)
    img -= mean1
    return img

def sen_spe(y_test,y_score,n_classes,save_path= None):
    # 计算每一类的ROC
    sen = [] #特异性
    spe = []  #敏感性
    y_pre = score_to_pre(y_score)
    for i in range(n_classes):
        temp_sen, temp_spe= cal_sen_spe(y_test[:, i], y_pre[:, i])
        sen.append(temp_sen)
        spe.append(temp_spe)
    avg_sen = sum(sen)/n_classes
    avg_spe = sum(spe)/n_classes
    print('sen:', sen)
    print('spe:', spe)
    print('\n avg_spe:%.6f, \n' % (avg_spe))
    print('\n avg_sen:%.6f, \n' % (avg_sen))
    if save_path == None:
        #excel 保存数据，后期画图
        pass
    else:
        pass
    return spe,sen,avg_sen,avg_spe

def cal_sen_spe(y_test, y_pre):
    T = sum(y_test)
    F = len(y_test) - T
    sen = sum(list(map(lambda x, y: 1 if x == y and x==1 else 0,y_test, y_pre)))/ T
    spe = sum(map(lambda x, y: 1 if x == y and x == 0 else 0,y_test, y_pre)) / F
    return sen ,spe

def score_to_pre(y):
    """
    根据概率计算预测值
    :param y:
    :return:
    """
    tmax = np.argmax(y,1)
    y_score = np.zeros_like(y)
    for i in range(len(tmax)):
        y_score[i][tmax[i]] = 1
    return y_score

def drawlines(history):
    history_dict=history.history
    csv_file = open('.../C-GCNN.csv', 'w', newline='')
    writer = csv.writer(csv_file)
    for key in history_dict:
        writer.writerow([key, history_dict[key]])
    csv_file.close()
    # loss_values=history_dict['loss']
    # acc_values=history_dict['acc']
    # val_loss_values=history_dict['val_loss']
    # val_acc_values=history_dict['val_acc']
    # epochs=range(1,len(loss_values)+1)

def roc(lstm_predictions,y_pre_temp):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(y_pre_temp[:, i], lstm_predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_pre_temp.ravel(), lstm_predictions.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(4):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= 4

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linewidth=1, alpha=0.6)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linewidth=1, alpha=0.6)

    plt.plot(fpr[0],tpr[0],color='r',label='ROC curve of class Normal (area = {1:0.2f})'
                       ''.format(0, roc_auc[0]), linewidth=1, alpha=0.6)
    plt.plot(fpr[1],tpr[1],color='green',label='ROC curve of class CIN 1 (area = {1:0.2f})'
                       ''.format(1, roc_auc[1]), linewidth=1, alpha=0.6)
    plt.plot(fpr[2],tpr[2],color='blue',label='ROC curve of class CIN 2/3 (area = {1:0.2f})'
                       ''.format(2, roc_auc[2]), linewidth=1, alpha=0.6)
    plt.plot(fpr[3],tpr[3],color='yellow',label='ROC curve of class Cancer (area = {1:0.2f})'
                       ''.format(3, roc_auc[3]), linewidth=1, alpha=0.6)
    csv_file1 = open('.../C-GCNN.csv', 'w', newline='')
    writer = csv.writer(csv_file1)
    for key in fpr:
        writer.writerow([key, fpr[key]])
    csv_file1.close()


    csv_file2 = open('.../C-GCNN.csv', 'w', newline='')
    writer = csv.writer(csv_file2)
    for key in tpr:
        writer.writerow([key, tpr[key]])
    csv_file2.close()

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FP Rate')
    plt.ylabel('TP Rate')
    plt.title('ROC curves of C-GCNN to classify four classes')
    plt.legend(loc="lower right")
    plt.show()

def assemble(lstm_data,iodine_data,green_data):
    print(lstm_data.shape,iodine_data.shape,green_data.shape)
    x_data = np.ndarray((green_data.shape[0]*3, 4), dtype=np.float64)
    for i in range(0,len(lstm_data)):
        x_data[3*i]=lstm_data[i]
        x_data[3*i+1]=green_data[i]
        x_data[3*i+2]=iodine_data[i]

    x_data = x_data.reshape(x_data.shape[0] // 3, 3, 4)
    x_data=x_data.reshape(x_data.shape+(1,))
    mean1 = x_data.mean(axis=0)
    x_data -= mean1

    print(x_data.shape)
    print('!'*32)
    return x_data



if __name__ == '__main__':
    trainpath = '.../train'
    testpath = '.../test'
    traintxt = '.../train_records.txt'
    testtxt = '.../test_records.txt'
    acetic_x_train, acetic_y_train = load_acetic_data(trainpath, traintxt)
    acetic_x_test, acetic_y_test = load_acetic_data(testpath, testtxt)
    green_x_train=load_green_data(trainpath, traintxt)
    green_x_test=load_green_data(testpath, testtxt)
    iodine_x_train=load_iodine_data(trainpath, traintxt)
    iodine_x_test=load_iodine_data(testpath, testtxt)
    acetic_cnn,green_cnn,iodine_cnn=load_cnn_model()
    acetic_y_train=label_preprocess(acetic_y_train)
    acetic_y_test=label_preprocess(acetic_y_test)
    acetic_x_train=img_preprocess(acetic_x_train)
    acetic_x_test = img_preprocess(acetic_x_test)
    green_x_train = img_preprocess(green_x_train)
    green_x_test = img_preprocess(green_x_test)
    iodine_x_train = img_preprocess(iodine_x_train)
    iodine_x_test = img_preprocess(iodine_x_test)

    print("-" * 10, "acetic cnn", "-" * 10)
    acetic_predictions1 = predict_cnn(acetic_cnn, acetic_x_train)
    acetic_predictions2 = predict_cnn(acetic_cnn, acetic_x_test)

    print("-" * 10, "green cnn", "-" * 10)
    green_predictions1=predict_cnn(green_cnn, green_x_train)
    green_predictions2 = predict_cnn(green_cnn, green_x_test)

    print("-" * 10, "iodine cnn", "-" * 10)
    iodine_prediction1=predict_cnn(iodine_cnn,iodine_x_train)
    iodine_prediction2=predict_cnn(iodine_cnn,iodine_x_test)

    lstm_model=load_lstm_model()
    print("-" * 10, "GRU", "-" * 10)
    lstm_predictions1 =predict_lstm(lstm_model, acetic_predictions1)
    lstm_predictions2 = predict_lstm(lstm_model, acetic_predictions2)

    x_train=assemble(lstm_predictions1,iodine_prediction1,green_predictions1)
    x_test= assemble(lstm_predictions2, iodine_prediction2, green_predictions2)

    y_prediction, history=train_model(x_train, acetic_y_train, x_test,acetic_y_test)
    drawlines(history)
    roc(y_prediction, acetic_y_test)

    spe, sen, avg_sen, avg_spe=sen_spe(acetic_y_test,y_prediction,4)


