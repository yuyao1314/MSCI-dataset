from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Concatenate, LSTM, Dropout, Activation, Reshape, \
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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def load_cnn_model():
    model = load_model('.../lstm_model.h5')
    return model


def predict_score(model,predict_data):
    scores = model.predict(predict_data, batch_size=16)
    return scores


def cnn_results_check(cnn_predictions):
    print(cnn_predictions.shape)
    print(cnn_predictions)
    print('.' * 32)
    cnn_predictions = cnn_predictions.reshape(cnn_predictions.shape[0] // 5, 5, cnn_predictions.shape[1])
    print(cnn_predictions.shape)
    print(cnn_predictions)
    print('.' * 32)
    return cnn_predictions


def train_model(train_data, train_labels, test_data, test_labels):
    model = Sequential()
    model.add(LSTM(256, dropout=0.2,input_shape=(train_data.shape[1],
                                                  train_data.shape[2]), name='lstm1'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # 召回率不要
    history=model.fit(train_data, train_labels, validation_data=(test_data, test_labels), batch_size=16, epochs=40, verbose=1)

    model.save('.../lstm.h5')
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


def load_img_label_data(datapath, datatxt):
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


def extract_features_and_store(x_train, y_train, x_test, y_test):
    y_train = np_utils.to_categorical(y_train, 4)
    y_test = np_utils.to_categorical(y_test, 4)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    mean1 = x_train.mean(axis=0)
    mean2 = x_test.mean(axis=0)
    x_train -= mean1
    x_test -= mean2
    return x_train, x_test, y_train, y_test

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
    csv_file = open('.../lstm.csv', 'w', newline='')
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
    csv_file1 = open('.../lstm.csv', 'w', newline='')
    writer = csv.writer(csv_file1)
    for key in fpr:
        writer.writerow([key, fpr[key]])
    csv_file1.close()


    csv_file2 = open('.../lstm.csv', 'w', newline='')
    writer = csv.writer(csv_file2)
    for key in tpr:
        writer.writerow([key, tpr[key]])
    csv_file2.close()


    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves of LSTM to classify four classes')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    trainpath = '.../train'
    testpath = '.../test'
    traintxt = '.../train_records.txt'
    testtxt = '.../test_records.txt'
    x_train, y_train = load_img_label_data(trainpath, traintxt)
    x_test, y_test = load_img_label_data(testpath, testtxt)
    train_data, test_data, train_labels, test_labels = extract_features_and_store(x_train, y_train, x_test, y_test)
    print("-" * 10, "time 1", "-" * 10)
    cnn_model = load_cnn_model()
    cnn_predictions1 = predict_score(cnn_model, train_data)
    cnn_predictions2 = predict_score(cnn_model, test_data)
    lstm_input_data1 = cnn_results_check(cnn_predictions1)
    lstm_input_data2 = cnn_results_check(cnn_predictions2)

    y_prediction, history=train_model(lstm_input_data1, train_labels, lstm_input_data2, test_labels)
    drawlines(history)
    roc(y_prediction, y_test)
    spe, sen, avg_sen, avg_spe=sen_spe(y_test,y_prediction,4)



