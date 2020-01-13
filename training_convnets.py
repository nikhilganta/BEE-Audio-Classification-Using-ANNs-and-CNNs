import cv2
from scipy.io import wavfile
import pickle
import glob
import numpy as np
from sklearn.utils import shuffle
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from sklearn.externals import joblib

path = "/home/nikhilganta/Documents/Intelligent Systems/Project 1/workspace/"

def save(ann, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(ann, fp)

def load(file_name):
    with open(path + file_name, 'rb') as fp:
        nn = pickle.load(fp)
    return nn

# For BEE2_1S
beepath = '/home/kulyukin-lab1/Nikhil Project/BEE2_1S/'
nonbeepath = '/home/kulyukin-lab1/Nikhil Project/BEE2_1S/'
# For BEE2_2S
beepath = '/home/kulyukin-lab1/Nikhil Project/BEE2_2S/'
nonbeepath = '/home/kulyukin-lab1/Nikhil Project/BEE2_2S/'

def beesPreProcessor(dataType):
    dataX = []
    dataY = []
    global beepath
    global nonbeepath
    bees = glob.glob(beepath + '/*/'+dataType+'/bee/*/*.png')
    nonbees = glob.glob(nonbeepath + '/*/'+dataType+'/no_bee/*/*.png')
    filePaths = []
    for file in bees:
        filePaths.append((file,(1,0)))
    for file in nonbees:
        filePaths.append((file,(0,1)))
    filePaths = shuffle(filePaths)
    for i in range(len(filePaths)):
        image = cv2.imread(filePaths[i][0])
        scaled_image = image/255.0
        scaled_image = cv2.resize(scaled_image,(90,90))
        scaled_image = np.array(scaled_image)
        dataX.append(scaled_image)
        dataY.append(filePaths[i][1])
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    return dataX,dataY

def buzz1ValidPreProcessor():
    dataX = []
    dataY = []
    beepath = '/home/kulyukin-lab1/Nikhil Project/BUZZ1/out_of_sample_data_for_validation/'
    cricketpath = '/home/kulyukin-lab1/Nikhil Project/BUZZ1/out_of_sample_data_for_validation/'
    noisepath = '/home/kulyukin-lab1/Nikhil Project/BUZZ1/out_of_sample_data_for_validation/'
    bees = glob.glob(beepath + '/bee_test/*.wav')
    crickets = glob.glob(cricketpath + '/cricket_test/*.wav')
    noises = glob.glob(noisepath + '/noise_test/*.wav')
    filePaths = []
    for file in bees:
        filePaths.append((file,(1,0,0)))
    for file in crickets:
        filePaths.append((file,(0,1,0)))
    for file in noises:
        filePaths.append((file,(0,0,1)))
    filePaths = shuffle(filePaths)
    for i in range(len(filePaths)):
        samplerate, audio = wavfile.read(filePaths[i][0])
        audio = audio/float(np.max(audio))

        # method 1
        mid = len(audio)//2
        piece1 = audio[mid-22000:mid+1]
        piece2 = audio[mid+1:mid+22000]
        audio = np.concatenate((piece1,piece2))

        # method 2
        # if (len(audio) < 88000):
        #     val = 88000 - len(audio)
        #     audio = np.pad(audio,(0,val),'constant',constant_values = (0,0))
        # elif (len(audio) > 88000):
        #     audio = audio[0:88000]

        dataX.append(audio)
        dataY.append(filePaths[i][1])
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    return dataX,dataY

def buzz1TestTrainPreProcessor():
    dataX = []
    dataY = []
    beepath = '/home/kulyukin-lab1/Nikhil Project/BUZZ1/bee/'
    cricketpath = '/home/kulyukin-lab1/Nikhil Project/BUZZ1/cricket/'
    noisepath = '/home/kulyukin-lab1/Nikhil Project/BUZZ1/noise/'
    bees = glob.glob(beepath + '/*.wav')
    crickets = glob.glob(cricketpath + '/*.wav')
    noises = glob.glob(noisepath + '/*.wav')
    filePaths = []
    for file in bees:
        filePaths.append((file,(1,0,0)))
    for file in crickets:
        filePaths.append((file,(0,1,0)))
    for file in noises:
        filePaths.append((file,(0,0,1)))
    filePaths = shuffle(filePaths)
    for i in range(len(filePaths)):
        samplerate, audio = wavfile.read(filePaths[i][0])
        audio = audio/float(np.max(audio))

        # method 1
        mid = len(audio)//2
        piece1 = audio[mid-22000:mid+1]
        piece2 = audio[mid+1:mid+22000]
        audio = np.concatenate((piece1,piece2))

        # method 2
        # if (len(audio) < 88000):
        #     val = 88000 - len(audio)
        #     audio = np.pad(audio,(0,val),'constant',constant_values = (0,0))
        # elif (len(audio) > 88000):
        #     audio = audio[0:88000]

        dataX.append(audio)
        dataY.append(filePaths[i][1])
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    return dataX,dataY

def buzz2PreProcessor(dataType):
    dataX = []
    dataY = []
    beepath = '/home/kulyukin-lab1/Nikhil Project/BUZZ2/'
    cricketpath = '/home/kulyukin-lab1/Nikhil Project/BUZZ2/'
    noisepath = '/home/kulyukin-lab1/Nikhil Project/BUZZ2/'
    bees = glob.glob(beepath + dataType + '/bee_' + dataType +'/*.wav')
    crickets = glob.glob(cricketpath + dataType + '/cricket_' + dataType +'/*.wav')
    noises = glob.glob(noisepath + dataType + '/cricket_' + dataType +'/*.wav')
    filePaths = []
    for file in bees:
        filePaths.append((file,(1,0,0)))
    for file in crickets:
        filePaths.append((file,(0,1,0)))
    for file in noises:
        filePaths.append((file,(0,0,1)))
    filePaths = shuffle(filePaths)
    for i in range(len(filePaths)):
        samplerate, audio = wavfile.read(filePaths[i][0])
        audio = audio/float(np.max(audio))

        # method 1
        mid = len(audio)//2
        piece1 = audio[mid-22000:mid+1]
        piece2 = audio[mid+1:mid+22000]
        audio = np.concatenate((piece1,piece2))

        # method 2
        # if (len(audio) < 88000):
        #     val = 88000 - len(audio)
        #     audio = np.pad(audio,(0,val),'constant',constant_values = (0,0))
        # elif (len(audio) > 88000):
        #     audio = audio[0:88000]

        dataX.append(audio)
        dataY.append(filePaths[i][1])
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    return dataX,dataY

#####################################################################################################################################
#####################################################################################################################################
# BEE1 Data

convBee1_train_d = load('convBee1_train_d.pck')
convBee1_test_d = load('convBee1_test_d.pck')
convBee1_valid_d = load('convBee1_valid_d.pck')

convBee1_train_dX = np.array(convBee1_train_d[0])
convBee1_train_dY = convBee1_train_d[1]
convBee1_test_dX = np.array(convBee1_test_d[0])
convBee1_test_dY = convBee1_test_d[1]
convBee1_valid_dX = np.array(convBee1_valid_d[0])
convBee1_valid_dY = convBee1_valid_d[1]

convBee1_train_dX = convBee1_train_dX.reshape([-1,32,32,3])
convBee1_test_dX = convBee1_test_dX.reshape([-1,32,32,3])

def train_convBee1_cnn():
    input_layer = input_data(shape=[None,32,32,3])
    conv_layer_1  = conv_2d(input_layer,
                            nb_filter=20,
                            filter_size=5,
                            activation='relu',
                            name='conv_layer_1')
    pool_layer_1  = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                       nb_filter=40,
                       filter_size=3,
                       activation='relu',
                       name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    fc_layer_1 = fully_connected(pool_layer_2, 100,
                                  activation='relu',
                                  name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    network = regression(fc_layer_2, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.01)
    model = tflearn.DNN(network)
    return model

def load_train_convBee1_cnn(beepath):
    input_layer = input_data(shape=[None,32,32,3])
    conv_layer_1  = conv_2d(input_layer,
                            nb_filter=20,
                            filter_size=5,
                            activation='relu',
                            name='conv_layer_1')
    pool_layer_1  = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                       nb_filter=40,
                       filter_size=3,
                       activation='relu',
                       name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    fc_layer_1 = fully_connected(pool_layer_2, 100,
                                  activation='relu',
                                  name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(beepath)
    return model

def test_tflearn_convBee1_cnn_model(ann_model, validX, validY):
    results = []
    for i in range(len(validX)):
        prediction = ann_model.predict(validX[i].reshape([-1, 32, 32, 3]))
        results.append(np.argmax(prediction, axis=1)[0] == \
                       np.argmax(validY[i]))
    return float(sum((np.array(results) == True)))/float(len(results))

NUM_EPOCHS = 50
BATCH_SIZE = 10
MODEL = train_convBee1_cnn()
MODEL.fit(convBee1_train_dX, convBee1_train_dY, n_epoch=NUM_EPOCHS,
          shuffle=True,
          validation_set=(convBee1_test_dX, convBee1_test_dY),
          show_metric=True,
          batch_size=BATCH_SIZE,
          run_id='BEE1_CNN_1')
SAVE_CNN_PATH = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/workspace/part2/BEE1_CNN.tfl'
MODEL.save(SAVE_CNN_PATH)
# For checking whether the network has been trained correctly
print(MODEL.predict(convBee1_test_dX[0].reshape([-1, 32, 32, 3])))

# Classifying the images on validation data and deriving the validation accuracy

convBee1_cnn_path = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/workspace/part2/BEE1_CNN.tfl'
convBee1_cnn = load_train_convBee1_cnn(convBee1_cnn_path)

if __name__ == '__main__':
    print('BEE1 CNN accuracy = {}'.format(test_tflearn_convBee1_cnn_model(convBee1_cnn, convBee1_valid_dX, convBee1_valid_dY)))

#####################################################################################################################################
#####################################################################################################################################
# BEE2_1S Data

# beesPreProcessor is a function which preprocesses the data as required and stored in these respective arrays for the training of 
# neural networks
convBee2_1S_train_dX, convBee2_1S_train_dY = beesPreProcessor('training')
convBee2_1S_test_dX, convBee2_1S_test_dY = beesPreProcessor('testing')
convBee2_1S_valid_dX, convBee2_1S_valid_dY = beesPreProcessor('validation')

convBee2_1S_train_dX = convBee2_1S_train_dX.reshape([-1,90,90,3])
convBee3_test_dX = convBee2_1S_test_dX.reshape([-1,90,90,3])

def train_convBee2_1S_cnn():
    input_layer = input_data(shape=[None,90,90,3])
    conv_layer_1  = conv_2d(input_layer,
                            nb_filter=20,
                            filter_size=5,
                            activation='relu',
                            name='conv_layer_1')
    pool_layer_1  = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                       nb_filter=40,
                       filter_size=3,
                       activation='relu',
                       name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    conv_layer_3 = conv_2d(pool_layer_2,
                       nb_filter=50,
                       filter_size=3,
                       activation='relu',
                       name='conv_layer_3')
    pool_layer_3 = max_pool_2d(conv_layer_3, 2, name='pool_layer_3')
    fc_layer_1  = fully_connected(pool_layer_3, 100,
                                  activation='relu',
                                  name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    network = regression(fc_layer_2, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.01)
    model = tflearn.DNN(network)
    return model

def load_train_convBee2_1S_cnn(beepath):
    input_layer = input_data(shape=[None, 90, 90, 3])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=20,
                           filter_size=5,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                           nb_filter=40,
                           filter_size=3,
                           activation='relu',
                           name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    conv_layer_3 = conv_2d(pool_layer_2,
                           nb_filter=50,
                           filter_size=3,
                           activation='relu',
                           name='conv_layer_3')
    pool_layer_3 = max_pool_2d(conv_layer_3, 2, name='pool_layer_3')
    fc_layer_1 = fully_connected(pool_layer_3, 100,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(beepath)
    return model

def test_tflearn_convBee2_1S_cnn_model(ann_model, validX, validY):
    results = []
    for i in range(len(validX)):
        prediction = ann_model.predict(validX[i].reshape([-1, 90, 90, 3]))
        results.append(np.argmax(prediction, axis=1)[0] == \
                       np.argmax(validY[i]))
    return float(sum((np.array(results) == True)))/float(len(results))

NUM_EPOCHS = 30
BATCH_SIZE = 10
MODEL = train_convBee2_1S_cnn()
MODEL.fit(convBee2_1S_train_dX, convBee2_1S_train_dY, n_epoch=NUM_EPOCHS,
          shuffle=True,
          validation_set=(convBee2_1S_test_dX, convBee2_1S_test_dY),
          show_metric=True,
          batch_size=BATCH_SIZE,
          run_id='BEE2_1S_CNN_1')
SAVE_CNN_PATH = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/workspace/part2/BEE2_1S_CNN.tfl'
MODEL.save(SAVE_CNN_PATH)
# For checking whether the network has been trained correctly
print(MODEL.predict(convBee2_1S_test_dX[0].reshape([-1, 90, 90, 3])))

# Classifying the images on validation data and deriving the validation accuracy

convBee2_1S_cnn_path = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/workspace/part2/BEE2_1S_CNN.tfl'
convBee2_1S_cnn = load_train_convBee2_1S_cnn(convBee2_1S_cnn_path)

if __name__ == '__main__':
    print('BEE2_1S CNN accuracy = {}'.format(test_tflearn_convBee2_1S_cnn_model(convBee2_1S_cnn, convBee2_1S_valid_dX, convBee2_1S_valid_dY)))



#####################################################################################################################################
#####################################################################################################################################
# BEE2_2S Data

# beesPreProcessor is a function which preprocesses the data as required and stored in these respective arrays for the training of 
# neural networks

convBee2_2S_train_dX, convBee2_2S_train_dY = beesPreProcessor('training')
convBee2_2S_test_dX, convBee2_2S_test_dY = beesPreProcessor('testing')
convBee2_2S_valid_dX, convBee2_2S_valid_dY = beesPreProcessor('validation')

convBee2_2S_train_dX = convBee2_2S_train_dX.reshape([-1,90,90,3])
convBee2_2S_test_dX = convBee2_2S_test_dX.reshape([-1,90,90,3])


def train_convBee2_2S_cnn():
    input_layer = input_data(shape=[None,90,90,3])
    conv_layer_1  = conv_2d(input_layer,
                            nb_filter=20,
                            filter_size=5,
                            activation='relu',
                            name='conv_layer_1')
    pool_layer_1  = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                       nb_filter=40,
                       filter_size=3,
                       activation='relu',
                       name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    conv_layer_3 = conv_2d(pool_layer_2,
                       nb_filter=50,
                       filter_size=3,
                       activation='relu',
                       name='conv_layer_3')
    pool_layer_3 = max_pool_2d(conv_layer_3, 2, name='pool_layer_3')
    fc_layer_1  = fully_connected(pool_layer_3, 100,
                                  activation='relu',
                                  name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    network = regression(fc_layer_2, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    model = tflearn.DNN(network)
    return model

def load_train_convBee2_2S_cnn(beepath):
    input_layer = input_data(shape=[None, 90, 90, 3])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=20,
                           filter_size=5,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                           nb_filter=40,
                           filter_size=3,
                           activation='relu',
                           name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    conv_layer_3 = conv_2d(pool_layer_2,
                           nb_filter=50,
                           filter_size=3,
                           activation='relu',
                           name='conv_layer_3')
    pool_layer_3 = max_pool_2d(conv_layer_3, 2, name='pool_layer_3')
    fc_layer_1 = fully_connected(pool_layer_3, 100,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(beepath)
    return model

def test_tflearn_convBee2_2S_cnn_model(ann_model, validX, validY):
    results = []
    for i in range(len(validX)):
        prediction = ann_model.predict(validX[i].reshape([-1, 90, 90, 3]))
        results.append(np.argmax(prediction, axis=1)[0] == \
                       np.argmax(validY[i]))
    return float(sum((np.array(results) == True)))/float(len(results))

NUM_EPOCHS = 30
BATCH_SIZE = 10
MODEL = train_convBee2_2S_cnn()
MODEL.fit(convBee2_2S_train_dX, convBee2_2S_train_dY, n_epoch=NUM_EPOCHS,
          shuffle=True,
          validation_set=(convBee2_2S_test_dX, convBee2_2S_test_dY),
          show_metric=True,
          batch_size=BATCH_SIZE,
          run_id='BEE2_2S_CNN_1')
SAVE_CNN_PATH = '/home/kulyukin-lab1/Nikhil Project/project1 workspace/part2/BEE2_2S_CNN.tfl'
MODEL.save(SAVE_CNN_PATH)
# For checking whether the network has been trained correctly
print(MODEL.predict(convBee2_2S_test_dX[0].reshape([-1, 90, 90, 3])))

# Classifying the images on validation data and deriving the validation accuracy

convBee2_2S_cnn_path = '/home/kulyukin-lab1/Nikhil Project/project1 workspace/part2/BEE2_2S_CNN.tfl'
convBee2_2S_cnn = load_train_convBee2_2S_cnn(convBee2_2S_cnn_path)

if __name__ == '__main__':
    print('BEE2_2S CNN accuracy = {}'.format(test_tflearn_convBee2_2S_cnn_model(convBee2_2S_cnn, convBee2_2S_valid_dX, convBee2_2S_valid_dY)))



########################################################################################################################################################################
########################################################################################################################################################################

# BUZZ1 Dataset

convBuzz1_train_test = buzz1TestTrainPreProcessor()
convBuzz1_train_d = convBuzz1_train_test[0][:7001],convBuzz1_train_test[1][:7001]
convBuzz1_test_d = convBuzz1_train_test[0][7001:],convBuzz1_train_test[1][7001:]
convBuzz1_train_dX = np.array(convBuzz1_train_d[0])
convBuzz1_train_dY = convBuzz1_train_d[1]
convBuzz1_test_dX = np.array(convBuzz1_test_d[0])
convBuzz1_test_dY = convBuzz1_test_d[1]

convBuzz1_train_dX = convBuzz1_train_dX.reshape([-1,440,100,1])
convBuzz1_test_dX = convBuzz1_test_dX.reshape([-1,440,100,1])

convBuzz1_valid_d = buzz1ValidPreProcessor()
convBuzz1_valid_dX = np.array(convBuzz1_valid_d[0])
convBuzz1_valid_dY = convBuzz1_valid_d[1]


def train_convBuzz1_cnn():
    input_layer = input_data(shape=[None,440,100,1])
    conv_layer_1  = conv_2d(input_layer,
                            nb_filter=20,
                            filter_size=5,
                            activation='relu',
                            name='conv_layer_1')
    pool_layer_1  = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                       nb_filter=40,
                       filter_size=3,
                       activation='relu',
                       name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    conv_layer_3 = conv_2d(pool_layer_2,
                       nb_filter=50,
                       filter_size=3,
                       activation='relu',
                       name='conv_layer_3')
    pool_layer_3 = max_pool_2d(conv_layer_3, 2, name='pool_layer_3')
    conv_layer_4 = conv_2d(pool_layer_3,
                       nb_filter=50,
                       filter_size=3,
                       activation='relu',
                       name='conv_layer_4')
    pool_layer_4 = max_pool_2d(conv_layer_4, 2, name='pool_layer_4')
    fc_layer_1 = fully_connected(pool_layer_4, 100,
                                  activation='relu',
                                  name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 60,
                                  activation='relu',
                                  name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 3,
                                 activation='softmax',
                                 name='fc_layer_3')
    network = regression(fc_layer_3, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.01)
    model = tflearn.DNN(network)
    return model

def load_train_convBuzz1_cnn(beepath):
    input_layer = input_data(shape=[None,440,100,1])
    conv_layer_1  = conv_2d(input_layer,
                            nb_filter=20,
                            filter_size=5,
                            activation='relu',
                            name='conv_layer_1')
    pool_layer_1  = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                       nb_filter=40,
                       filter_size=3,
                       activation='relu',
                       name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    conv_layer_3 = conv_2d(pool_layer_2,
                       nb_filter=50,
                       filter_size=3,
                       activation='relu',
                       name='conv_layer_3')
    pool_layer_3 = max_pool_2d(conv_layer_3, 2, name='pool_layer_3')
    conv_layer_4 = conv_2d(pool_layer_3,
                       nb_filter=50,
                       filter_size=3,
                       activation='relu',
                       name='conv_layer_4')
    pool_layer_4 = max_pool_2d(conv_layer_4, 2, name='pool_layer_4')
    fc_layer_1 = fully_connected(pool_layer_4, 100,
                                  activation='relu',
                                  name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 60,
                                  activation='relu',
                                  name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 3,
                                 activation='softmax',
                                 name='fc_layer_3')
    model = tflearn.DNN(fc_layer_3)
    model.load(beepath)
    return model

def test_tflearn_convBuzz1_cnn_model(ann_model, validX, validY):
    results = []
    for i in range(len(validX)):
        prediction = ann_model.predict(validX[i].reshape([-1, 440, 100, 1]))
        results.append(np.argmax(prediction, axis=1)[0] == \
                       np.argmax(validY[i]))
    return float(sum((np.array(results) == True)))/float(len(results))

NUM_EPOCHS = 50
BATCH_SIZE = 10
MODEL = train_convBuzz1_cnn()
MODEL.fit(convBuzz1_train_dX, convBuzz1_train_dY, n_epoch=NUM_EPOCHS,
          shuffle=True,
          validation_set=(convBuzz1_test_dX, convBuzz1_test_dY),
          show_metric=True,
          batch_size=BATCH_SIZE,
          run_id='Buzz1_CNN_1')
SAVE_CNN_PATH = '/home/kulyukin-lab1/Nikhil Project/workspace/part2/Buzz1_CNN.tfl'
MODEL.save(SAVE_CNN_PATH)
# For checking whether the network has been trained correctly
print(MODEL.predict(convBuzz1_test_dX[0].reshape([-1, 440, 100, 1])))

# Classifying the images on validation data and deriving the validation accuracy

convBuzz1_cnn_path = '/home/kulyukin-lab1/Nikhil Project/workspace/part2/Buzz1_CNN.tfl'
convBuzz1_cnn = load_train_convBuzz1_cnn(convBuzz1_cnn_path)

if __name__ == '__main__':
    print('Buzz1 CNN accuracy = {}'.format(test_tflearn_convBuzz1_cnn_model(convBuzz1_cnn, convBuzz1_valid_dX, convBuzz1_valid_dY)))



########################################################################################################################################################################
########################################################################################################################################################################

# BUZZ2 Dataset

convBuzz2_train_d = buzz2PreProcessor('train')
convBuzz2_test_d = buzz2PreProcessor('test')
convBuzz2_train_dX = np.array(convBuzz2_train_d[0])
convBuzz2_train_dY = convBuzz2_train_d[1]
convBuzz2_test_dX = np.array(convBuzz2_test_d[0])
convBuzz2_test_dY = convBuzz2_test_d[1]

convBuzz2_train_dX = convBuzz2_train_dX.reshape([-1,440,100,1])
convBuzz2_test_dX = convBuzz2_test_dX.reshape([-1,440,100,1])

convBuzz2_valid_d = buzz2PreProcessor('valid')
convBuzz2_valid_dX = np.array(convBuzz2_valid_d[0])
convBuzz2_valid_dY = convBuzz2_valid_d[1]


def train_convBuzz2_cnn():
    input_layer = input_data(shape=[None,440,100,1])
    conv_layer_1  = conv_2d(input_layer,
                            nb_filter=20,
                            filter_size=5,
                            activation='relu',
                            name='conv_layer_1')
    pool_layer_1  = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                       nb_filter=40,
                       filter_size=3,
                       activation='relu',
                       name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    conv_layer_3 = conv_2d(pool_layer_2,
                       nb_filter=50,
                       filter_size=3,
                       activation='relu',
                       name='conv_layer_3')
    pool_layer_3 = max_pool_2d(conv_layer_3, 2, name='pool_layer_3')
    conv_layer_4 = conv_2d(pool_layer_3,
                       nb_filter=50,
                       filter_size=3,
                       activation='relu',
                       name='conv_layer_4')
    pool_layer_4 = max_pool_2d(conv_layer_4, 2, name='pool_layer_4')
    fc_layer_1 = fully_connected(pool_layer_4, 90,
                                  activation='relu',
                                  name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 50,
                                  activation='relu',
                                  name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 3,
                                 activation='softmax',
                                 name='fc_layer_3')
    network = regression(fc_layer_3, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.01)
    model = tflearn.DNN(network)
    return model

def load_train_convBuzz2_cnn(beepath):
    input_layer = input_data(shape=[None,440,100,1])
    conv_layer_1  = conv_2d(input_layer,
                            nb_filter=20,
                            filter_size=5,
                            activation='relu',
                            name='conv_layer_1')
    pool_layer_1  = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                       nb_filter=40,
                       filter_size=3,
                       activation='relu',
                       name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    conv_layer_3 = conv_2d(pool_layer_2,
                       nb_filter=50,
                       filter_size=3,
                       activation='relu',
                       name='conv_layer_3')
    pool_layer_3 = max_pool_2d(conv_layer_3, 2, name='pool_layer_3')
    conv_layer_4 = conv_2d(pool_layer_3,
                       nb_filter=50,
                       filter_size=3,
                       activation='relu',
                       name='conv_layer_4')
    pool_layer_4 = max_pool_2d(conv_layer_4, 2, name='pool_layer_4')
    fc_layer_1 = fully_connected(pool_layer_4, 90,
                                  activation='relu',
                                  name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 50,
                                  activation='relu',
                                  name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 3,
                                 activation='softmax',
                                 name='fc_layer_3')
    model = tflearn.DNN(fc_layer_3)
    model.load(beepath)
    return model

def test_tflearn_convBuzz2_cnn_model(ann_model, validX, validY):
    results = []
    for i in range(len(validX)):
        prediction = ann_model.predict(validX[i].reshape([-1, 440, 100, 1]))
        results.append(np.argmax(prediction, axis=1)[0] == \
                       np.argmax(validY[i]))
    return float(sum((np.array(results) == True)))/float(len(results))

NUM_EPOCHS = 50
BATCH_SIZE = 10
MODEL = train_convBuzz2_cnn()
MODEL.fit(convBuzz2_train_dX, convBuzz2_train_dY, n_epoch=NUM_EPOCHS,
          shuffle=True,
          validation_set=(convBuzz2_test_dX, convBuzz2_test_dY),
          show_metric=True,
          batch_size=BATCH_SIZE,
          run_id='Buzz2_CNN_1')
SAVE_CNN_PATH = '/home/kulyukin-lab1/Nikhil Project/workspace/part2/Buzz2_CNN.tfl'
MODEL.save(SAVE_CNN_PATH)
# For checking whether the network has been trained correctly
print(MODEL.predict(convBuzz2_test_dX[0].reshape([-1, 440, 100, 1])))

# Classifying the images on validation data and deriving the validation accuracy

convBuzz2_cnn_path = '/home/kulyukin-lab1/Nikhil Project/workspace/part2/Buzz2_CNN.tfl'
convBuzz2_cnn = load_train_convBuzz2_cnn(convBuzz2_cnn_path)

if __name__ == '__main__':
    print('Buzz2 CNN accuracy = {}'.format(test_tflearn_convBuzz2_cnn_model(convBuzz2_cnn, convBuzz2_valid_dX, convBuzz2_valid_dY)))