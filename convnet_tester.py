import cv2
from scipy.io import wavfile
import tflearn
import numpy as np
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d

def load_image_convnet_bee1(path):
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
    model.load(path)
    return model

def fit_image_convnet_bee1(convnet,image_path):
    image = cv2.imread(image_path)
    scaled_gray_image = image/255.0
    scaled_gray_image = np.array(scaled_gray_image)
    prediction = convnet.predict(scaled_gray_image.reshape([-1, 32, 32, 3]))
    output = [0,0]
    index = np.argmax(prediction, axis=1)[0]
    output[index] = 1
    return output

def load_image_convnet_bee2_1S(path):
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
    model.load(path)
    return model

def fit_image_convnet_bee2_1S(convnet,image_path):
    image = cv2.imread(image_path)
    scaled_gray_image = image/255.0
    scaled_gray_image = cv2.resize(scaled_gray_image,(90,90))
    scaled_gray_image = np.array(scaled_gray_image)
    prediction = convnet.predict(scaled_gray_image.reshape([-1, 90, 90, 3]))
    output = [0,0]
    index = np.argmax(prediction, axis=1)[0]
    output[index] = 1
    return output

def load_image_convnet_bee2_2S(path):
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
    model.load(path)
    return model

def fit_image_convnet_bee2_2S(convnet,image_path):
    image = cv2.imread(image_path)
    scaled_gray_image = image/255.0
    scaled_gray_image = cv2.resize(scaled_gray_image,(90,90))
    scaled_gray_image = np.array(scaled_gray_image)
    prediction = convnet.predict(scaled_gray_image.reshape([-1, 90, 90, 3]))
    output = [0,0]
    index = np.argmax(prediction, axis=1)[0]
    output[index] = 1
    return output

def load_audio_convnet_buzz1(path):
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
    model.load(path)
    return model

def fit_audio_convnet_buzz1(convnet, audio_path):
    samplerate, audio = wavfile.read(audio_path)
    audio = audio/float(np.max(audio))

    mid = len(audio)//2
    piece1 = audio[mid-22000:mid+1]
    piece2 = audio[mid+1:mid+22000]
    audio = np.concatenate((piece1,piece2))

    prediction = convnet.predict(audio.reshape([-1, 440, 100, 1]))
    output = [0,0,0]
    index = np.argmax(prediction, axis=1)[0]
    output[index] = 1
    return output

def load_audio_convnet_buzz2(path):
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
    model.load(path)
    return model

def fit_audio_convnet_buzz2(convnet, audio_path):
    samplerate, audio = wavfile.read(audio_path)
    audio = audio/float(np.max(audio))

    mid = len(audio)//2
    piece1 = audio[mid-22000:mid+1]
    piece2 = audio[mid+1:mid+22000]
    audio = np.concatenate((piece1,piece2))

    prediction = convnet.predict(audio.reshape([-1, 440, 100, 1]))
    output = [0,0,0]
    index = np.argmax(prediction, axis=1)[0]
    output[index] = 1
    return output