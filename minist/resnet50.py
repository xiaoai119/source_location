# Keras imports
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)

from keras.applications.resnet50 import ResNet50

# Paper: https://arxiv.org/abs/1512.03385
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.models import load_model
from keras.utils import np_utils
import numpy as np
import os
from sklearn.model_selection import train_test_split

def build_resnet50(img_shape=(3, 224, 224), n_classes=1000, l2_reg=0.,
                load_pretrained=False, freeze_layers_from='base_model'):
    # Decide if load pretrained weights from imagenet
    if load_pretrained:
        weights = 'imagenet'
    else:
        weights = None

    # Get base model
    base_model = ResNet50(include_top=False, weights=weights,
                       input_tensor=None, input_shape=img_shape)

    # Add final layers
    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(n_classes, activation='softmax', name='fc1000')(x)

    # This is the model we will train
    model = Model(input=base_model.input, output=predictions)

    # Freeze some layers
    if freeze_layers_from is not None:
        if freeze_layers_from == 'base_model':
            print ('   Freezing base model layers')
            for layer in base_model.layers:
                layer.trainable = False
        else:
            for i, layer in enumerate(model.layers):
                print(i, layer.name)
            print ('   Freezing from layer 0 to ' + str(freeze_layers_from))
            for layer in model.layers[:freeze_layers_from]:
               layer.trainable = False
            for layer in model.layers[freeze_layers_from:]:
               layer.trainable = True

    return model

def load_data(resultpath):
    datapath = os.path.join(resultpath, "data10_4.npz")
    if os.path.exists(datapath):
        data = np.load(datapath)
        X, Y = data["X"], data["Y"]
    else:
        X = np.array(np.arange(432000)).reshape(10, 120, 120, 3)
        Y = [0, 0, 1, 1, 2, 2, 3, 3, 2, 0]
        X = X.astype('float32')
        Y = np_utils.to_categorical(Y, 4)
        np.savez(datapath, X=X, Y=Y)
        print('Saved dataset to dataset.npz.')
    print('X_shape:{}\nY_shape:{}'.format(X.shape, Y.shape))
    return X, Y

def train_model(resultpath):
    model = build_resnet50()

    # if want to use SGD, first define sgd, then set optimizer=sgd
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0, nesterov=True)

    # select loss\optimizer\
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(), metrics=['accuracy'])
    model.summary()

    # draw the model structure
    plot_model(model, show_shapes=True,
               to_file=os.path.join(resultpath, 'model.png'))

    # load data
    X, Y = load_data(resultpath)

    # split train and test data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=2)

    # input data to model and train
    history = model.fit(X_train, Y_train, batch_size=2, epochs=10,
                        validation_data=(X_test, Y_test), verbose=1, shuffle=True)

    # evaluate the model
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', acc)


def main():
    train_model()

if __name__ == "__main__":
    main()

