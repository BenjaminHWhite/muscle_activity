import os
import numpy as np
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf
from keras.utils import multi_gpu_model, to_categorical

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
import keras.backend as K
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Dropout, Conv2D, GlobalAveragePooling2D, Input, \
    BatchNormalization, Activation,AveragePooling2D, MaxPooling2D, concatenate
import pandas as pd
from skimage import io
import h5py
from tqdm import tqdm
import itertools
from pprint import pprint
from keras.optimizers import Adam
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import time
import datetime
import tempfile
from sklearn.mixture import GaussianMixture

K.set_image_data_format = 'channels_first'


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1)):
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=bn_axis, scale=False)(x)
    x = Activation('relu')(x)
    return x


def InceptionV3(input_shape=None, basefilter=8):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    img_input = Input(shape=input_shape)

    x = conv2d_bn(img_input, basefilter * 2, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, basefilter * 2, 3, 3, padding='valid')
    x = conv2d_bn(x, basefilter * 4, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, basefilter * 5, 1, 1, padding='valid')
    x = conv2d_bn(x, basefilter * 12, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, basefilter * 4, 1, 1)

    branch5x5 = conv2d_bn(x, basefilter * 4, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, basefilter * 3, 5, 5)

    branch3x3dbl = conv2d_bn(x, basefilter * 4, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, basefilter * 6, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, basefilter * 6, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, basefilter * 2, 1, 1)
    x = concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, basefilter * 4, 1, 1)

    branch5x5 = conv2d_bn(x, basefilter * 3, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, basefilter * 4, 5, 5)

    branch3x3dbl = conv2d_bn(x, basefilter * 4, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, basefilter * 6, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, basefilter * 6, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, basefilter * 4, 1, 1)
    x = concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, basefilter * 4, 1, 1)

    branch5x5 = conv2d_bn(x, basefilter * 3, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, basefilter * 4, 5, 5)

    branch3x3dbl = conv2d_bn(x, basefilter * 4, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, basefilter * 6, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, basefilter * 6, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, basefilter * 4, 1, 1)
    x = concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, basefilter * 24, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, basefilter * 4, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, basefilter * 6, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, basefilter * 6, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, basefilter * 12, 1, 1)

    branch7x7 = conv2d_bn(x, basefilter * 8, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, basefilter * 8, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, basefilter * 12, 7, 1)

    branch7x7dbl = conv2d_bn(x, basefilter * 8, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, basefilter * 8, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, basefilter * 8, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, basefilter * 8, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, basefilter * 12, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, basefilter * 12, 1, 1)
    x = concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, basefilter * 12, 1, 1)

        branch7x7 = conv2d_bn(x, basefilter * 10, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, basefilter * 10, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, basefilter * 12, 7, 1)

        branch7x7dbl = conv2d_bn(x, basefilter * 10, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, basefilter * 10, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, basefilter * 10, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, basefilter * 10, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, basefilter * 12, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, basefilter * 12, 1, 1)
        x = concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, basefilter * 12, 1, 1)

    branch7x7 = conv2d_bn(x, basefilter * 12, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, basefilter * 12, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, basefilter * 12, 7, 1)

    branch7x7dbl = conv2d_bn(x, basefilter * 12, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, basefilter * 12, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, basefilter * 12, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, basefilter * 12, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, basefilter * 12, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, basefilter * 12, 1, 1)
    x = concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, basefilter * 12, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, basefilter * 20, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, basefilter * 12, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, basefilter * 12, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, basefilter * 12, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, basefilter * 12, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=channel_axis,
        name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, basefilter * 20, 1, 1)

        branch3x3 = conv2d_bn(x, basefilter * 24, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, basefilter * 24, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, basefilter * 24, 3, 1)
        branch3x3 = concatenate(
            [branch3x3_1, branch3x3_2],
            axis=channel_axis,
            name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, basefilter * 28, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, basefilter * 24, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, basefilter * 24, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, basefilter * 24, 3, 1)
        branch3x3dbl = concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, basefilter * 12, 1, 1)
        x = concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))

    model = Model(img_input, x, name='inception_v3')

    return model


def shuffle(X, Y):  # assmume matrix X is NxHxWxC, 1st dimension needs to be shuffled, Y is Nx1
    dim = X.shape

    Aindx = np.random.choice(dim[0], size=dim[0] // 2, replace=False)
    Bindx = list(set(range(0, dim[0])) - set(Aindx))
    # print(set(Aindx).intersection(Bindx))
    for l in tqdm(range(0, dim[0] // 2 - 1)):
        i = Aindx[l]
        j = Bindx[l]
        temp = X[i, :, :, :]
        X[i, :, :, :] = X[j, :, :, :]
        X[j, :, :, :] = temp

        temp = Y[i]
        Y[i] = Y[j]
        Y[j] = temp

    return X, Y


class ModelMGPU(Model):

    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''
        Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)


def main(opt):
    time_id = time.strftime('%d-%m-%Y_%H-%M-%S')
    print('Unique ID is %s ' % time_id)
    L = int(0)
    numgpu = int(opt['numgpu'])
    C = len(opt['labels'])
    for i in range(len(csvs)):
        x = pd.read_csv(csvs[i], header=[0])
        x = x.values
        L = L + len(x)
    print('Approximately %d frames are collected from %d images.' % (L, len(csvs)))

    dframes = opt['nframes'] // 2
    exten = os.path.splitext(atlases[0])[1]
    if exten.lower() == '.tif':
        x = io.imread(atlases[0])

    elif exten.lower() == '.h5':
        f = h5py.File(atlases[0], 'r')
        dataset = list(f.keys())
        if len(dataset) > 1:
            print('WARNING: More than one key found in the h5 file {}'.format(atlases[0]))
            print('WARNING: Reading only the first key {} '.format(dataset[0]))
        x = f[dataset[0]]

    else:
        sys.exit('ERROR: Images must be either .tif or .h5 files. You entered {}'.format(atlases[i]))

    dim = x.shape
    W = dim[1]
    H = dim[2]
    X = np.zeros((L, W, H, opt['nframes']), dtype=np.float32)
    Y = np.zeros((L, 1), dtype=np.int32)

    print('WARNING: Approximate memory required = %d GB' % (np.ceil(sys.getsizeof(X) / (1024.0 ** 3))))

    count = 0
    for i in range(0, len(atlases)):
        print('Reading {}'.format(atlases[i]))
        exten = os.path.splitext(atlases[i])[1]
        if exten.lower() == '.tif':
            x = io.imread(atlases[i])

        elif exten.lower() == '.h5':
            f = h5py.File(atlases[i], 'r')
            dataset = list(f.keys())
            if len(dataset) > 1:
                print('WARNING: More than one key found in the h5 file {}'.format(atlases[i]))
                print('WARNING: Reading only the first key {} '.format(dataset[0]))
            x = f[dataset[0]]

        else:
            sys.exit('ERROR: Images must be either .tif or .h5 files. You entered {}'.format(atlases[i]))

        x = np.asarray(x, dtype=np.float32)
        print('Image size = {}'.format(x.shape))

        print('Finding a scaling factor..')
        S = 1000000
        indx = np.random.choice(np.prod(x.shape), size=(S, 1), replace=False)
        indx = np.unravel_index(indx, x.shape, order='C')
        indx = np.asarray(indx, dtype=int)
        print(indx.shape)
        y = np.zeros((S, 1), dtype=np.float32)
        for j in range(0, S):
            y[j] = x[indx[0, j, 0], indx[1, j, 0], indx[2, j, 0]]

        y = y[y > 0]
        y = np.asarray(y, dtype=np.float32)

        gmm = GaussianMixture(n_components=2, covariance_type='diag', tol=0.001,
                              reg_covar=1e-06, max_iter=100, n_init=1, precisions_init=None,
                              weights_init=(0.9, 0.1), init_params='kmeans',
                              means_init=np.reshape((40, 400), (2, 1)),
                              warm_start=True, verbose=1, verbose_interval=1)
        gmm.fit(y.reshape(-1, 1))

        y = gmm.means_[1]

        print('Scaling image by %.2f' % (y))
        x = x / y

        

        csv = pd.read_csv(csvs[i], header=[0])
        csv = csv.values
        print('CSV size = {}'.format(csv.shape))
        print('Collecting data ..')
        for j in tqdm(range(0, csv.shape[0])):
            J = csv[j, 1]
            label = str(csv[j, 4])
            try:
                indx = list(opt['labels']).index(label)
            except:
                indx = -1
            if (J > dframes) & (J < csv.shape[0] - dframes - 1) & (indx >= 0):
                X[count, :, :, :] = np.transpose(x[J - dframes:J + dframes + 1, :, :], [1, 2, 0])
                Y[count] = indx
                count = count + 1

    X = X[0:count, :, :, :]
    Y = Y[0:count]

    uY, cY = np.unique(Y, return_counts=True)
    # frequencies = np.asarray((uY, cY)).T
    for i in range(len(opt['labels'])):
        print('Label = %10s | Count = %4d' % (opt['labels'][i], cY[i]))

    #basemodel = ResNet50(input_shape=(None, None, opt['nframes']), classes=C, include_top=False, weights=None)
    basemodel = InceptionV3(basefilter=opt['basefilter'], input_shape=(None, None, opt['nframes']))
    x = basemodel.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    out = Dense(C, activation='sigmoid')(x)
    model = Model(inputs=basemodel.input, outputs=out)
    print('Total number of parameters = {}'.format(model.count_params()))

    if opt['initmodel'] != None:
        if os.path.isfile(opt['initmodel']):
            print('Loading a pre-trained model : {}'.format(opt['initmodel']))
            try:
                oldmodel = load_model(opt['initmodel'])
                model.set_weights(oldmodel.get_weights())

            except Exception as e:
                print(str(e))
                try:
                    print('Loading only weights from the pre-trained model : {}'.format(opt['initmodel']))
                    model.load_weights(opt['initmodel'], by_name=True)
                except Exception as e:
                    print(str(e))
                    print('ERROR: Pre-trained model can not be loaded. Initilizing from a random one.')

    if numgpu > 1:
        model = ModelMGPU(model, gpus=numgpu)

    model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    Y = to_categorical(Y, num_classes=C)

    print('Training data size {} and {}'.format(X.shape, Y.shape))

    uid = str(os.path.basename(tempfile.mktemp())[3:].upper())
    tempoutname = 'Model_' + uid + '_' + time_id + '_nFrames_' + str(opt['nframes']).zfill(
        2) + '_epoch-{epoch:03d}_val_acc-{val_accuracy:.4f}.h5'
    tempoutname = os.path.join(opt['outdir'], tempoutname)

    outname = 'Model_' + uid + '_' + time_id + '_nFrames_' + str(opt['nframes']).zfill(2) + '.h5'
    outname = os.path.join(opt['outdir'], outname)
    print('Trained models will be written at {}'.format(outname))

    callbacks = [ModelCheckpoint(tempoutname, monitor='val_accuracy', verbose=1, save_best_only=False,
                                 period=1, mode='max')]
    dlr = ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=5,
                            mode='max', verbose=1, cooldown=2, min_lr=1e-8)
    earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=10,
                              verbose=1, mode='max', restore_best_weights=True)
    callbacks.append(dlr)
    callbacks.append(earlystop)
    batchsize = 40 * numgpu

    # Some Fisher-Yates shuffle to shuffle training/validation data
    print('Shuffling data...')
    X, Y = shuffle(X, Y)
    t1 = time.process_time()
    model.fit(X, Y, epochs=50, batch_size=batchsize, verbose=1, validation_split=0.2,
              callbacks=callbacks, shuffle=True)
    
    t2 = time.process_time()
    # print('Final model is saved at {}'.format(outname))
    # model.save(filepath=outname)
    print('Elapsed time = {}'.format(datetime.timedelta(seconds=(t2 - t1))))


if __name__ == '__main__':
    arg = argparse.ArgumentParser(description='Training for movement detection')
    parser = arg.add_argument_group('Required arguments')
    parser.add_argument('--atlases', required=True, nargs='+', type=str, dest='ATLAS',
                        help='Atlas images to train. All atlases must have same width and height, '
                             'but might have differet number of frames, as noted in the corresponding csv file. '
                             'Preferably use HDF5 .h5 files, because they will be read multiple times.')
    parser.add_argument('--csv', required=True, type=str, nargs='+', dest='CSV',
                        help='CSV files with five columns, with frame numbers and movement type (e.g. Switch, '
                             'Rest etc).')
    parser.add_argument('--nframe', type=int, default=25, dest='NFRAMES',
                        help='Total number of frames to consider while training, must be odd.')
    parser.add_argument('--outdir', required=True, type=str, dest='OUTDIR',
                        help='Output directory where the trained models are written.')

    parser.add_argument('--discard', required=False, type=str, nargs='+', dest='DISCARD', default=['NaN'],
                        help='Discard this movement from training. Default is NaN. E.g. --discard NaN Twitch. '
                             'It is case sensitive.')
    parser.add_argument('--basefilter', type=int, required=False, dest='BASEFILTER', default=16,
                        help='Base filter for Inception v3 module. Default is 16, which results in ~22million '
                             'parameters.')
    parser.add_argument('--gpuids', type=str, required=False, dest='GPU', default='0',
                        help='Specifc GPUs to use for training, separated by comma. E.g., --gpuids 3,4,5 ')
    parser.add_argument('--initmodel', type=str, dest='INITMODEL', required=False,
                        help='Existing pre-trained model. If provided, the weights from the pre-trained model will be '
                             'used to initiate the training.')

    results = arg.parse_args()

    if os.getenv('CUDA_VISIBLE_DEVICES') is None or len(os.getenv('CUDA_VISIBLE_DEVICES')) == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = results.GPU
        print('Setting CUDA_VISIBLE_DEVICES to {}'.format(os.getenv('CUDA_VISIBLE_DEVICES')))
    else:
        print('SLURM already sets GPU id to {}, I will not change it.'.format(os.getenv('CUDA_VISIBLE_DEVICES')))
        results.GPU = os.getenv('CUDA_VISIBLE_DEVICES')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    numgpu = len(str(results.GPU).split(','))

    if os.path.isdir(results.OUTDIR):
        os.makedirs(results.OUTDIR, exist_ok=True)

    NF = int(results.NFRAMES)
    if NF != 2 * (NF // 2) + 1:
        sys.exit('Number of frames must be odd, you entered {}'.format(results.NFRAMES))
    if len(results.ATLAS) != len(results.CSV):
        sys.exit('ERROR: Number of images (%d) must be same as number of csvs (%d)'
                 % (len(results.ATLAS), len(results.CSV)))

    atlases = []
    csvs = []
    for j in range(len(results.ATLAS)):
        atlases.append(results.ATLAS[j])
        csvs.append(results.CSV[j])

    y = []
    for i in range(len(csvs)):
        x = pd.read_csv(csvs[i], header=[0])
        x = x.values
        x = x[:, 4]
        y.append(np.unique(list(x)))

    y = list(itertools.chain.from_iterable(y))
    y = np.unique(y)

    y = list(y)
    # discard = ['NaN', 'Twitch', 'Crunch', 'Icrunch']
    # discard = ['NaN', 'Twitch']
    discard = list(results.DISCARD)
    print('Discarded labels = {}'.format(discard))
    for i in range(len(discard)):
        try:
            y.remove(discard[i])
        except:
            pass
        try:
            y.remove(discard[i].lower())
        except:
            pass

    print('{} unique classes found in {} csv files.'.format(len(y), len(csvs)))

    opt = {'numatlas': len(atlases),
           'outdir': results.OUTDIR,
           'atlases': atlases,
           'csvs': csvs,
           'numgpu': numgpu,
           'initmodel': results.INITMODEL,
           'nframes': results.NFRAMES,
           'labels': y,
           'basefilter': results.BASEFILTER,
           }

    pprint(opt)
    main(opt)
