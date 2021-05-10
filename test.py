import numpy as np
import time, os, sys,datetime, argparse, warnings, h5py, copy
warnings.filterwarnings("ignore")
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Model, load_model
from skimage import io
from tqdm import  tqdm
from sklearn.mixture import GaussianMixture
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



if __name__ == "__main__":
    arg = argparse.ArgumentParser(description='Testing for movement detection')
    parser = arg.add_argument_group('Required arguments')
    parser.add_argument('--image', required=True, type=str, dest='IMAGE',
                        help='Image to test, must have same size as the atlases.')
    parser.add_argument('--csv', required=True, type=str, dest='CSV',
                        help='Output CSV files.')
    parser.add_argument('--nframe', type=int, dest='NFRAMES', required=True,
                        help='Number of frames used while training.')
    parser.add_argument('--model', type=str, dest='MODEL', required=True,
                        help='Trained model. ')
    parser.add_argument('--discard', type=str, nargs='+', dest='DISCARD', required=False, default=['NaN'],
                        help='Discarded labels during training. Default is NaN')

    parser.add_argument('--gpuid', type=str, required=False, dest='GPU', default='0',
                        help='Specifc GPU to use. Default 0. Optional.')

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


    print('Reading {}'.format(results.IMAGE))
    exten = os.path.splitext(results.IMAGE)[1]
    if exten.lower() == '.tif':
        x = io.imread(results.IMAGE)
        #x = np.transpose(x,[0,2,1])  # @TODO, this is temporary, atlas and subject must be both either tif or h5
    elif exten.lower() == '.h5':
        f = h5py.File(results.IMAGE, 'r')
        dataset = list(f.keys())
        if len(dataset) > 1:
            print('WARNING: More than one key found in the h5 file {}'.format(results.IMAGE))
            print('WARNING: Reading only the first key {} '.format(dataset[0]))
        x = f[dataset[0]]

    else:
        sys.exit('ERROR: Images must be either .tif or .h5 files. You entered {}'.format(results.IMAGE))

    x = np.asarray(x, dtype=np.float32)
    print('Image size = {}'.format(x.shape))

    print('Finding a scaling factor..')
    S = 1000000
    indx = np.random.choice(np.prod(x.shape), size=(S, 1), replace=False)
    indx = np.unravel_index(indx, x.shape, order='C')
    indx = np.asarray(indx, dtype=int)
    #print(indx.shape)
    y = np.zeros((S, 1), dtype=np.float32)
    for j in range(0, S):
        y[j] = x[indx[0, j, 0], indx[1, j, 0], indx[2, j, 0]]

    y = y[y > 0]
    y = np.asarray(y, dtype=np.float32)

    gmm = GaussianMixture(n_components=2, covariance_type='diag', tol=0.001,
                          reg_covar=1e-06, max_iter=100, n_init=1, precisions_init=None,
                          weights_init=(0.9, 0.1), init_params='kmeans',
                          means_init=np.reshape((20, 400), (2, 1)),
                          warm_start=True, verbose=1, verbose_interval=1)
    gmm.fit(y.reshape(-1, 1))

    y = gmm.means_[1]

    print('Scaling image by %.2f' % (y))
    x = x / y
    

    dim = x.shape
    dframes = int(results.NFRAMES)//2

    movements = ['AntComp','BotSwing','Brace', 'Crunch','Dgap','Lift','Pgap','Rest','Swing','Twitch']

    discard = list(results.DISCARD)
    print('Discarded labels = {}'.format(discard))
    for i in range(0,len(discard)):
        try:
            movements.remove(discard[i])
        except:
            pass

    movements = list(movements)

    model = load_model(results.MODEL)
    t1 = time.time()

    Y = np.ones((dim[0],1),dtype=int)
    Y = Y*(7) 
    X = np.zeros((1,dim[1],dim[2],results.NFRAMES),dtype=np.float32)
    for l in tqdm(range(dframes,dim[0]-dframes)):
        X[0,:,:,:] = np.transpose(x[l-dframes:l+dframes+1,:,:],[1,2,0])
        y = model.predict(X)
        Y[l] = np.argmax(y, axis=1)

    Y = np.asarray(Y, dtype=int)
    print(np.unique(Y))
    Y=list(Y)

    with open(results.CSV, 'w') as out:
        for l in range(dim[0]):
            if Y[l] >= -1:
                indx = int(Y[l])
                #print('Frame {} = {}'.format(l,movements[indx]))
                out.write('{},{}'.format(l+1,movements[indx]))
                out.write('\n')

    t2 = time.time()
    print('Elapsed time = {}'.format(datetime.timedelta(seconds=(t2 - t1))))
