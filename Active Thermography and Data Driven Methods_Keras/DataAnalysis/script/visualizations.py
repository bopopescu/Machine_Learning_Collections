from util import *
import shutil
from matplotlib import colors as mcolors
from sklearn.decomposition import PCA

from keras.models import *
from keras.callbacks import *
import keras.backend as K
K.set_learning_phase(0)

VIS_PATH = '../region_figures'
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
# colors = mcolors.BASE_COLORS
# print colors


######################################################
# 0 Helper Functions
######################################################
def check_path(target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)

def gkern(kernlen, nsig):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    kernel = np.expand_dims(kernel, axis=0)
    return kernel

def conv(darr, k_size, stride, gaussian=False, sigma=None):
    if gaussian:
        kernel = gkern(k_size, sigma)
    else:
        kernel = np.ones((1,k_size,k_size)) / (k_size ** 2.)
    kstart = int(np.floor((k_size-1)/2.))
    kend = -int(np.ceil((k_size-1)/2.))

    darr = convolveim(darr, kernel, mode='constant')
    darr = darr[:, kstart:kend:stride, kstart:kend:stride]
    return darr

######################################################
# 1 Display the mean of each material on same plot
######################################################
def display_all_mats_mean():
    for c, m in enumerate(dist_30):
        print m
        tseries = []
        for t in range(100):
            darr = np.load(os.path.join(DATA_PATH, m, 'trial%d'%t, 'vdo.npy'))
            timestamp = np.load(os.path.join(DATA_PATH, m, 'trial%d'%t, 'timestamp.npy'))

            darr = darr.mean(axis=(1,2))
            temp = resampling(timestamp, darr, 200)

            tseries.append(temp)
        tseries = np.array(tseries)
        tseries = tseries.mean(axis=0)
        tseries = normalize2(tseries)
        print tseries.shape

        plt.plot(np.linspace(0,5,200), tseries, color=colors.keys()[c], label=m)

# display_all_mats_mean()
# plt.legend(loc='lower right',
#           ncol=3, fancybox=True, shadow=True)
# plt.show()


######################################################
# 2.1 Effect of different heating time
######################################################
def simple_model(timestamps, t_h, include_heating=True, normalization=False):
    print t_h
    template = []

    if include_heating:
        for t in timestamps:
            if t < t_h:
                template.append(np.sqrt(t))
            else:
                template.append(np.sqrt(t) - np.sqrt(t - t_h))
        template = np.array(template)
    else:
        template = np.sqrt(timestamps + t_h) - np.sqrt(timestamps)

    if normalization:
        template = normalize(template)

    return template

## Study effect of heating time
def different_heating_times():
    tstamp = np.linspace(0,10,500)
    for t in np.linspace(1,5,11):
        # data = simple_model(tstamp, t, False, True)
        data = simple_model(tstamp, t, True, False)
        plt.plot(tstamp, data)
    plt.show()

# different_heating_times()

######################################################
# 2.2 Intrinsic Heat Response
######################################################
def display_region_time_series(material, trial, color):

    darr = np.load(os.path.join(DATA_PATH, material, 'trial%d'%trial, 'vdo.npy'))
    timestamp = np.load(os.path.join(DATA_PATH, material, 'trial%d'%trial, 'timestamp.npy'))

    temp = darr.mean(axis=(1,2))
    temp = resampling(timestamp, temp, 200)
    temp = normalize2(temp)

    plt.plot(np.linspace(0,5,200), temp, color, label=material)

    plt.xlabel('Time(s)')
    plt.ylabel('Thermal Camera Reading')


# tar_path = os.path.join(VIS_PATH, 'intrinsic_resp')
# check_path(tar_path)
# base_mats = ['neoprene', 'abs', 'wood', 'cardboard']
# for base_mat in base_mats:
#     plt.figure()
#     display_region_time_series('%s_30'%base_mat, 1, 'r')
#     display_region_time_series('%s_40'%base_mat, 1, 'g')
#     display_region_time_series('%s_30_30'%base_mat, 1, 'b')
#     # plt.ylim([7400,7770])
#     plt.title('%s standardized'%base_mat)
#     plt.legend()
#     plt.savefig(os.path.join(tar_path, base_mat))

######################################################
# 3 PCA of dataset (by material and by trial)
######################################################
def show_dataset_pca_by_material():
    pca = PCA(n_components=2)

    dataset = []
    for c, m in enumerate(dist_30):
        print m
        tseries = []
        for t in range(100):
            darr = np.load(os.path.join(DATA_PATH, m, 'trial%d'%t, 'vdo.npy'))
            timestamp = np.load(os.path.join(DATA_PATH, m, 'trial%d'%t, 'timestamp.npy'))

            darr = darr.mean(axis=(1,2))
            temp = resampling(timestamp, darr, 200)
            temp = normalize2(temp)

            tseries.append(temp)
        tseries = np.array(tseries)
        tseries = tseries.mean(axis=0)
        dataset.append(tseries)

    pca.fit(dataset)
    lowdim = pca.transform(dataset)
    lowdim = np.split(lowdim, NUM_CLASSES)
    for idx, item in enumerate(lowdim):
        plt.scatter(item[:,0], item[:,1], label=dist_30[idx])

def show_full_dataset_pca():
    pca = PCA(n_components=2)

    dataset = []
    for c, m in enumerate(dist_30):
        print m
        for t in range(100):
            darr = np.load(os.path.join(DATA_PATH, m, 'trial%d'%t, 'vdo.npy'))
            timestamp = np.load(os.path.join(DATA_PATH, m, 'trial%d'%t, 'timestamp.npy'))

            darr = darr.mean(axis=(1,2))
            temp = resampling(timestamp, darr, 200)
            temp = normalize2(temp)

            dataset.append(temp)

    pca.fit(dataset)
    lowdim = pca.transform(dataset)
    lowdim = np.split(lowdim, NUM_CLASSES)
    for idx, item in enumerate(lowdim):
        plt.scatter(item[:,0], item[:,1], label=dist_30[idx])

# show_dataset_pca_by_material()
# show_full_dataset_pca()
# plt.legend()
# plt.show()

######################################################
# 4 Display the PCA of extracted features
######################################################
def feature_extract(model, data, class_num): # take in standardized data
    data = data.reshape((1,) + data.shape + (1,1,))

    class_weights = model.layers[-1].get_weights()[0]
    final_conv_layer = model.layers[-2]
    get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([data])

    print "Predicted Class: %s" % classes[np.argmax(predictions)]
    return conv_outputs[0,:]

def feature_pca():
    pca = PCA(n_components=2)

    model = load_trained_model('../TrainedModels/dist_30_split.hdf5')
    # print model.layers

    dataset = []
    for c, m in enumerate(dist_30):
        print m
        for t in range(100):
            darr = np.load(os.path.join(DATA_PATH, m, 'trial%d'%t, 'vdo.npy'))
            timestamp = np.load(os.path.join(DATA_PATH, m, 'trial%d'%t, 'timestamp.npy'))

            darr = darr.mean(axis=(1,2))
            temp = resampling(timestamp, darr, 200)
            temp = normalize2(temp)

            feature = feature_extract(model, temp, c)

            dataset.append(feature)

    pca.fit(dataset)
    lowdim = pca.transform(dataset)

    lowdim = np.split(lowdim, NUM_CLASSES)
    for idx, item in enumerate(lowdim):
        plt.scatter(item[:,0], item[:,1], label=dist_30[idx])

# feature_pca()
# plt.legend()
# plt.show()


######################################################
# 5 Delete bad trials
######################################################
def delete_bad_trials(m):
    while len(get_trials(m)) > 90:
        dataset = []
        ts = get_trials(m)
        for t in ts:
            darr = np.load(os.path.join(DATA_PATH, m, 'trial%d'%t, 'vdo.npy'))
            timestamp = np.load(os.path.join(DATA_PATH, m, 'trial%d'%t, 'timestamp.npy'))

            darr = darr.mean(axis=(1,2))
            temp = resampling(timestamp, darr, 200)
            temp = normalize2(temp)

            dataset.append(temp)
        dataset = np.array(dataset)
        avg = dataset.mean(axis=0)

        bookkeep = []
        for idx, item in enumerate(dataset):
            dist = np.sqrt(np.sum((item - dataset)**2.))
            bookkeep.append((ts[idx],dist))

        bookkeep = sorted(bookkeep, key=lambda x: x[-1], reverse=True)
        print len(bookkeep), bookkeep[0]

        shutil.rmtree(os.path.join(DATA_PATH, m, 'trial%d'%bookkeep[0][0]))
        time.sleep(0.2)

def rename_trials(m):
    counter = 0
    for idx in get_trials(m):
        os.rename(os.path.join(DATA_PATH, m, 'trial%d'%idx), os.path.join(DATA_PATH, m, 'trial%d'%counter))
        counter += 1

def visualize_trials(m):
    ts = get_trials(m)
    for t in ts:
        darr = np.load(os.path.join(DATA_PATH, m, 'trial%d'%t, 'vdo.npy'))
        timestamp = np.load(os.path.join(DATA_PATH, m, 'trial%d'%t, 'timestamp.npy'))

        darr = darr.mean(axis=(1,2))
        temp = resampling(timestamp, darr, 200)
        temp = normalize2(temp)

        plt.plot(temp)
    plt.title(m)
    plt.show()


# for mat in ['granite_30']:
#     print mat
#     # delete_bad_trials(mat)
#     # rename_trials(mat)
#     # print get_trials(mat)

#     visualize_trials(mat)


######################################################
# 6 Display the extracted time series
######################################################

def show_extracted_tseries(m, t, k_size, stride, gaussian=False, sigma=1):
    darr = np.load(os.path.join(DATA_PATH, m, 'trial%d'%t, 'vdo.npy'))
    timestamp = np.load(os.path.join(DATA_PATH, m, 'trial%d'%t, 'timestamp.npy'))

    darr = conv(darr, k_size, stride, gaussian=gaussian, sigma=sigma)

    _, h, w = darr.shape
    for i in range(h):
        for j in range(w):
            temp = darr[:,i,j]
            temp = resampling(timestamp, temp, 200)
            temp = normalize2(temp)

            plt.plot(np.linspace(0,5,200), temp)

    plt.title(m)
    # plt.show()

# plt.figure()
# show_extracted_tseries('wood_30', 6, 17, 2)
# plt.figure()
# show_extracted_tseries('wood_30', 6, 15, 4, True)
# plt.figure()
# show_extracted_tseries('wood_30', 6, 15, 4, True, 0.5)
# plt.figure()
# show_extracted_tseries('wood_30', 6, 15, 4, True, 0.25)
# plt.figure()
# show_extracted_tseries('wood_30', 6, 15, 4, True, 0.125)
# plt.show()

######################################################
# 7 Create Confusion Matrix (Pixel mode and Video 'vote' mode)
######################################################
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def classify_regions(material, trial_num, model, k_size, stride, gaussian):
    darr = np.load(os.path.join(DATA_PATH, material, 'trial%d'%trial_num, 'vdo.npy'))
    timestamp = np.load(os.path.join(DATA_PATH, material, 'trial%d'%trial_num, 'timestamp.npy'))

    darr = conv(darr, k_size, stride, gaussian=gaussian)

    _, h, w = darr.shape

    X = []
    for i in range(h):
        for j in range(w):
            temp = darr[:,i,j]
            temp = resampling(timestamp, temp, 200)
            temp = normalize2(temp)
            X.append(temp)

    X = np.array(X)
    X = X.reshape(X.shape + (1,1,))
    # print X.shape

    ys = model.predict(X)
    # print 'Result shape: ', ys.shape

    cs = np.argmax(ys, axis=1).astype('int')
    scores = np.zeros_like(ys)
    scores[range(ys.shape[0]), cs] = 1.

    return scores.sum(axis=0)

def create_conf_mat(materials, model_path, k_size, stride, gaussian=False, pixel=True):
    model = load_trained_model(model_path)

    conf_mat = []
    for m in materials:
        res = np.zeros((1, NUM_CLASSES))
        for t in range(50, 100):
            disp_to_term('%s %d            '%(m, t))
            subres = classify_regions(m, t, model, k_size, stride, gaussian=gaussian)

            if pixel:
                res += subres
            else:
                temp = np.zeros_like(subres)
                temp[subres.argmax()] = 1.
                res += temp


        conf_mat.append(res[0,:].astype('int'))

    dia = np.diag(conf_mat)
    order = sorted(range(len(dia)), key=lambda k: dia[k], reverse=True)
    mats = [classes[i] for i in order]
    conf_mat = np.array([val[order] for val in conf_mat])
    conf_mat = conf_mat[order]

    print '\n\n'
    plot_confusion_matrix(conf_mat, mats,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues)
    plt.show()

create_conf_mat(dist_30, '../TrainedModels/k17s2.hdf5', 17, 2, pixel=False)


######################################################
# 8 CAM
######################################################

def cam(model, class_num, material, trial_num, k_size, stride, gaussian=False):

    darr = np.load(os.path.join(DATA_PATH, material, 'trial%d'%trial_num, 'vdo.npy'))
    timestamp = np.load(os.path.join(DATA_PATH, material, 'trial%d'%trial_num, 'timestamp.npy'))

    darr = conv(darr, k_size, stride, gaussian=gaussian)

    darr = darr.mean(axis=(1,2))
    darr = resampling(timestamp, darr, 200)
    darr = normalize2(darr)

    orig = darr.copy()
    data = darr.reshape((1,) + darr.shape + (1,1,))
    # print data.shape

    #Get the 512 input weights to the softmax.
    class_weights = model.layers[-1].get_weights()[0]
    final_conv_layer = model.layers[-3]
    get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([data])
    conv_outputs = conv_outputs[0, :, :, :]

    print "%s %d Predicted Class: %s" % (material, trial_num, classes[np.argmax(predictions)])

    #Create the class activation map.
    cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])
    for i, w in enumerate(class_weights[:, class_num]):
        cam += w * conv_outputs[:, :, i]
    cam /= np.max(cam)
    cam[cam < 0] = 0.

    plt.plot(range(200), orig, c='c', zorder=1, linewidth=0.4, alpha=0.9) # b, winter
    plt.scatter(range(200), orig, c=cam, s=(((cam * 5)**2) / 5.), alpha=1.0, zorder=2, cmap='cool') # brg # seismic


def gen_cam(model_path, fig_path, k_size, stride, gaussian=False):
    model = load_trained_model(model_path)
    for cn, m in enumerate(dist_30):
        print 'Curr Mat: ', m
        plt.figure()
        for i in range(50,100):
            cam(model, cn, m, i, k_size, stride, gaussian=gaussian)
        plt.colorbar()
        plt.title(m)
        plt.savefig(os.path.join(fig_path, m))

# gen_cam('../TrainedModels/k17s2.hdf5', '../region_figures/cam', 17, 2)



