from util import *

def top_dtw_pixels_plot(material, trial_num, normalization=True, subtract_min=False, num_pixels=500, save_img_path=None):


    timestamp, darr = load_trial_data(material, trial_num, subtract_min)
    window = load_pickle(os.path.join(WINDOW_PATH, material, 'trial%d.pkl'%(trial_num)))
    mat = load_pickle(os.path.join(CHE_PATH, material, 'trial%d.pkl'%(trial_num)))
    # mat = load_pickle(os.path.join(DTW_PATH, material, 'trial%d.pkl'%(trial_num)))
    template = quick_normalized_model(timestamp)

    if normalization:
        plt.plot(timestamp, template, 'r', linewidth=1)

    pixels = []
    for i in range(lHeight):
        for j in range(lWidth):
            if inside_window(window, (i,j)):
                pixels.append((mat[i][j], i, j))
                disp_to_term("Pixel (%d,%d) with variance %.2f           " % (i,j,mat[i][j]))
            else:
                disp_to_term("Pixel (%d,%d) not in window           " % (i,j))

    print "\nSorting.."
    pixels = sorted(pixels, key=lambda x: x[0], reverse=True)[:num_pixels]
    print "Done"

    dist_list = []
    for ind, item in enumerate(pixels):
        var, i, j = item
        dist = mat[i][j]
        dist_list.append(dist)
        disp_to_term("Drawing pixel #%d/%d (%d,%d) with dtwdist %.3E      " % (ind, num_pixels, i, j, dist))

        temp = darr[:,i,j]
        if normalization:
            temp = normalize(temp)

        plt.plot(timestamp, temp, linewidth=0.2)
        plt.title('%s %d'%(material, trial_num))

    print "Average DTWDist: ", np.mean(dist_list)
    print '\nDone'

    if save_img_path is None:
        plt.show()
    else:
        savepath = os.path.join(save_img_path, material)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        savefile = os.path.join(savepath, 'trial%d'%(trial_num))

        plt.savefig(savefile)
        plt.close()

def top_che_pixels_plot(material, trial_num, normalization=True, subtract_min=False, num_pixels=500, save_img_path=None):


    timestamp, darr = load_trial_data(material, trial_num, subtract_min)

    pixels = []
    for i in range(lHeight):
        for j in range(lWidth):

            che = darr[:,i,j].max()-darr[:,i,j].min()
            pixels.append((che, i, j))
            disp_to_term("Pixel (%d,%d) with variance %.2f           " % (i,j,che))


    print "\nSorting.."
    pixels = sorted(pixels, key=lambda x: x[0], reverse=True)[:num_pixels]
    print "Done"

    dist_list = []
    for ind, item in enumerate(pixels):
        dist, i, j = item
        dist_list.append(dist)
        disp_to_term("Drawing pixel #%d/%d (%d,%d) with dtwdist %.3E      " % (ind, num_pixels, i, j, dist))

        temp = darr[:,i,j]
        if normalization:
            temp = normalize(temp)

        plt.plot(timestamp, temp, linewidth=0.2)
        plt.title('%s %d'%(material, trial_num))

    # plt.ylim((7420,7520))

    print "Average DTWDist: ", np.mean(dist_list)
    print '\nDone'

    if save_img_path is None:
        plt.show()
    else:
        savepath = os.path.join(save_img_path, material)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        savefile = os.path.join(savepath, 'trial%d'%(trial_num))

        plt.savefig(savefile)
        plt.close()

def average_dtwDist(material, trial_num, top_num=500):
    print material, trial_num
    window = load_pickle(os.path.join(WINDOW_PATH, material, 'trial%d.pkl'%(trial_num)))
    dtwmat = load_pickle(os.path.join(DTW_PATH, material, 'trial%d.pkl'%(trial_num)))

    dtw_list = []
    for i in range(lHeight):
        for j in range(lWidth):
            if inside_window(window, (i,j)):
                dtw_list.append(dtwmat[i][j])

    dtw_list = sorted(dtw_list)[:top_num]
    print dtw_list[0], dtw_list[-1]
    print "Average dtwDist: ", np.mean(dtw_list)

# material, trial_num = ('stainlesssteel_t', 0) # ('stainlesssteel', 0) # ('neoprene',4)

# per_pixel_reading_plot(material, trial_num, diff=20)
# top_variance_pixels_plot(material, trial_num, num_pixels=2000)
# create_video_from_trial(material, trial_num, '../1s/videos', use_min_pixel=False)
# rank_by_dist_then_variance_plot(material, trial_num, normalization=False, subtract_min=False, num_pixels=1000, heating_time=1.0, euclidean=False, downsampling=500)


# # m = 'aluminum'
# # m = 'stainlesssteel'
# # m = 'glass'
# m = 'porcelain'

# #37:15 221  54:30 255

# # range(6,10)
# #
# for trial in range(10):
#     print m, trial
#     calculate_DWT_for_all(m, trial, normalization=True, subtract_min=False, heating_time=1.0, euclidean=False, downsampling=500)

# for m in ['porcelain', 'aluminum']:
#     for trial in range(2):
#         create_window_using_frame_after(m, trial, frame=900, path='../1s/windows')

# visualize_extraxted_pixels(material, trial_num, path='')
# visualize_extracted_pixel_locations(material, trial_num, path='')
# visualize_overall_DWTDist(material, trial_num, path='')
# play_trial(material, trial_num, step=0.000001, subtract_min=False, jump=10)

# window = load_pickle('../1s/windows/glass/trial1.pkl')
# mat = np.zeros((256,324))
# for i in range(256):
#     for j in range(324):
#         if inside_window(window, (i,j)):
#             mat[i][j] = 1.

# plt.matshow(mat)
# plt.show()

# for m in ['neoprene']:
#     for trial in range(3,10):
#         print m, trial
#         generate_DWT_Distance_matrix(m, trial, normalization=True, subtract_min=False, euclidean=False, downsampling=500)
#         # generate_euclidean_Dist_matrix(m, trial, subtract_min=False, base_path='../1s/eucMat')
#         # generate_ChebyshevDistance_matrix(m, trial, subtract_min=False, base_path='../1s/chebMat')
#         # generate_variance_matrix(m, trial, subtract_min=False, base_path='../varianceMat')

# for m in ['wood']:
#     for trial in range(3,10):
#         print m, trial
#         generate_DWT_Distance_matrix(m, trial, normalization=True, subtract_min=False, euclidean=False, downsampling=500)

# path = '../1s/DTWMat'
# for m in difficult:
#     visualize_mat(m, trial_num, path)

# create_difficult_dataset(base_name='difficult', normalization=False, subtract_min=False, num_pixels=500)

# top_DTWDist_pixels_plot(material, trial_num, normalization=False, subtract_min=False, num_pixels=500)
# top_variance_pixels_plot(material, trial_num, normalization=True, subtract_min=False, num_pixels=500)
# top_eucDist_pixels_plot(material, trial_num, normalization=True, smoothing=True, subtract_min=False, num_pixels=50)

# for m in materials:
#     for i in get_trials(m):
#         print m,i
#         top_dtw_pixels_plot(m, i, normalization=False, subtract_min=False, num_pixels=250, save_img_path='../1s/plt/che_unnormalized')
# # top_DTWDist_pixels_plot(material, trial_num, normalization=True, subtract_min=False, num_pixels=5)



# in_window_average_eucDist()

# mm = quick_normalized_model(np.linspace(0.,30.,1500))
# ls = mm - np.mean(mm)
# print np.sum(ls*ls)

# time_series_study(material, trial_num, normalization=True, smoothing=False, subtract_min=False, num_pixels=500)
# in_window_dot_prod()

# for m in materials:
#     for i in get_trials(m):
#         average_dtwDist(m,i)
# #     # average_dtwDist(m, 0)

# for m in materials:
#     for t in range(10):
#         print m,t
#         top_variance_dist_study(m, t, normalization=True, subtract_min=False, num_pixels=500)

# xp = [0,2]
# a = np.ones((2, 3, 4))
# for i in range(3):
#     for j in range(4):
#         a[:,i,j] = i+j
# a[1,:,:] += 1
# b = np.min(a, axis=0)
# print a
# print b
# a -= b
# print a
# print a
# b = a.reshape((2,-1)).T
# print b

# f = interp1d(xp, b, axis=1)
# b = f([0, 0.5, 1, 1.5, 2])

# b = b.reshape(b.shape + (1,1,))
# print b
# print b.shape

# def extract_binary_2(filename):
#     arr = array.array('H')
#     arr.fromfile(open(filename, 'rb'), os.path.getsize(filename)/arr.itemsize)
#     return int(filename.split('/')[-1][:-4]), np.array(arr, dtype='float64').reshape((lHeight, lWidth))


# newp = '../../Data/2minn'
# for filename in glob('../../Data/2min/*.bin'):
#     newfilepath = os.path.join(newp, filename.split('/')[-1])
#     disp_to_term(newfilepath + '             ')

#     with open(filename, 'rb') as f:
#         ba = bytearray(f.read())
#         news = bytearray()
#         for i, c in enumerate(ba):
#             if (i/2)%2==0:
#                 news.append(c)

#         with open(newfilepath, 'wb') as newf:
#             newf.write(news)


# newDATA_PATH = '../../Data/data2'
# for material in ['noise']:
#     for t in get_trials(material):
#         fs = get_trial_files(material, t)

#         newpath = os.path.join(newDATA_PATH, material, 'trial%d'%t)
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)

#         for filename in fs:
#             newfilepath = os.path.join(newpath, filename.split('/')[-1])
#             disp_to_term(newfilepath + '             ')

#             with open(filename, 'rb') as f:
#                 ba = bytearray(f.read())
#                 news = bytearray()
#                 for i, c in enumerate(ba):
#                     if (i/2)%2==0:
#                         news.append(c)

#                 with open(newfilepath, 'wb') as newf:
#                     newf.write(news)

#             # ground,_ = extract_binary(filename)
#             # t,_ = extract_binary_2(newfilepath)

#             # if not np.equal(ground, t):
#             #     print "\n\n\n!!!!!!!!!!!"
#             #     time.sleep(20)

# create_per_pixel_dataset(materials, 'hard', normalization=True, subtract_min=False, num_pixels=500)

# material = 'easy'
# trial_num = 0
# # # print len(get_trial_files(material, trial_num))
# # # top_che_pixels_plot(material, trial_num, normalization=False, subtract_min=False, num_pixels=500, save_img_path=None)
# create_video_from_trial(material, trial_num, base_path='../2min', use_min_pixel=False)



# for m in ['abs_30','abs_50']:
#     top_che_pixels_plot(m, 0, normalization=False, subtract_min=False, num_pixels=500, save_img_path=FIG_PATH)

# model = load_trained_model('../TrainedModels/vardist_50_easy.hdf5')
# # render_trial('neoprene_40', 0, model, FCN_PATH)
# for cc in classes:
#     print(cc+'_50:')
#     classify_video(cc+'_50', 0, model, num_pixels=250)
# for cc in classes:
#     print(cc+'_40:')
#     classify_video(cc+'_40', 0, model, num_pixels=250)
# # classify_video('neoprene_40', 0, model, num_pixels=250)
# # classify_video('neoprene_50', 0, model, num_pixels=250)


######################
# 09/30/2017

# # top_che_pixels_plot('noise', 0, normalization=False, subtract_min=False, num_pixels=10000, save_img_path=None)

# def get_rand_function(num, numiter=2):
#     res = np.random.randn(num)
#     for _ in range(numiter):
#         res -= res.mean()
#         res /= res.std()
#         res = np.cumsum(res)
#         # plt.plot(res)
#         # plt.show()
#     res -= res.min()
#     res /= res.max()
#     return res * np.random.randint(300)

# for i in range(10):
#     a = get_rand_function(300)
#     plt.plot(a)
#     plt.show()

def fit_model(material, trial_num):
    darr = np.load(os.path.join(DATA_PATH, material, 'trial%d'%trial_num, 'vdo.npy'))
    timestamp = np.load(os.path.join(DATA_PATH, material, 'trial%d'%trial_num, 'timestamp.npy'))

    _, h, w = darr.shape
    for i in range(h):
        for j in range(w):
            plt.plot(timestamp, darr[:,i,j])
    plt.ylim([7400, 7770])
    plt.title(material)
    plt.show()


sys.exit(0)