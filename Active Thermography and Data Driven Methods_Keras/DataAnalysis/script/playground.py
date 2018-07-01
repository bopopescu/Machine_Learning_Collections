from util import *
from multiprocessing import Pool

# material = 'cardboard' # 'stainlesssteel' # 'neoprene' # 'abs' # 'porcelain' # 'aluminum' #
# trial_num = 8 # [5,9]

# # play_trial(material, trial_num, 0.00000001, subtract_min=False, jump=10)

# # for material in materials:
# #     for trial_num in trial_nums:
# #         print material, trial_num
# #         create_video_from_trial(material, trial_num, '../1s/videos', use_min_pixel=False)

# # thermistor_plot_all_trial(material)

# # create_per_pixel_dataset(materials, trial_nums=10, base_name='dataset', subtract_min=False, num_pixels=500)

# # per_pixel_variance_plot(material, trial_num)
# create_video_from_trial(material, trial_num, path='', use_min_pixel=False)
# sys.exit(0)

# def work(args):
#     print args
#     m, trial = args
#     generate_DWT_Distance_matrix(m, trial, normalization=True, subtract_min=False, euclidean=False, downsampling=100, base_path=DTW_PATH)


# jobs = []
# for m in materials:
#     ts = check_existing_trials(m, DTW_PATH)
#     for trial in get_trials(m):
#         if trial in ts:
#             continue
#         print m, trial
#         jobs.append((m,trial))
#         # generate_DWT_Distance_matrix(m, trial, normalization=True, subtract_min=False, euclidean=False, downsampling=100, base_path=DTW_PATH)

# print jobs
# pool = Pool(processes=4)
# pool.map(work, jobs)
# for m in materials:
#     for trial in get_trials(m):
#         print m, trial
#         generate_ChebyshevDistance_matrix(m, trial, subtract_min=False, base_path=CHE_PATH)
#         # create_window_using_frame_after(m, trial, frame=900, path=WINDOW_PATH)


# material = 'cardboard'
# trial = 2
# timestamp, darr = load_trial_data(material, trial, False)
# temp = darr[:,41,323].copy()
# temp = resampling(timestamp, temp, pts=100)
# template = quick_normalized_model(np.linspace(0., TRIAL_INTERVAL, num=100))
# print DTWDist(normalize(temp), template)

# create_per_pixel_dataset(materials, base_name='hard', normalization=True, subtract_min=False, num_pixels=500)
# create_informative_dataset(materials, base_name='informative', normalization=True, subtract_min=False, binary=True)

# model = load_trained_model('../TrainedModels/hard.hdf5')
# # # model = load_pickle('../TrainedModels/svm_easy.pkl')
# for m in materials:
#     for t in get_trials(m):
#         print m, t
#         render_trial(m, t, model, FCN_PATH)

# m, t = ('6mat_1',0) # ('porcelain', 0) # ('neoprene', 0) # ('wood', 0) # ('neoprene', 0) # ('aluminum', 6) #
# display_class_activations(m, t, FCN_PATH)
# render_trial(m, t, model, FCN_PATH)
# classify_video(m,t,model)


#############
# 09/30/2017
# material = 'neoprene_30'
# play_trial(material, 0, step=0.000001, subtract_min=False, normalization=False, window=vdo_window, jump=1)
# generate_ChebyshevDistance_matrix(material, 0, subtract_min=False, base_path=CHE_PATH)

# chemat = load_pickle(os.path.join(CHE_PATH, material, 'trial0.pkl'))
# thresh =  6 # 8 - 4mat_35
# chemat[chemat < thresh] = 0
# chemat[chemat >= thresh] = 200
# plt.matshow(chemat)
# plt.colorbar()
# plt.show()

# for m in materials:
#     # create_window_using_frame_after(m, 0, frame=950, path=WINDOW_PATH)
#     generate_ChebyshevDistance_matrix(m, 0, subtract_min=False, base_path=CHE_PATH)

# create_per_pixel_dataset(dist_50, 'dist_50_easy', normalization=True, subtract_min=False, num_pixels=500)

# new_path = '../../Data/cropped'
# def save_cropped_vdo():
#     for m in dist_30:
#         for trial in get_trials(m):
#             timestamp, darr = load_trial_data(m, trial, False, normalization=False, use_min_pixel=False, window=vdo_window, t_limit=OBSERVATION_INTERVAL)
#             newfile = os.path.join(new_path, m, 'trial%d'%trial)
#             if not os.path.exists(newfile):
#                 os.makedirs(newfile)
#             np.save(os.path.join(newfile,'vdo'), darr)
#             np.save(os.path.join(newfile,'timestamp'), timestamp)


# save_cropped_vdo()

def display_region_time_series(material, trial, color):
    k_size = 17
    kernel = np.ones((1,k_size,k_size)) / (k_size ** 2.)

    darr = np.load(os.path.join(DATA_PATH, material, 'trial%d'%trial, 'vdo.npy'))
    timestamp = np.load(os.path.join(DATA_PATH, material, 'trial%d'%trial, 'timestamp.npy'))

    # darr = convolveim(darr, kernel, mode='constant')
    # darr = darr[:, k_size//2:-(k_size//2), k_size//2:-(k_size//2)]

    # _, h, w = darr.shape
    # for i in range(h):
    #     for j in range(w):
    #         temp = darr[:,i,j].copy()
    #         # temp = resampling(timestamp, temp, 200)
    #         # # temp = normalize(temp)
    #         temp = normalize2(temp)
    #         plt.plot(timestamp,temp,color)
    plt.plot(timestamp, darr.mean(axis=(1,2)), color, label=material)
    # print timestamp.max()
    # plt.title(material)
    plt.xlabel('Time(s)')
    plt.ylabel('Thermal Camera Reading')




create_regional_dataset(dist_30, 'cropped_k15s4', normalization=True)

# base_mats = ['neoprene', 'abs', 'wood', 'cardboard']
# for base_mat in base_mats:
#     plt.figure()
#     display_region_time_series('%s_30'%base_mat, 1, 'r')
#     display_region_time_series('%s_40'%base_mat, 1, 'g')
#     display_region_time_series('%s_30_30'%base_mat, 1, 'b')
#     # plt.ylim([7400,7770])
#     plt.title('%s standardized'%base_mat)
#     plt.legend()
#     plt.savefig(base_mat)


# plt.figure()
# display_region_time_series('acrylic_30', 11, 'r')
# # for i in range(100):
# #     display_region_time_series('aluminum_30', i, 'r')
# #     display_region_time_series('stainlesssteel_30', i, 'b')
# # display_region_time_series('%s_40'%base_mat, 1, 'g')
# # display_region_time_series('%s_30_30'%base_mat, 1, 'b')
# # plt.ylim([7400,7770])
# # plt.title('%s standardized'%base_mat)
# # plt.legend()
# plt.show()
# # plt.savefig(base_mat)

# model = load_trained_model('../TrainedModels/dist_30_split.hdf5')
# for m in dist_30:
#     labels = []
#     for i in range(100):
#         label = classify_regions(m, i, model, dist_30)
#         labels.append(label)
#     print 'Current Material: ', m
#     print Counter(labels)

# for m in ['castiron_40', 'stainlesssteel_40', 'aluminum_40']:
#     print "Material: ", m
#     for i in range(4):
#         classify_regions(m, i, model, dist_40)
# # for m in agl_30:
# #     for trial in range(3):
# #         classify_regions(m, trial, model, agl_30)

# def show_region(material, trial_num):
#     darr = np.load(os.path.join(DATA_PATH, material, 'trial%d'%trial_num, 'vdo.npy'))
#     timestamp = np.load(os.path.join(DATA_PATH, material, 'trial%d'%trial_num, 'timestamp.npy'))

#     _, h, w = darr.shape
#     for i in range(h):
#         for j in range(w):
#             plt.plot(timestamp, darr[:,i,j])
#     # plt.ylim([7400, 7770])
#     plt.title(material)
#     plt.show()


# for m in dist_30:
#     plt.figure()
#     for i in range(10):
#         display_region_time_series(m, i)
#     plt.title(m)
#     plt.savefig(m)

# for i in range(10):
#     display_region_time_series('aluminum_30', i)
# plt.show()


# for m in dist_30:
#     knn_classify_regions(m, 66, 0)
# for m in dist_40:
#     knn_classify_regions(m, 66, 0)
# for m in agl_30:
#     for trial in range(3):
#         knn_classify_regions(m, trial, 0)
