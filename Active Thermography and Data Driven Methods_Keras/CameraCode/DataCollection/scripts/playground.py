from util import *

# display_binary(os.path.join(DATA_PATH, 'test', 'trial0', ))

material = 'test' # 'stainlesssteel' # 'neoprene' # 'abs' # 'porcelain' # 'aluminum' #
trial_num = 0 # [5,9]

play_trial(material, trial_num, step=0.000001, subtract_min=False, jump=10)


# res = []
# fs = get_trial_files(material, trial_num)

# for f in fs:
#     _,data = extract_binary(f)
#     data1 = data[:,0::2]
#     data2 = data[:,1::2]

#     # print data
#     # print data1.shape, data2.shape
#     res.append((data1 == data2).all())

# print all(res)