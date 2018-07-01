experiment_name = 'cropped' # 'region'

# DATA_PATH = '../data'
# DATA_PATH = '/Volumes/LACIE SHARE/CamData/1s_with_p'
# DATA_PATH = '/Volumes/LACIE SHARE/CamData/4s_with_p'
# DATA_PATH = '/Volumes/LACIE SHARE/CamData/1s'
# DATA_PATH = '../../Data/1s'
# DATA_PATH = '../../Data/2min'
# DATA_PATH = '../../Data/allmat2'
DATA_PATH = '../../Data/%s' % experiment_name

WINDOW_PATH = '../%s/windows' % experiment_name
VIDEO_PATH = '../%s/videos' % experiment_name
FIG_PATH = '../%s/figures' % experiment_name
DTW_PATH = '../%s/DTWMat' % experiment_name
VAR_PATH = '../%s/VarMat' % experiment_name
EUC_PATH = '../%s/eucMat' % experiment_name
CHE_PATH = '../%s/cheMat' % experiment_name

# FCN_PATH = '../1s/FCN_plt'
# FCN_PATH = '../1s/easyFCN_plt'
FCN_PATH = '../%s/FCN_plt_generalization' % experiment_name

SVM_PATH = '../%s/SVM_plt' % experiment_name

PORT_NUM = 9999

lWidth = 324
lHeight = 256
IMG_SIZE_BYTES = lWidth * lHeight * 2
FPS = 60.0

NUM_TRIALS = 10
REST_INTERVAL = 5.0
TRIAL_INTERVAL = 10.0
HEAT_INTERVAL = 1.0
OBSERVATION_INTERVAL = 5.0
NUM_OBSERVATIONS = 200

# NUM_TRIALS = 1
# REST_INTERVAL = 5.0
# TRIAL_INTERVAL = 120.0
# # HEAT_INTERVAL = 1.0
# OBSERVATION_INTERVAL = 120.0
# # NUM_OBSERVATIONS = 100

temp_dev_nm = '/dev/cu.usbmodem2673411' # thermal teensy serial number
baudrate = 115200
temp_dev = []
temp_inputs = 1
freq = 100.
check_time = .00067
k_check_time = .002


# materials = ['abs', 'aluminum', 'cardboard', 'glass', 'neoprene', 'porcelain', 'stainlesssteel', 'wood']
# materials = ['abs', 'cardboard', 'neoprene', 'wood', 'aluminum', 'glass', 'porcelain', 'stainlesssteel']
# difficult = ['aluminum', 'glass', 'porcelain', 'stainlesssteel']
# materials = ['abs', 'cardboard', 'neoprene', 'wood']
# materials = ['abs', 'cardboard', 'neoprene', 'wood', 'aluminum', 'glass', 'porcelain', 'stainlesssteel', 'tape', 'neoprene_t', 'aluminum_t', 'stainlesssteel_t']
# taped = ['tape', 'neoprene_t', 'aluminum_t', 'stainlesssteel_t']
# materials = ['easy']
# materials = ['4mat_1', '6mat_1']
# materials = ['4mat_2', '6mat_2'] + ['4mat_3', '6mat_3'] + ['4mat_4', '6mat_4'] + ['4mat_5', '6mat_5']
# materials = ['4mat_35', '4mat_45', '4mat_55', '4mat_65', '4mat_75', '4mat_85', '4mat_85_2']
# materials = ["abs_30", "cardboard_30", "glass_30", "rubber_30", "abs_40", "cardboard_40", "glass_40", "rubber_40", "abs_50", "cardboard_50", "glass_50", "rubber_50", "bear_30", "castiron_30", "neoprene_30", "wood_30", "bear_40", "castiron_40", "neoprene_40", "wood_40", "bear_50", "castiron_50", "neoprene_50", "wood_50", "bottle_30", "cup_30", "plastic_30", "bottle_40", "cup_40", "plastic_40", "bottle_50", "cup_50", "plastic_50"]
# classes = ['abs', 'cardboard', 'rubber', 'glass', 'bear', 'castiron', 'neoprene', 'wood', 'plastic', 'cup', 'bottle']
# classes = ['abs', 'cardboard', 'bear', 'castiron', 'neoprene', 'plastic', 'bottle']
easy = ['abs', 'cardboard', 'neoprene', 'wood']
hard = ['aluminum', 'brick', 'eps', 'foam', 'polyester', 'uhmw']
additional = ['acrylic', 'castiron', 'granite', 'mdf', 'stainlesssteel']
classes = easy + hard + additional
dist_30 = [i+'_30' for i in classes]
dist_40 = [i+'_40' for i in classes]
dist_50 = [i+'_50' for i in classes]
agl_30 = [i+'_30_30' for i in classes]

NUM_CLASSES = len(classes)

# 53 * 37
vdo_window = (102, 134, 245, 277)


# Smoothing
order = 8 # 5?
fs = 100.0
cutoff = 2


