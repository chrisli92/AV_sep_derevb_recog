
gpu_id = '2'
task = 'mvdr'
# ================ Data processing =================
sampling_rate = 16000  # Hz
out_spk = 1
batch_size = 32
max_epoch = 100
diag_loading_ratio='1e-5'
add_visual=False # or False
AF_premasking=False
resume=False
use_torch_solver=True

# ========================path ==========
# original mixture pickle path
training_path = "/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data/train_pretrain_32_rev1_le6_ark.pkl"
validation_path = "/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data/val_rev4_le6_ark.pkl"
test_path = "/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data/test_rev4.pkl"
replay_path = "/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data/test_replay.pkl"
lrs3_test_path = "/users/bdda/gnli/data/data_bdda7/MC_MS_Simu/egs/lrs3/pkl_data/test_rev4.pkl"


# ================= Network ===================== #
sample_duration = 0.0025  # s / 2.5ms
L = int(sampling_rate * sample_duration)  # length of the filters [20] samples
N = 256  # number of filters in encoder
B = 256  # number of channels in bottleneck 1x1-conv block
H = 512  # number of channels in convolutional blocks
P = 3  # kernel size in convolutional bolcks
X = 8  # number of convolutional blocks in each repeat
R = 4  # number of repeats
V = 256
U = 128
norm = "BN"  # /cLN/gLN/BN
causal = False
activation_function = 'linear'  # /sigmoid/softmax/linear/relu
model_type = 'uTGT'  # 'hard-encoder-tasnet/encoder-beam-tasnet/beam-tasnet/dir-informed-tgt-tasnet/dir-informed-pit-tasnet
fusion_idx = 0
av_fusion_idx = 1
fix_stft = True
cosIPD = True
sinIPD = False

# ================= DF settings ===================== #
debug_mode = 0
input_features = ['LPS', 'IPD', 'AF']
# input_features = ['LPS', 'IPD']
speaker_feature_dim = 1  # or 2
# 0.56, 0.42, 0.3, 0.28m, 0.2, 0.12,0.1,0.05,0.01
mic_pairs = [[0, 14], [1, 13], [2, 12], [0, 6], [11, 3], [10, 4], [11, 7], [6, 9], [7, 8]]
n_mic = 15
merge_mode = 'sum'
FFT_SIZE = 512
HOP_SIZE = 256
NEFF = FFT_SIZE // 2 + 1
factor = 10
lip_fea = 'lipemb' # or 'landmark'
# loss_type = ['SISNR', 'LFB', 'SISNR_NOISE'] #or 'SISNR'
loss_type = ['SISNR', 'SISNR_NOISE'] 

# ================= Training ===================== #
seed = 0
log_display_step = 100
if task=='tf-masking':
    lr = 1e-3
else: # for mvdr
    lr = 5e-4
lr_decay = 1e-5
cuda = True  # True
max_error_deg = 5
s_max_keep_inteval = int(sampling_rate * 1.0)
s_min_keep_inteval = int(sampling_rate * 0.25)
num_mel_bins = 40
# =================Training =======================#
if not AF_premasking:
    if resume:
        training_model_subpath="model_mvdr-AF_premasking=False/uTGT-LPS_IPD_AF-b12-BN-fd1-fixTrue-f0-cosTrue-sinFalse-last-55.pt.tar"
    else:
        training_model_subpath="model_tfmasking-AF_premasking=False-Batch_Size=64/uTGT-LPS_IPD_AF-b64-BN-fd1-fixTrue-f0-cosTrue-sinFalse-best-57.pt.tar"

# ================= Evaluating ===================== #
replay=False
write_wav = False
all_metrics = True
inference_dataset_name = 'lrs3_test' # val train_pretrain test replay
ckpt_epoch = 79  # 76 78 73
# wav_dir_name = f"{inference_dataset_name}-{ckpt_epoch}"
wav_dir_name = f"{inference_dataset_name}_eps={ckpt_epoch}_dl={diag_loading_ratio}"
ckpt = f"uTGT-LPS_IPD_AF-b32-BN-fd1-fixTrue-f0-cosTrue-sinFalse-best-{ckpt_epoch}.pt.tar"


# ================= Directory ===================== #
data_dir = '/project_bdda7/bdda/gnli/Data/lrs2_new/pkl_data'
sum_dir = './summary_0{}-{}-{}'.format(lip_fea, '+'.join(input_features), '+'.join(loss_type))
model_save_dir = sum_dir + f'/model_mvdr-AF_premasking={AF_premasking}-Batch_Size={batch_size}-Dloading={diag_loading_ratio}/'



log_dir = model_save_dir + '/log/'
loss_dir = sum_dir + '/loss/'
model_name = '{}-{}-b{}-{}-fd{}-fix{}-f{}-cos{}-sin{}'.format(model_type,
                                                              '_'.join(input_features),
                                                              batch_size, norm, speaker_feature_dim, fix_stft,
                                                              fusion_idx, cosIPD, sinIPD)

# import os
if write_wav:
    write_name = f'{model_save_dir}/{wav_dir_name}'
#     if not os.path.exists(write_name):
#         os.makedirs(write_name)
