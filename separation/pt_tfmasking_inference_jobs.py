import numpy as np
import soundfile
from params import *
from data_generator_jobs import CHDataset
import torch
import os
# from pt_dervb_net import DervbNet
from pt_avnet import ConvTasNet
from pt_trainer import get_logger
import sys
import result_analysis as ra
from signalprocess import audiowrite, audioread
# from pt_avnet_mvdr import ConvTasNet



gpuid = int(gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'

logger = get_logger(__name__)

logger.info(f"hop_size:{HOP_SIZE}, diag_loading_ratio: {diag_loading_ratio}, tfmasking")

if replay:
    print(f"this test is for replay")
    from data_generator_for_test_replay import CHDataset, CHDataLoader
else:
    from data_generator_jobs import CHDataset, CHDataLoader


class NNetComputer(object):
    def __init__(self, save_model_dir=model_save_dir, gpuid=gpuid):
        print("model name is {}".format(ckpt))
        nnet = ConvTasNet(norm=norm,
                          out_spk=out_spk,
                          non_linear=activation_function,
                          causal=causal,
                          cosIPD=True,
                          input_features=input_features,
                          spk_fea_dim=speaker_feature_dim, )
        # import pdb; pdb.set_trace()
        self.model_name = ckpt
        cpt_fname = os.path.join(save_model_dir, self.model_name)
        cpt = torch.load(cpt_fname, map_location="cpu")
        nnet.load_state_dict(cpt["model_state_dict"])
        logger.info("Load checkpoint from {}, epoch {:d}".format(
            cpt_fname, cpt["epoch"]))
        self.device = torch.device(
            "cuda:{}".format(gpuid)) if gpuid >= 0 else torch.device("cpu")
        self.nnet = nnet.to(self.device) if gpuid >= 0 else nnet
        self.nnet.eval()

    def compute(self, samples, directions, spk_num, lip_video):
        with torch.no_grad():
            raw = torch.tensor(samples, dtype=torch.float32, device=self.device)
            doa = torch.tensor(directions, dtype=torch.float32, device=self.device)
            spk_num = torch.tensor(spk_num, dtype=torch.float32, device=self.device)
            lip_video = torch.tensor(lip_video, dtype=torch.float32, device=self.device)
            # import pdb; pdb.set_trace()
            separate_samples, _, _ = self.nnet([raw, doa, spk_num, lip_video, None])  # should have shape [#, NSPK, S]
            # separate_samples, _ = self.nnet([raw, doa, spk_num, lip_video, None])  # should have shape [#, NSPK, S]
            # import pdb; pdb.set_trace()
            separate_samples = [np.squeeze(s.detach().cpu().numpy()) for s in separate_samples]
            return separate_samples


def run(eval_type):
    dataset = CHDataset(stage=eval_type, inference_job_num=int(sys.argv[1]), job=int(sys.argv[2]))
    computer = NNetComputer() 
    pc = ra.MultiChannelPerformance(name=computer.model_name)

    # unvalid_file_name = []
    # wav_scp_dct = {}
    # wav_dir = ''
    # abs_path = os.getcwd()
    for item_idx in range(len(dataset.wav_list)):
        # print("this is the {}-th wav".format(item_idx))
        wav_file = list(dataset.wav_list)[item_idx]
        # mix_wav, s1, lip_video, wav_file, ad, spk_rt60 = dataset.get_data(wav_file)
        mix_wav, ref, directions, spk_num, lip_video, ad, wav_name, spk_rt60 = dataset.get_data(wav_file)
         # import pdb; pdb.set_trace()
        clean_ref = ref[0]
        s1 = ref[1] # rvb
      
        separate_sample = computer.compute(mix_wav, directions, spk_num, lip_video)
        # wav = computer.compute(mix_wav, lip_video, s1)
        # wav = [mix_wav[0]]

        norm = np.linalg.norm(mix_wav[0], np.inf)
        est_s1 = separate_sample[0]
        # import pdb; pdb.set_trace()
        est_s1 = est_s1 * norm / np.max(np.abs(est_s1))
        
        # est_s1 = mix_wav[0]
        
        min_len = min(len(clean_ref), len(est_s1))
        clean_ref = clean_ref[:min_len]
        est_s1 = est_s1[:min_len]
        
        snr1 = ra.get_SI_SNR(est_s1, clean_ref)
        pesq = ra.get_PESQ(est_s1, clean_ref)
        stoi = ra.get_STOI(est_s1, clean_ref)
        srmr = 0
        if all_metrics:
            srmr = ra.get_SRMR(est_s1, clean_ref)
        
        logger.info("{}:{}/nspk:{},ad:{} snr1: {:.2f}, pesq: {:.2f}, stoi: {:.3f}, srmr: {:.2f}".format(wav_file, item_idx+1, spk_num, ad, snr1, pesq, stoi, srmr))
        # print("{}:{}/nspk:{},ad:{} snr1: {:.2f}, pesq: {:.2f}, stoi: {:.3f}, srmr: {:.2f}".format(wav_file, item_idx+1, spk_num, ad, snr1, pesq, stoi, srmr))

        pc.update_performance(wav_file, snr1, ad, spk_num, spk_rt60,  metric='SISNR')
        pc.update_performance(wav_file, pesq, ad, spk_num, spk_rt60, metric='PESQ')
        pc.update_performance(wav_file, stoi, ad, spk_num, spk_rt60, metric='STOI')
        if all_metrics:
            pc.update_performance(wav_file, srmr, ad, spk_num, spk_rt60,  metric='SRMR')

        if write_wav:
            # write_name = f'{model_save_dir}/{wav_dir_name}'
            # if not os.path.exists(write_name):
            #     os.makedirs(write_name)
          
            audiowrite(est_s1,f"{write_name}/{wav_file}.wav", sampling_rate)

        
        # if item_idx >= 100:
        #     break

    # logger.info("Compute over {:d} utterances".format(item_idx+1))
    pc.summary()
    print(f"model_type-{model_type}-diag_loading_ratio-{diag_loading_ratio}")
    
    
            
        

if __name__ == "__main__":

    run(inference_dataset_name)