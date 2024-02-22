
import pickle
import numpy as np

with open(r'E:\zzx\final_clisa\thu_ep1\runs_srt\rawdata_24video_batch24_timeLen5_tf16_sf16_multiFact2_lr0.000700_wd0.015000_epochs100_randSeed7_fold10_accSel_newPre\wd0.008\results_finetune.pkl', 'rb') as f:
    try:
        results_pretrain = pickle.load(f)
    except EOFError:
        pass