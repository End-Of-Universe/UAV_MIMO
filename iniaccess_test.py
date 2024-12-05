import torch
import h5py
import numpy as np
import time

from params import params
from network import Selector_1
from access_and_track import initial_access_1, initial_access_2, initial_access_3, initial_access_4
from dft_codebook import Dft_codebook
from generate_dataset import random_split

## 对比穷搜法和数据驱动的方法
def compare():
    env_h5 = h5py.File(params["env_save_pth"], 'r')

    bs_y_num, bs_z_num, ue_y_num, ue_z_num = params["bs_y_antenna_num"], params["bs_z_antenna_num"], \
                                                params["ue_y_antenna_num"], params["ue_z_antenna_num"]
    
    _, _, test_idx = random_split(size = params["ue_num"],
                                  rate = (0.7, 0.15, 0.15),
                                  save_path = params["dataset_idx_pth"] 
                                  )
    test_idx = np.sort(test_idx)

    h_ini = env_h5["channel"][test_idx, :, :, 0]
    delta_pos_ini = env_h5["delta_pos"][test_idx, :, 0]
    ue_rot_ini = env_h5["ue_rot"][test_idx, :, 0]
    env_h5.close()

    narrow_codebook = Dft_codebook(bs_z_num, bs_y_num, ue_z_num, ue_y_num)
    wide_codebook = Dft_codebook(params["bs_z_wide"], params["bs_y_wide"], params["ue_z_wide"], params["ue_y_wide"])

    selector = Selector_1()
    selector.load(params["selector_1_params_pth"])

    test_set_size = test_idx.shape[0]
    for user in range(test_set_size):
        h = h_ini[user]
        delta_pos = delta_pos_ini[user]
        ue_rot = ue_rot_ini[user]

        way1_start = time.time()
        beam_bs_1, beam_ue_1, psd_opt_1 = initial_access_1(h, narrow_codebook)
        way1_end = time.time()

        way2_start = time.time()
        beam_bs_2, beam_ue_2, psd_opt_2 = initial_access_2(h, narrow_codebook, wide_codebook)
        way2_end = time.time()

        way3_start = time.time()
        beam_bs_3, beam_ue_3, psd_opt_3 = initial_access_3(h, delta_pos, ue_rot, narrow_codebook, 1, 5)
        way3_end = time.time()

        way4_start = time.time()
        beam_bs_4, beam_ue_4, psd_opt_4 = initial_access_4(h, delta_pos, ue_rot, narrow_codebook, selector, 5)
        way4_end = time.time()

        way1_spend = (way1_end - way1_start) * 1000
        way2_spend = (way2_end - way2_start) * 1000
        way3_spend = (way3_end - way3_start) * 1000
        way4_spend = (way4_end - way4_start) * 1000

        print("--------------------- %d/%d  ---------------------" % (user + 1, test_set_size))
        print("RSRP   way1: %5ddB, way2: %5ddB, way3: %5ddB, way4: %5ddB" % (psd_opt_1, psd_opt_2, psd_opt_3, psd_opt_4))
        print("TIME   way1: %5dms, way2: %5dms, way3: %5dms, way4: %5dms" % (way1_spend, way2_spend, way3_spend, way4_spend))


if __name__ == "__main__":
    ave_delta_psd = compare()
    print("ave_delta_psd: %.4f dB" % (ave_delta_psd))