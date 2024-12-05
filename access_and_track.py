import numpy as np
import h5py
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import json
import utils
from dft_codebook import Dft_codebook

itype = np.int32
ftype = np.float32
ctype = np.complex64

with open("./config.json", 'r') as file:
    config = json.load(file)

def track(h, ue_idx, bs_idx):
    '''
    1. since UE performs beam tracking first, thus the estimation accuracy of UE 
    is worse than that of BS
    
    2. enlarge the neighborhood search range r, the estimation accuracy can be improved.
    '''
    r = config["UAV_Scenario_Config"]["track_r"]
    
    # threshold = utils.calculate_overall_noise() - 10 # SNR >= -10 dB 

    bs_y_narrow, bs_x_narrow, ue_y_narrow, ue_x_narrow = config["UAV_Scenario_Config"]["bs_y_antenna_num"], config["UAV_Scenario_Config"]["bs_z_antenna_num"], \
        config["UAV_Scenario_Config"]["ue_y_antenna_num"], config["UAV_Scenario_Config"]["ue_z_antenna_num"]
    
    narrow_codebook = Dft_codebook(bs_y_narrow, bs_x_narrow, ue_y_narrow, ue_x_narrow)
    narrow_codebook.generate_dft_codebook()

    tic_num = h.shape[-1]

    est_BS_y = np.zeros((tic_num))
    est_BS_z = np.zeros((tic_num))
    est_UE_y = np.zeros((tic_num))
    est_UE_z = np.zeros((tic_num))

    for tic in range(tic_num):
        h_ = h[:, :, tic]
        
        # 1. bs transmits the last narrow beam, ue sweeps all narrow beams around the last UE narrow beam 
        ue_idx_z, ue_idx_y = np.unravel_index(ue_idx, [config["UAV_Scenario_Config"]["ue_z_antenna_num"], config["UAV_Scenario_Config"]["ue_y_antenna_num"]])

        hr = np.matmul(h_, narrow_codebook.bs_dft_codebook[:, bs_idx])
        br = np.matmul(np.conjugate(narrow_codebook.ue_dft_codebook).T, hr)

        psd_r = utils.convert_to_L1RSRP(np.abs(br), config["UAV_Scenario_Config"]["ue_tx_power_max"])
        psd_r = np.reshape(psd_r, [config["UAV_Scenario_Config"]["ue_z_antenna_num"], config["UAV_Scenario_Config"]["ue_y_antenna_num"]])

        ue_idx_z_l = np.maximum(ue_idx_z - r, 0)
        ue_idx_z_h = np.minimum(ue_idx_z + r, config["UAV_Scenario_Config"]["ue_z_antenna_num"])
        ue_idx_y_l = np.maximum(ue_idx_y - r, 0)
        ue_idx_y_h = np.minimum(ue_idx_y + r, config["UAV_Scenario_Config"]["ue_y_antenna_num"])
        psd_r_local = psd_r[ue_idx_z_l: ue_idx_z_h, ue_idx_y_l: ue_idx_y_h]
      
        max_index = np.unravel_index(np.argmax(psd_r_local), psd_r_local.shape)

        ue_idx_z = ue_idx_z_l + max_index[0]
        ue_idx_y = ue_idx_y_l + max_index[1]
        est_UE_y[tic] = ue_idx_y
        est_UE_z[tic] = ue_idx_z
        print("track BS -> UE stage: %d, UE beam: y_idx-%d, z_idx-%d" % (tic + 1, ue_idx_y, ue_idx_z))

        ue_idx = ue_idx_z * config["UAV_Scenario_Config"]["ue_y_antenna_num"] + ue_idx_y
        
        # 2. ue transmits the current narrow beam, bs sweeps all narrow beams around the last BS narrow beam 
        bs_idx_z, bs_idx_y = np.unravel_index(bs_idx, [config["UAV_Scenario_Config"]["bs_z_antenna_num"], config["UAV_Scenario_Config"]["bs_y_antenna_num"]])
        ht = np.matmul(np.conjugate(narrow_codebook.ue_dft_codebook[:, ue_idx]).T, h_)
        bt = np.matmul(ht, narrow_codebook.bs_dft_codebook) 
  
        psd_t = utils.convert_to_L1RSRP(np.abs(bt), config["UAV_Scenario_Config"]["bs_tx_power_max"])
        psd_t = np.reshape(psd_t, [config["UAV_Scenario_Config"]["bs_z_antenna_num"], config["UAV_Scenario_Config"]["bs_y_antenna_num"]])
        bs_idx_z_l = np.maximum(bs_idx_z - r, 0)
        bs_idx_z_h = np.minimum(bs_idx_z + r, config["UAV_Scenario_Config"]["bs_z_antenna_num"])
        bs_idx_y_l = np.maximum(bs_idx_y - r, 0)
        bs_idx_y_h = np.minimum(bs_idx_y + r, config["UAV_Scenario_Config"]["bs_y_antenna_num"])
        psd_t_local = psd_t[bs_idx_z_l: bs_idx_z_h, bs_idx_y_l: bs_idx_y_h]
        
        max_index = np.unravel_index(np.argmax(psd_t_local), psd_t_local.shape)

        bs_idx_z = bs_idx_z_l + max_index[0]
        bs_idx_y = bs_idx_y_l + max_index[1]
        est_BS_y[tic] = bs_idx_y
        est_BS_z[tic] = bs_idx_z

        print("track UE -> BS stage: %d, BS beam: y_idx-%d, z_idx-%d" % (tic + 1, bs_idx_y, bs_idx_z))
        bs_idx = bs_idx_z * config["UAV_Scenario_Config"]["bs_y_antenna_num"] + bs_idx_y

    return est_BS_y, est_BS_z, est_UE_y, est_UE_z

## 遍历所有波束组合，选出最优波束对
def initial_access_1(h,
                     codebook
                     ):
    thresh_hold = utils.calculate_overall_noise() - 10
    bs_y_num, bs_z_num, ue_y_num, ue_z_num = config["UAV_Scenario_Config"]["bs_y_antenna_num"], config["UAV_Scenario_Config"]["bs_z_antenna_num"], \
        config["UAV_Scenario_Config"]["ue_y_antenna_num"], config["UAV_Scenario_Config"]["ue_z_antenna_num"]

    psd_opt = -1e10
    idx_bs_1d_opt = None
    idx_ue_1d_opt = None

    for idx_bs in range(h.shape[0]):
        hr = np.matmul(h, codebook.bs_dft_codebook[:, idx_bs])
        br = np.matmul(np.conjugate(codebook.ue_dft_codebook).T, hr)
        psd_r = utils.convert_to_L1RSRP(np.abs(br), config["UAV_Scenario_Config"]["ue_tx_power_max"])

        idx_ue = np.argmax(psd_r)
        psd_max = psd_r[idx_ue]

        if psd_max > psd_opt:
            psd_opt = psd_max
            idx_bs_1d_opt = idx_bs
            idx_ue_1d_opt = idx_ue

    return idx_bs_1d_opt, idx_ue_1d_opt, psd_opt

# 先找到一个可通信的波束，然后宽波束搜索，再窄波束遍历
def initial_access_2(h, 
                     narrow_codebook,
                     wide_codebook
                     ):
    threshold = utils.calculate_overall_noise() - 10            # SNR >= -10 dB 

    bs_y_narrow, bs_z_narrow, ue_y_narrow, ue_z_narrow = config["UAV_Scenario_Config"]["bs_y_antenna_num"], config["UAV_Scenario_Config"]["bs_z_antenna_num"], \
        config["UAV_Scenario_Config"]["ue_y_antenna_num"], config["UAV_Scenario_Config"]["ue_z_antenna_num"]
    
    bs_y_wide, bs_z_wide, ue_y_wide, ue_z_wide = config["UAV_Scenario_Config"]["bs_y_wide"], config["UAV_Scenario_Config"]["bs_z_wide"], \
        config["UAV_Scenario_Config"]["ue_y_wide"], config["UAV_Scenario_Config"]["ue_z_wide"]
    
    h_ = h.reshape(ue_y_narrow, ue_z_narrow, bs_y_narrow, bs_z_narrow)

    ## 使用左下角的天线阵进行宽波束传输
    h_w = h_[:ue_y_wide, :ue_z_wide, :bs_y_wide, :bs_z_wide]
    h_w = h_w.reshape(ue_y_wide * ue_z_wide, bs_y_wide * bs_z_wide)

    # 1. bs transmits one wide beam per time, ue sweeps all wide beams, 
    #    the result is optimal BS-UE beam pair   
    idx_mid = wide_codebook.bs_dft_codebook.shape[1] // 2
    idx_set = list()
    ## 越靠近中间天线增益越高，所以从中间开始搜
    for i in range(idx_mid):
        idx_set.append(idx_mid - i - 1)
        idx_set.append(idx_mid + i)
        
    idx_bs_opt = -1e10
    for idx in idx_set: 
        hr = np.matmul(h_w, wide_codebook.bs_dft_codebook[:, idx])
        br = np.matmul(np.conjugate(wide_codebook.ue_dft_codebook).T, hr)
        psd_r = utils.convert_to_L1RSRP(np.abs(br), config["UAV_Scenario_Config"]["ue_tx_power_max"])
        index_wide_r = np.argmax(psd_r)
        # if np.max(psd_r) >= threshold:
        if psd_r[index_wide_r] > idx_bs_opt:
            psd_r = np.reshape(psd_r, [ue_y_wide, ue_z_wide])
            idx_ue_opt = index_wide_r       ## 宽波束索引
            ue_wide_max_index = np.unravel_index(idx_ue_opt, [ue_y_wide, ue_z_wide])
            # print(ue_wide_max_index)

            idx_bs_opt = idx                    ## 宽波束索引
            bs_wide_max_index = np.unravel_index(idx_bs_opt, [bs_y_wide, bs_z_wide])
            # print("successfully initial access!")
            break

    # if idx_bs_opt == -1e10:  
    #     idx_ue_opt = idx_set[0]
    #     idx_bs_opt = idx_set[0]
    #     ue_wide_max_index = np.unravel_index(idx_ue_opt, [ue_y_wide, ue_z_wide])
    #     bs_wide_max_index = np.unravel_index(idx_bs_opt, [bs_y_wide, bs_z_wide])
    #     print('fail to initial access!')
    #     return (None for _ in range(3))
        
    ue_z_times = ue_z_narrow // ue_z_wide
    ue_y_times = ue_y_narrow // ue_y_wide
    bs_z_times = bs_z_narrow // bs_z_wide
    bs_y_times = bs_y_narrow // bs_y_wide

    # 2. bs transmits the optimal wide beam, ue sweeps all narrow beams 
    h_nw = h_[:ue_y_narrow, :ue_z_narrow, :bs_y_wide, :bs_z_wide]
    h_nw = h_nw.reshape(ue_y_narrow * ue_z_narrow, bs_y_wide * bs_z_wide)

    hr = np.matmul(h_nw, wide_codebook.bs_dft_codebook[:, idx_bs_opt])
    br = np.matmul(np.conjugate(narrow_codebook.ue_dft_codebook).T, hr)

    psd_r = utils.convert_to_L1RSRP(np.abs(br), config["UAV_Scenario_Config"]["ue_tx_power_max"])
    psd_r = np.reshape(psd_r, [ue_y_narrow, ue_z_narrow])
    
    psd_r_local = psd_r[ue_wide_max_index[0] * ue_y_times: (ue_wide_max_index[0] + 1) * ue_y_times,
                        ue_wide_max_index[1] * ue_z_times: (ue_wide_max_index[1] + 1) * ue_z_times]
    ue_narrow_max_index = np.unravel_index(np.argmax(psd_r_local), psd_r_local.shape)
    
    ue_narrow_max_index_y = ue_narrow_max_index[0] + ue_wide_max_index[0] * ue_y_times
    ue_narrow_max_index_z = ue_narrow_max_index[1] + ue_wide_max_index[1] * ue_z_times
    
    ue_idx = ue_narrow_max_index_y * ue_z_narrow + ue_narrow_max_index_z
   
    # 3. ue transmits the optimal narrow beam, bs sweeps all narrow beams 
    h_n = h_[:ue_y_narrow, :ue_z_narrow, :bs_y_narrow, :bs_z_narrow]
    h_n = h_n.reshape(ue_y_narrow * ue_z_narrow, bs_y_narrow * bs_z_narrow)

    ht = np.matmul(np.conjugate(narrow_codebook.ue_dft_codebook[:, ue_idx]).T, h_n)
    bt = np.matmul(ht, narrow_codebook.bs_dft_codebook)

    psd_t = utils.convert_to_L1RSRP(np.abs(bt), config["UAV_Scenario_Config"]["bs_tx_power_max"])
    psd_t = np.reshape(psd_t, [bs_y_narrow, bs_z_narrow])
    
    psd_t_local = psd_t[bs_wide_max_index[0] * bs_y_times: (bs_wide_max_index[0] + 1) * bs_y_times,
                        bs_wide_max_index[1] * bs_z_times: (bs_wide_max_index[1] + 1) * bs_z_times]
    bs_narrow_max_index = np.unravel_index(np.argmax(psd_t_local), psd_t_local.shape)
    
    psd_opt = psd_t_local[bs_narrow_max_index[0], bs_narrow_max_index[1]]

    bs_narrow_max_index_y = bs_narrow_max_index[0] + bs_wide_max_index[0] * bs_y_times
    bs_narrow_max_index_z = bs_narrow_max_index[1] + bs_wide_max_index[1] * bs_z_times
    
    bs_idx = bs_narrow_max_index_y * bs_z_narrow + bs_narrow_max_index_z
    
    return bs_idx, ue_idx, psd_opt

## 根据LOS径的方向角估算BS和UE端的波束搜索中心，然后围绕中心以半径search_r进行搜索
def initial_access_3(h,             ## 信道矩阵 (Nt * Nr, Nt * Nr)
                     delta_pos,     ## 相对位置 (3, )
                     ue_rot,        ## UAV姿态
                     codebook,      ## DFT码本
                     bs_search_r,   ## BS端检索半径
                     ue_search_r,   ## UE端检索半径
                     ):
    thresh_hold = utils.calculate_overall_noise() - 10

    theta_zod_los, phi_aod_los, theta_zoa_los, phi_aoa_los = utils.calculate_los_angle(delta_pos, ue_rot)
    bs_Nz, bs_Ny, ue_Nz, ue_Ny = config["UAV_Scenario_Config"]["bs_z_antenna_num"], config["UAV_Scenario_Config"]["bs_y_antenna_num"], config["UAV_Scenario_Config"]["ue_z_antenna_num"], config["UAV_Scenario_Config"]["ue_y_antenna_num"]
    z_center_bs, y_center_bs = utils.calculate_upa_index(theta_zod_los, phi_aod_los, bs_Nz, bs_Ny)
    z_center_ue, y_center_ue = utils.calculate_upa_index(theta_zoa_los, phi_aoa_los, ue_Nz, ue_Ny)
    
    bs_idx_set = list()     ## BS端搜索的波束索引集合
    ue_idx_set = list()     ## UE端搜索的波束索引集合
    ## 生成BS端所有用于遍历的波束
    z_bs_start = max(0, z_center_bs - bs_search_r)
    z_bs_end = min(bs_Nz - 1, z_center_bs + bs_search_r + 1)
    y_bs_start = max(0, y_center_bs - bs_search_r)
    y_bs_end = min(bs_Ny - 1, y_center_bs + bs_search_r + 1)
    for z_bs in range(z_bs_start, z_bs_end):
        for y_bs in range(y_bs_start, y_bs_end):
            bs_idx_set.append(z_bs + y_bs * bs_Nz)
    ## 生成UE端所有用于遍历的波束
    z_ue_start = max(0, z_center_ue - ue_search_r)
    z_ue_end = min(ue_Nz - 1, z_center_ue + ue_search_r + 1)
    y_ue_start = max(0, y_center_ue - ue_search_r)
    y_ue_end = min(ue_Ny - 1, y_center_ue + ue_search_r + 1)
    for z_ue in range(z_ue_start, z_ue_end):
        for y_ue in range(y_ue_start, y_ue_end):
            ue_idx_set.append(z_ue + y_ue * ue_Nz)

    psd_opt = -1e10
    idx_bs_1d_opt = None
    idx_ue_1d_opt = None

    for idx_bs in bs_idx_set:
        hr = np.matmul(h, codebook.bs_dft_codebook[:, idx_bs])
        br = np.matmul(np.conjugate(codebook.ue_dft_codebook[:, ue_idx_set]).T, hr)
        psd_r = utils.convert_to_L1RSRP(np.abs(br), config["UAV_Scenario_Config"]["ue_tx_power_max"])
        
        temp_index = np.argmax(psd_r)
        idx_ue = ue_idx_set[temp_index]
        psd_max = psd_r[temp_index]

        if psd_max > psd_opt:
            psd_opt = psd_max
            idx_bs_1d_opt = idx_bs
            idx_ue_1d_opt = idx_ue

    return idx_bs_1d_opt, idx_ue_1d_opt, psd_opt


def initial_access_4(h, 
                     delta_pos, 
                     ue_rot, 
                     codebook, 
                     selector, 
                     k = 5
                     ):
    thresh_hold = utils.calculate_overall_noise()

    bs_Nz, bs_Ny, ue_Nz, ue_Ny = config["UAV_Scenario_Config"]["bs_z_antenna_num"], config["UAV_Scenario_Config"]["bs_y_antenna_num"], config["UAV_Scenario_Config"]["ue_z_antenna_num"], config["UAV_Scenario_Config"]["ue_y_antenna_num"]
    Nt = bs_Nz * bs_Ny
    Nr = ue_Nz * ue_Ny

    delta_pos = torch.tensor(delta_pos.reshape(1, 3), dtype=torch.float32)       # 不在torch上运行
    ue_rot = torch.tensor(ue_rot.reshape(1, 3), dtype=torch.float32)             # 不在torch上运行

    selector.eval()

    bs_p, ue_p = selector(delta_pos, ue_rot)    # bs_p: (1, Nt), ue_p: (1, Nr)
    bs_p = bs_p.squeeze(0).detach().numpy()
    ue_p = ue_p.squeeze(0).detach().numpy()

    bs_beam_candidate = np.argsort(bs_p)[Nt - k : ]
    ue_beam_candidate = np.argsort(ue_p)[Nr - k : ]

    psd_opt = -1e10
    idx_bs_1d_opt = None
    idx_ue_1d_opt = None    

    for idx_bs in bs_beam_candidate:
        hr = np.matmul(h, codebook.bs_dft_codebook[:, idx_bs])
        br = np.matmul(np.conjugate(codebook.ue_dft_codebook[:, ue_beam_candidate]).T, hr)
        psd_r = utils.convert_to_L1RSRP(np.abs(br), config["UAV_Scenario_Config"]["ue_tx_power_max"])

        temp_index = np.argmax(psd_r)
        idx_ue = ue_beam_candidate[temp_index]
        psd_max = psd_r[temp_index]

        if psd_max > psd_opt:
            psd_opt = psd_max
            idx_bs_1d_opt = idx_bs
            idx_ue_1d_opt = idx_ue
    
    return idx_bs_1d_opt, idx_ue_1d_opt, psd_opt

if __name__ == "__main__":
    env_h5 = h5py.File(config["UAV_Scenario_Config"]["env_save_pth"], 'r')

    bs_y_num, bs_z_num, ue_y_num, ue_z_num = config["UAV_Scenario_Config"]["bs_y_antenna_num"], config["UAV_Scenario_Config"]["bs_z_antenna_num"], \
                                                config["UAV_Scenario_Config"]["ue_y_antenna_num"], config["UAV_Scenario_Config"]["ue_z_antenna_num"]
    user = 499
    h_ini = env_h5["channel"][user, :, :, 0]
    delta_pos_ini = env_h5["delta_pos"][user, :, 0]
    rot_ini = env_h5["ue_rot"][user, :, 0]
    env_h5.close()
 
    narrow_codebook = Dft_codebook(bs_z_num, bs_y_num, ue_z_num, ue_y_num)
    

    idx_bs_1d_opt_1, idx_ue_1d_opt_1, psd_opt_1 = initial_access_1(h_ini, narrow_codebook)

    wide_codebook = Dft_codebook(config["UAV_Scenario_Config"]["bs_z_wide"], config["UAV_Scenario_Config"]["bs_y_wide"], config["UAV_Scenario_Config"]["ue_z_wide"], config["UAV_Scenario_Config"]["ue_y_wide"])
    idx_bs_1d_opt_2, idx_ue_1d_opt_2, psd_opt_2 = initial_access_2(h_ini, 
                                                                   narrow_codebook, 
                                                                   wide_codebook
                                                                   )

    idx_bs_1d_opt_3, idx_ue_1d_opt_3, psd_opt_3 = initial_access_3(h_ini, 
                                                                   delta_pos_ini, 
                                                                   rot_ini, 
                                                                   narrow_codebook, 
                                                                   1, 4)
    
    from network import Selector_1
    selector = Selector_1()
    selector.load(config["UAV_Scenario_Config"]["model_pth"])

    idx_bs_1d_opt_4, idx_ue_1d_opt_4, psd_opt_4 = initial_access_4(h_ini, 
                                                                   delta_pos_ini, 
                                                                   rot_ini,
                                                                   narrow_codebook, 
                                                                   selector,
                                                                   5)



    print("way1: bs_opt: %d, ue_opt: %d, psd_opt: %.2f" % (idx_bs_1d_opt_1, idx_ue_1d_opt_1, psd_opt_1))

    print("way2: bs_opt: %d, ue_opt: %d, psd_opt: %.2f" % (idx_bs_1d_opt_2, idx_ue_1d_opt_2, psd_opt_2))

    print("way3: bs_opt: %d, ue_opt: %d, psd_opt: %.2f" % (idx_bs_1d_opt_3, idx_ue_1d_opt_3, psd_opt_3))

    print("way4: bs_opt: %d, ue_opt: %d, psd_opt: %.2f" % (idx_bs_1d_opt_4, idx_ue_1d_opt_4, psd_opt_4))


    # ## 显示真实的los角度
    # los_zod, los_aod, los_zoa, los_aoa = utils.calculate_los_angle(delta_pos_ini, rot_ini)
    # los_zod, los_aod, los_zoa, los_aoa = utils.radian_2_degree(los_zod, los_aod, los_zoa, los_aoa)
    # print("real-     aod: %.2f, zod: %.2f, aoa: %.2f, zoa: %.2f" % (los_aod, los_zod, los_aoa, los_zoa))

    # ## 显示way1最优波束的角度
    # est_zod, est_aod = utils.calculate_upa_angle(k = idx_bs_1d_opt_1, z_num = config["UAV_Scenario_Config"]["bs_z_antenna_num"], y_num = config["UAV_Scenario_Config"]["bs_y_antenna_num"])
    # est_zoa, est_aoa = utils.calculate_upa_angle(k = idx_ue_1d_opt_1, z_num = config["UAV_Scenario_Config"]["ue_z_antenna_num"], y_num = config["UAV_Scenario_Config"]["ue_y_antenna_num"])
    # est_zod, est_aod, est_zoa, est_aoa = utils.radian_2_degree(est_zod, est_aod, est_zoa, est_aoa)
    # print("way1 estimate- aod: %.2f, zod: %.2f, aoa: %.2f, zoa: %.2f" % (est_aod, est_zod, est_aoa, est_zoa))

    # ## 显示way3最优波束的角度
    # est_zod, est_aod = utils.calculate_upa_angle(k = idx_bs_1d_opt_3, z_num = config["UAV_Scenario_Config"]["bs_z_antenna_num"], y_num = config["UAV_Scenario_Config"]["bs_y_antenna_num"])
    # est_zoa, est_aoa = utils.calculate_upa_angle(k = idx_ue_1d_opt_3, z_num = config["UAV_Scenario_Config"]["ue_z_antenna_num"], y_num = config["UAV_Scenario_Config"]["ue_y_antenna_num"])
    # est_zod, est_aod, est_zoa, est_aoa = utils.radian_2_degree(est_zod, est_aod, est_zoa, est_aoa)
    # print("way3 estimate- aod: %.2f, zod: %.2f, aoa: %.2f, zoa: %.2f" % (est_aod, est_zod, est_aoa, est_zoa))

    

