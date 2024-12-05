from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import h5py
import os
import json

with open("./config.json", 'r') as file:
    config = json.load(file)

class Dataset_UAV_iniAccess(Dataset):
    def __init__(self, 
                 delta_pos0,
                 ue_rot0,
                 bs_label,
                 ue_label
                 ):
        super().__init__()
        self.delta_pos0 = delta_pos0
        self.ue_rot0 = ue_rot0
        self.bs_label = bs_label
        self.ue_label = ue_label
    
    def __len__(self):
        return self.delta_pos0.shape[0]
    
    def __getitem__(self, index):
        return self.delta_pos0[index, :], self.ue_rot0[index, :], self.bs_label[index], self.ue_label[index]

class Dataset_UAV_track(Dataset):
    def __init__(self, 
                 delta_pos,         # (sample_num, window_size, 3)
                 ue_rot,            # (sample_num, window_size, 3)
                 SNR_input,         # (sample_num, window_size)
                 SNR_output,        # (sample_num, )
                 SNR_base           # (sample_num)
                 ):
        super().__init__()
        self.delta_pos = delta_pos
        self.ue_rot = ue_rot
        self.SNR_input = SNR_input
        self.SNR_output = SNR_output
        self.SNR_base = SNR_base
    
    def __len__(self):
        return self.delta_pos.shape[0]
    
    def __getitem__(self, index):
        return self.delta_pos[index], self.ue_rot[index], self.SNR_input[index], self.SNR_output[index], self.SNR_base[index]


def generate_track_loaders(rate = (0.7, 0.3)):
    dataset_pth = config["Dataset_Config"]["dataset_pth"] % (config["Dataset_Config"]["ue_num"], 
                                                            config["Dataset_Config"]["track_num"])
    track_dataset_h5 = h5py.File(dataset_pth, 'r')
    
    info = track_dataset_h5["Info"]
    # window_size = info["window_input"][()]
    ue_num = info["ue_num"][()]
    
    SNR_input_window_train = []
    SNR_input_window_valid = []
    SNR_output_window_train = []
    SNR_output_window_valid = []
    ue_rot_window_train = []
    ue_rot_window_valid = []
    delta_pos_window_train = []
    delta_pos_window_valid = []

    sample_num_all = 0
    for user in range(ue_num):
        group_user = track_dataset_h5["Data_user_%d" % user]
        sample_num = group_user["sample_num"][()]
        sample_num_all += sample_num
        div = int(sample_num * rate[0])
        SNR_input_window_train.append(group_user["SNR_input_window"][:][0:div])
        SNR_input_window_valid.append(group_user["SNR_input_window"][:][div:])
        SNR_output_window_train.append(group_user["SNR_output_window"][:][0:div])
        SNR_output_window_valid.append(group_user["SNR_output_window"][:][div:])
        delta_pos_window_train.append(group_user["delta_pos_window"][:][0:div])
        delta_pos_window_valid.append(group_user["delta_pos_window"][:][div:])
        ue_rot_window_train.append(group_user["ue_rot_window"][:][0:div])
        ue_rot_window_valid.append(group_user["ue_rot_window"][:][div:])
    
    track_dataset_h5.close()

    # delta_pos_window:  (sample_num, window_size, 3)  (radian)
    # ue_rot_window:     (sample_num, window_size, 3)  (m)
    # SNR_input_window:  (sample_num, window_size)     (dB)
    # SNR_output_window: (sample_num)                  (dB)

    SNR_input_window_train = np.concatenate(SNR_input_window_train, axis = 0)
    SNR_input_window_valid = np.concatenate(SNR_input_window_valid, axis = 0)
    SNR_output_window_train = np.concatenate(SNR_output_window_train, axis = 0)
    SNR_output_window_valid = np.concatenate(SNR_output_window_valid, axis = 0)
    delta_pos_window_train = np.concatenate(delta_pos_window_train, axis = 0)
    delta_pos_window_valid = np.concatenate(delta_pos_window_valid, axis = 0)
    ue_rot_window_train = np.concatenate(ue_rot_window_train, axis = 0)
    ue_rot_window_valid = np.concatenate(ue_rot_window_valid, axis = 0)
    
    print("The size of UAV Dataset is %d" % sample_num_all)

    ## 数据预处理 -- 标准化
    SNR_base_train = np.copy(SNR_input_window_train[:, 0])
    SNR_base_valid = np.copy(SNR_input_window_valid[:, 0])
    
    SNR_input_window_train = SNR_input_window_train / SNR_base_train[:, None] - 1
    SNR_input_window_valid = SNR_input_window_valid / SNR_base_valid[:, None] - 1
    
    SNR_output_window_train = SNR_output_window_train / SNR_base_train[:, None] - 1
    SNR_output_window_valid = SNR_output_window_valid / SNR_base_valid[:, None] - 1
    
    delta_pos_window_train = delta_pos_window_train / delta_pos_window_train[:, 0, :][:, None, :] - 1
    delta_pos_window_valid = delta_pos_window_valid / delta_pos_window_valid[:, 0, :][:, None, :] - 1

    ue_rot_window_train = ue_rot_window_train / ue_rot_window_train[:, 0, :][:, None, :] - 1
    ue_rot_window_valid = ue_rot_window_valid / ue_rot_window_valid[:, 0, :][:, None, :] - 1
    
    train_set = Dataset_UAV_track(delta_pos_window_train, ue_rot_window_train, SNR_input_window_train, SNR_output_window_train, SNR_base_train)
    valid_set = Dataset_UAV_track(delta_pos_window_valid, ue_rot_window_valid, SNR_input_window_valid, SNR_output_window_valid, SNR_base_valid)
    
    train_loader = DataLoader(train_set, shuffle=True, batch_size=config["Network_Config"]["batch_size"], drop_last=True)
    valid_loader = DataLoader(valid_set, shuffle=False, batch_size=config["Network_Config"]["batch_size"], drop_last=True)
    
    
    return train_loader, valid_loader

def generate_window_data():
    dataset_pth = config["Dataset_Config"]["dataset_pth"] % (config["Dataset_Config"]["ue_num"], 
                                                             config["Dataset_Config"]["track_num"])
    if not os.path.exists(dataset_pth):
        print("UAV dataset doesn't exist!")
        return 
    
    track_dataset_h5 = h5py.File(dataset_pth, "r+")
    info = track_dataset_h5["Info"]
    
    ue_num = info["ue_num"][()]
    times_pilot = info["times_pilot"][()]

    window_input = config["Dataset_Config"]["window_input"]
    window_output = config["Dataset_Config"]["window_output"]
    window = window_input + window_output
    
    if "window_input" in info:
        del info["window_input"]
    info.create_dataset("window_input", data = window_input)

    if "window_output" in info:
        del info["window_output"]
    info.create_dataset("window_output", data = window_output)
    
    delta_pos_window_list = [[] for _ in range(ue_num)]         ## 相对位置滑窗数据集       (window_size, 3)
    ue_rot_window_list = [[] for _ in range(ue_num)]            ## UAV姿态角滑窗数据集      (window_size, 3)
    SNR_input_window_list = [[] for _ in range(ue_num)]         ## SNR(dB)输入滑窗数据集    (window_input, )
    SNR_output_window_list = [[] for _ in range(ue_num)]        ## SNR(dB)输出滑窗数据集    (window_output, )

    for user in range(ue_num):
        group_user = track_dataset_h5["Data_user_%d" % user]
        SNR_u = group_user["SNR"][:]
        stage_u = group_user["stage"][:]
        delta_pos = group_user["delta_pos"][:]
        ue_rot = group_user["ue_rot"][:]
        count = 1
        for t in range(1, times_pilot):
            if stage_u[t] != stage_u[t - 1]:
                count = 1
            else:
                if count >= window - 1:
                    start = t - window + 1
                    div = t - window_output + 1
                    end = t + 1
                    SNR_input_window_list[user].append(SNR_u[start : div])
                    SNR_output_window_list[user].append(SNR_u[div : end])
                    delta_pos_window_list[user].append(delta_pos[start : div])
                    ue_rot_window_list[user].append(ue_rot[start : div])
                count += 1
        if "delta_pos_window" in group_user:
            del group_user["delta_pos_window"]
        if "ue_rot_window" in group_user:
            del group_user["ue_rot_window"]
        if "SNR_input_window" in group_user:
            del group_user["SNR_input_window"]
        if "SNR_output_window" in group_user:
            del group_user["SNR_output_window"]
        if "sample_num" in group_user:
            del group_user["sample_num"]

        group_user.create_dataset("SNR_input_window", data = np.stack(SNR_input_window_list[user], axis = 0))
        group_user.create_dataset("SNR_output_window", data = np.stack(SNR_output_window_list[user], axis = 0))
        group_user.create_dataset("ue_rot_window", data = np.stack(ue_rot_window_list[user], axis = 0))
        group_user.create_dataset("delta_pos_window", data = np.stack(delta_pos_window_list[user], axis = 0))
        group_user.create_dataset("sample_num", data = len(SNR_input_window_list[user]))
    track_dataset_h5.close()
    print("Window Dataset has been generated!")
    

if __name__ == "__main__":
    generate_window_data()

    # track_dataset_h5 = h5py.File(config["UAV_Scenario_Config"]["tracking_dataset_pth"], 'r')
    # SNR_input_window = track_dataset_h5["SNR_input_window"][:]
    # SNR_output_window = track_dataset_h5["SNR_output_window"][:]
    # ue_rot_window = track_dataset_h5["ue_rot_window"][:]
    # track_dataset_h5.close()

    train_loader, valid_loader = generate_track_loaders()

    # print(len(train_loader) * config["UAV_Scenario_Config"]["batch_size"])
    for (delta_pos_batch, ue_rot_batch, SNR_input_batch, SNR_output_batch, SNR_base_batch) in train_loader:
        print(delta_pos_batch.shape)
        print(ue_rot_batch.shape)
        print(SNR_input_batch.shape)
        print(SNR_output_batch.shape)
        print(SNR_base_batch.shape)
        break
