import torch
import os
import json
import torch.nn.functional as F
from torch import nn
from generate_dataset import generate_track_loaders, generate_window_data
from network import model_dict, Track_Loss

os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with open("./config.json", 'r') as file:
    config = json.load(file)

# delta_SNR_hat:  (batch_size, window_output)
# delta_SNR_real: (batch_size, window_output)
# SNR_base:       (batch_size, )
def calculate_abs_MAE(delta_SNR_hat, 
                      delta_SNR_real, 
                      SNR_base
                      ):
    SNR_hat = (delta_SNR_hat + 1) * SNR_base[:, None]
    SNR_real = (delta_SNR_real + 1) * SNR_base[:, None]
    loss = torch.abs(SNR_hat - SNR_real)
    abs_mae = torch.mean(torch.sum(loss, dim = 1), dim = 0)
    return abs_mae

def calculate_abs_MSE(delta_SNR_hat, 
                      delta_SNR_real, 
                      SNR_base
                      ):
    SNR_hat = (delta_SNR_hat + 1) * SNR_base[:, None]
    SNR_real = (delta_SNR_real + 1) * SNR_base[:, None]
    loss = (SNR_hat - SNR_real) ** 2
    abs_mse = torch.mean(torch.sum(loss, dim = 1), dim = 0)
    return abs_mse

def calculate_acc(delta_SNR_hat, 
                  delta_SNR_real, 
                  SNR_base,
                  SNR_var
                  ):
    SNR_hat = (delta_SNR_hat + 1) * SNR_base[:, None]#.cpu().detach().numpy()
    SNR_real = (delta_SNR_real + 1) * SNR_base[:, None]#).cpu().detach().numpy()
    # SNR_var = SNR_var.cpu().detach().numpy()

    window_output = SNR_hat.shape[1]
    batch_size = SNR_hat.shape[0]
    count = batch_size

    down_line = SNR_hat - 1.96 * SNR_var
    up_line = SNR_hat + 1.96 * SNR_var
    
    # flag = (down_line > SNR_real) or (up_line < SNR_real)

    for u in range(batch_size):
        for t in range(window_output):
            if down_line[u, t] > SNR_real[u, t] or up_line[u, t] < SNR_real[u, t]:
                count -= 1
                break
    return count / batch_size

class Predictor:
    def __init__(self, 
                 model_type = "Linear",
                 train_phase = 1,
                 ):
        if train_phase == 1 or train_phase == 2:
            self.train_phase = train_phase
        else:
            raise RuntimeError("train_phase must be chosen from (1, 2)!")
        self.model_type = model_type
        self.predictor = model_dict[model_type]["predictor"].to(device)
        self.criterion = Track_Loss(train_phase = train_phase).to(device) #nn.L1Loss().to(device)
        self.optimizer = torch.optim.RMSprop(self.predictor.parameters(), lr=1e-3)
        
    def train_epoch(self, 
                    train_loader
                    ):
        
        self.predictor.train()
        
        loss_train = 0
        mae_train = 0
        mse_train = 0
        acc_train = 0
        for (delta_pos_batch, ue_rot_batch, SNR_input_batch, SNR_output_batch, SNR_base_batch) in train_loader:
            delta_pos_batch = delta_pos_batch.to(device)
            ue_rot_batch = ue_rot_batch.to(device)
            SNR_input_batch = SNR_input_batch.to(device)
            SNR_output_batch = SNR_output_batch.to(device)
            SNR_base_batch = SNR_base_batch.to(device)
            
            self.optimizer.zero_grad()
            
            mu_batch, var_batch = self.predictor(SNR_input_batch, delta_pos_batch, ue_rot_batch, self.train_phase)
            # mu_batch = self.predictor(SNR_input_batch)

            loss = self.criterion(mu_batch, var_batch, SNR_output_batch, SNR_base_batch)
            # loss = self.criterion(mu_batch, SNR_output_batch)
            
            loss.backward()

            self.optimizer.step()
            loss_train += loss.detach()
            mae_train += calculate_abs_MAE(mu_batch, SNR_output_batch, SNR_base_batch)
            mse_train += calculate_abs_MSE(mu_batch, SNR_output_batch, SNR_base_batch)
            acc_train += calculate_acc(mu_batch, SNR_output_batch, SNR_base_batch, var_batch)

        loss_train /= len(train_loader)
        mae_train /= len(train_loader)
        mse_train /= len(train_loader)
        acc_train /= len(train_loader)

        return loss_train, mae_train, mse_train, acc_train
    
    def valid_epoch(self,
                    valid_loader
                    ): 
        self.predictor.eval()
        
        loss_valid = 0
        mae_valid = 0
        mse_valid = 0
        acc_valid = 0
        for (delta_pos_batch, ue_rot_batch, SNR_input_batch, SNR_output_batch, SNR_base_batch) in valid_loader:
            delta_pos_batch = delta_pos_batch.to(device)
            ue_rot_batch = ue_rot_batch.to(device)
            SNR_input_batch = SNR_input_batch.to(device)
            SNR_output_batch = SNR_output_batch.to(device)
            SNR_base_batch = SNR_base_batch.to(device)
            
            mu_batch, var_batch = self.predictor(SNR_input_batch, delta_pos_batch, ue_rot_batch, self.train_phase)
            # mu_batch = self.predictor(SNR_input_batch)

            loss = self.criterion(mu_batch, var_batch, SNR_output_batch, SNR_base_batch)
            # loss = self.criterion(mu_batch, SNR_output_batch)
            
            loss_valid += loss.detach()
            mae_valid += calculate_abs_MAE(mu_batch, SNR_output_batch, SNR_base_batch)
            mse_valid += calculate_abs_MSE(mu_batch, SNR_output_batch, SNR_base_batch)
            acc_valid += calculate_acc(mu_batch, SNR_output_batch, SNR_base_batch, var_batch)

        loss_valid /= len(valid_loader)
        mae_valid /= len(valid_loader)
        mse_valid /= len(valid_loader)
        acc_valid /= len(valid_loader)
        return loss_valid, mae_valid, mse_valid, acc_valid
    
    def train(self, 
              num_epoch, 
              train_loader, 
              valid_loader, 
              reTrain = False
              ):
        torch.cuda.empty_cache()

        if not reTrain:
            self.predictor.load(train_phase = self.train_phase)

        min_loss_valid = 1e10
        train_mae_propotion = 0.7
        for i in range(num_epoch):
            loss_train, mae_train, mse_train, acc_train = self.train_epoch(train_loader)
            loss_valid, mae_valid, mse_valid, acc_valid = self.valid_epoch(valid_loader)

            print("epoch ------- %d/%d -------" % (i + 1, num_epoch))
            print("train --- loss: %.4f, MAE: %.4f, MSE: %.4f, acc: %.4f" % (loss_train, mae_train, mse_train, acc_train))
            print("valid --- loss: %.4f, MAE: %.4f, MSE: %.4f, acc: %.4f" % (loss_valid, mae_valid, mse_valid, acc_valid))
            if self.train_phase == 1:
                temp_value = mae_valid * (1 - train_mae_propotion) + mae_train * train_mae_propotion
                if temp_value < min_loss_valid:
                    min_loss_valid = temp_value
                    self.predictor.save(self.train_phase)
            else:
                temp_value = (loss_train + loss_valid) / 2
                if temp_value < min_loss_valid:
                    min_loss_valid = temp_value
                    self.predictor.save(self.train_phase)
    

if __name__ == "__main__":
    print("training device: %s" % str(device))
    generate_window_data()
    train_loader, valid_loader = generate_track_loaders(rate = (0.7, 0.3))
    predictor = Predictor(model_type = "Linear",
                          train_phase = 2)
    
    predictor.train(num_epoch = config["Network_Config"]["num_epoch"],
                    train_loader = train_loader, 
                    valid_loader = valid_loader, 
                    reTrain = config["Network_Config"]["reTrain"]
                    )
    # predictor.test(test_loader = test_loader)
    
