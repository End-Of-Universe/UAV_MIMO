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

## 通过差分归一化的SNR恢复成绝对SNR
## 并计算MAE
def calculate_abs_MAE(delta_SNR_hat, 
                      delta_SNR_real, 
                      SNR_base
                      ):
    SNR_hat = (delta_SNR_hat + 1) * SNR_base[:, None]
    SNR_real = (delta_SNR_real + 1) * SNR_base[:, None]
    loss = torch.abs(SNR_hat - SNR_real)
    abs_mae = torch.mean(torch.sum(loss, dim = 1), dim = 0)
    return abs_mae

## 通过差分归一化的SNR恢复成绝对SNR
## 并计算MSE
def calculate_abs_MSE(delta_SNR_hat, 
                      delta_SNR_real, 
                      SNR_base):
    SNR_hat = (delta_SNR_hat + 1) * SNR_base[:, None]
    SNR_real = (delta_SNR_real + 1) * SNR_base[:, None]
    loss = (SNR_hat - SNR_real) ** 2
    abs_mse = torch.mean(torch.sum(loss, dim = 1), dim = 0)
    return abs_mse

class Predictor:
    def __init__(self, 
                 model_type = "LSTM"
                 ):
        self.predictor = model_dict[model_type]["predictor"].to(device)
        self.criterion = Track_Loss(type = "MSE").to(device) #nn.L1Loss().to(device)
        self.optimizer = torch.optim.RMSprop(self.predictor.parameters(), lr=1e-3)
        self.model_path = model_dict[model_type]["model_pth"]
        
    def train_epoch(self, 
                    train_loader
                    ):
        self.predictor.train()
        
        loss_train = 0
        mae_train = 0
        mse_train = 0
        for (delta_pos_batch, ue_rot_batch, SNR_input_batch, SNR_output_batch, SNR_base_batch) in train_loader:
            delta_pos_batch = delta_pos_batch.to(device)
            ue_rot_batch = ue_rot_batch.to(device)
            SNR_input_batch = SNR_input_batch.to(device)
            SNR_output_batch = SNR_output_batch.to(device)
            SNR_base_batch = SNR_base_batch.to(device)
            
            self.optimizer.zero_grad()
            
            mu_batch, var_batch = self.predictor(SNR_input_batch, delta_pos_batch, ue_rot_batch)

            loss = self.criterion(mu_batch, var_batch, SNR_output_batch, SNR_base_batch)
            
            loss.backward()

            self.optimizer.step()
            loss_train += loss.detach()
            mae_train += calculate_abs_MAE(mu_batch, SNR_output_batch, SNR_base_batch)
            mse_train += calculate_abs_MSE(mu_batch, SNR_output_batch, SNR_base_batch)

        loss_train /= len(train_loader)
        mae_train /= len(train_loader)
        mse_train /= len(train_loader)

        return loss_train, mae_train, mse_train
    
    def valid_epoch(self,
                    valid_loader
                    ): 
        self.predictor.eval()
        
        loss_valid = 0
        mae_valid = 0
        mse_valid = 0
        for (delta_pos_batch, ue_rot_batch, SNR_input_batch, SNR_output_batch, SNR_base_batch) in valid_loader:
            delta_pos_batch = delta_pos_batch.to(device)
            ue_rot_batch = ue_rot_batch.to(device)
            SNR_input_batch = SNR_input_batch.to(device)
            SNR_output_batch = SNR_output_batch.to(device)
            SNR_base_batch = SNR_base_batch.to(device)
            
            mu_batch, var_batch = self.predictor(SNR_input_batch, delta_pos_batch, ue_rot_batch)
            # mu_batch = self.predictor(SNR_input_batch)

            loss = self.criterion(mu_batch, var_batch, SNR_output_batch, SNR_base_batch)
            # loss = self.criterion(mu_batch, SNR_output_batch)
            
            loss_valid += loss.detach()
            mae_valid += calculate_abs_MAE(mu_batch, SNR_output_batch, SNR_base_batch)
            mse_valid += calculate_abs_MSE(mu_batch, SNR_output_batch, SNR_base_batch)

        loss_valid /= len(valid_loader)
        mae_valid /= len(valid_loader)
        mse_valid /= len(valid_loader)
        return loss_valid, mae_valid, mse_valid
    
    def train(self, 
              num_epoch, 
              train_loader, 
              valid_loader, 
              reTrain = False
              ):
        torch.cuda.empty_cache()

        if not reTrain and os.path.exists(self.model_path):
            self.predictor.load(self.model_path)

        min_loss_valid = 1e10
        for i in range(num_epoch):
            loss_train, mae_train, mse_train = self.train_epoch(train_loader)
            loss_valid, mae_valid, mse_valid = self.valid_epoch(valid_loader)

            print("epoch ------- %d/%d -------" % (i + 1, num_epoch))
            print("train --- loss: %.4f, MAE: %.4f, MSE: %.4f" % (loss_train, mae_train, mse_train))
            print("valid --- loss: %.4f, MAE: %.4f, MSE: %.4f" % (loss_valid, mae_valid, mse_valid))
            if (mae_valid + mae_train) / 2 < min_loss_valid:
                min_loss_valid = (mae_valid + mae_train) / 2
                self.predictor.save(self.model_path)
    


if __name__ == "__main__":
    print("training device: %s" % str(device))
    generate_window_data()
    train_loader, valid_loader = generate_track_loaders(rate = (0.7, 0.3))
    predictor = Predictor(model_type = "Linear")
    predictor.train(num_epoch = config["Network_Config"]["num_epoch"],
                    train_loader = train_loader, 
                    valid_loader = valid_loader, 
                    reTrain = config["Network_Config"]["reTrain"]
                    )
    # predictor.test(test_loader = test_loader)
    
