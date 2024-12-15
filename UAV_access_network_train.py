import torch
import os
import json
import torch.nn.functional as F
from torch import nn
from generate_dataset import generate_access_loaders
from network import model_dict

os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with open("./config.json", 'r') as file:
    config = json.load(file)

# delta_SNR_hat:  (batch_size, window_output)
# delta_SNR_real: (batch_size, window_output)
# SNR_base:       (batch_size, )

class Predictor:
    def __init__(self, 
                 type = "bs"
                 ):
        if type == "bs":
            self.predictor = model_dict["Prob_Accessor"]["bs_predictor"].to(device)
            self.model_path = model_dict["Prob_Accessor"]["bs_model_pth"]
        elif type == "ue":
            self.predictor = model_dict["Prob_Accessor"]["ue_predictor"].to(device)
            self.model_path = model_dict["Prob_Accessor"]["ue_model_pth"]
        else:
            raise RuntimeError("type must choosen from (\"bs\", \"ue\")")
        
        self.type = type
        self.criterion = nn.CrossEntropyLoss().to(device) #nn.L1Loss().to(device)
        self.optimizer = torch.optim.RMSprop(self.predictor.parameters(), lr=1e-3)
        
    def train_epoch(self, 
                    train_loader
                    ):
        self.predictor.train()
        
        loss_train = 0
        acc_train = 0
        for (delta_pos_batch, ue_rot_batch, label_bs_batch, label_ue_batch) in train_loader:
            delta_pos_batch = delta_pos_batch.to(device)
            ue_rot_batch = ue_rot_batch.to(device)
            self.optimizer.zero_grad()
            if self.type == "bs":
                label_bs_batch = label_bs_batch.to(device)
                prob = self.predictor(delta_pos_batch, ue_rot_batch)
                loss = self.criterion(prob, label_bs_batch)
                acc = torch.mean((torch.argmax(prob, dim = 1) == label_bs_batch).to(torch.float32))
            else:
                label_bs_batch = label_ue_batch.to(device)
                prob = self.predictor(delta_pos_batch, ue_rot_batch)
                loss = self.criterion(prob, label_ue_batch)
                acc = torch.mean((torch.argmax(prob, dim = 1) == label_ue_batch).to(torch.float32))

            loss.backward()

            self.optimizer.step()
            loss_train += loss.detach()
            acc_train += acc

        loss_train /= len(train_loader)
        acc_train /= len(train_loader)

        return loss_train, acc_train
    
    def valid_epoch(self,
                    valid_loader
                    ): 
        self.predictor.eval()
        
        loss_valid = 0
        acc_valid = 0
        for (delta_pos_batch, ue_rot_batch, label_bs_batch, label_ue_batch) in valid_loader:
            delta_pos_batch = delta_pos_batch.to(device)
            ue_rot_batch = ue_rot_batch.to(device)
            if self.type == "bs":
                label_bs_batch = label_bs_batch.to(device)
                prob = self.predictor(delta_pos_batch, ue_rot_batch)
                loss = self.criterion(prob, label_bs_batch)
                acc = torch.mean((torch.argmax(prob, dim = 1) == label_bs_batch).to(torch.float32))
            else:
                label_bs_batch = label_ue_batch.to(device)
                prob = self.predictor(delta_pos_batch, ue_rot_batch)
                loss = self.criterion(prob, label_ue_batch)
                acc = torch.mean((torch.argmax(prob, dim = 1) == label_ue_batch).to(torch.float32))

            loss_valid += loss.detach()
            acc_valid += acc

        loss_valid /= len(valid_loader)
        acc_valid /= len(valid_loader)
        return loss_valid, acc_valid
    
    def train(self, 
              num_epoch, 
              train_loader, 
              valid_loader, 
              reTrain = False
              ):
        torch.cuda.empty_cache()

        if not reTrain and os.path.exists(self.model_path):
            self.predictor.load(self.model_path)

        max_acc = 0
        for i in range(num_epoch):
            loss_train, acc_train = self.train_epoch(train_loader)
            loss_valid, acc_valid = self.valid_epoch(valid_loader)

            print("epoch ------- %d/%d -------" % (i + 1, num_epoch))
            print("train --- loss: %.4f, Acc: %.4f" % (loss_train, acc_train))
            print("valid --- loss: %.4f, Acc: %.4f" % (loss_valid, acc_train))
            if (acc_train + acc_valid) / 2 > max_acc:
                max_acc = (acc_valid + acc_train) / 2
                self.predictor.save(self.model_path)
    

if __name__ == "__main__":
    print("training device: %s" % str(device))

    train_loader, valid_loader = generate_access_loaders(rate = (0.7, 0.3))
    predictor = Predictor(type = "ue")
    predictor.train(num_epoch = config["Network_Config"]["num_epoch"],
                    train_loader = train_loader, 
                    valid_loader = valid_loader, 
                    reTrain = config["Network_Config"]["reTrain"]
                    )
    # predictor.test(test_loader = test_loader)
    
