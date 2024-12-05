import torch
from network import Selector_1, Loss
import os

from params import params
from generate_dataset import generate_loaders
from utils import calculate_topk_acc

os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Predictor:
    def __init__(self):
        self.predictor = Selector_1(hidden_size = 30).to(device)
        self.criterion = Loss().to(device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-2)
        self.model_path = params["selector_1_params_pth"]
        
    def train_epoch(self, 
                    train_loader
                    ):
        
        self.predictor.train()
        
        loss_train = 0
        bs_acc_train = 0
        ue_acc_train = 0
        for (delta_pos_batch, ue_rot_batch, bs_label_batch, ue_label_batch) in train_loader:
            delta_pos_batch = delta_pos_batch.to(device)
            ue_rot_batch = ue_rot_batch.to(device)
            bs_label_batch = bs_label_batch.to(device)
            ue_label_batch = ue_label_batch.to(device)
            
            self.optimizer.zero_grad()
            
            bs_p, ue_p = self.predictor(delta_pos_batch, ue_rot_batch)
            
            loss = self.criterion(bs_p, ue_p, bs_label_batch, ue_label_batch)
            
            loss.backward()

            self.optimizer.step()
            loss_train += loss.detach()
            bs_acc_train += calculate_topk_acc(bs_p, bs_label_batch)
            ue_acc_train += calculate_topk_acc(ue_p, ue_label_batch)

        loss_train /= len(train_loader)
        bs_acc_train /= len(train_loader)
        ue_acc_train /= len(train_loader)

        return loss_train, bs_acc_train, ue_acc_train
    
    def valid_epoch(self,
                    valid_loader
                    ): 
        self.predictor.eval()
        
        loss_valid = 0
        bs_acc_valid = 0
        ue_acc_valid = 0
        for (delta_pos_batch, ue_rot_batch, bs_label_batch, ue_label_batch) in valid_loader:
            delta_pos_batch = delta_pos_batch.to(device)
            ue_rot_batch = ue_rot_batch.to(device)
            bs_label_batch = bs_label_batch.to(device)
            ue_label_batch = ue_label_batch.to(device)
            
            bs_p, ue_p = self.predictor(delta_pos_batch, ue_rot_batch)
            
            loss = self.criterion(bs_p, ue_p, bs_label_batch, ue_label_batch)
            
            loss_valid += loss.detach()

            bs_acc_valid += calculate_topk_acc(bs_p, bs_label_batch)
            ue_acc_valid += calculate_topk_acc(ue_p, ue_label_batch)
        loss_valid /= len(valid_loader)
        bs_acc_valid /= len(valid_loader)
        ue_acc_valid /= len(valid_loader)

        return loss_valid, bs_acc_valid, ue_acc_valid
    
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
            loss_train, bs_acc_train, ue_acc_train = self.train_epoch(train_loader)
            loss_valid, bs_acc_valid, ue_acc_valid = self.valid_epoch(valid_loader)

            print("epoch: ------- %d/%d -------" % (i + 1, num_epoch))
            print("train --- loss: %.4f, BS acc: %.4f, UE acc: %.4f" % (loss_train, bs_acc_train, ue_acc_train))
            print("valid --- loss: %.4f, BS acc: %.4f, UE acc: %.4f" % (loss_valid, bs_acc_valid, ue_acc_valid))
            if loss_valid < min_loss_valid:
                min_loss_valid = loss_valid
                self.predictor.save(self.model_path)

if __name__ == "__main__":
    print("training device: %s" % str(device))
    train_loader, valid_loader, test_loader = generate_loaders()
    predictor = Predictor()
    predictor.train(num_epoch = params["num_epoch"],
                    train_loader = train_loader, 
                    valid_loader = valid_loader, 
                    reTrain = params["reTrain"]
                    )
    
