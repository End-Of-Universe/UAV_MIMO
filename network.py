import torch
import torch.nn as nn
import numpy as np
import json
import torch.nn.functional as F
import os

with open("./config.json", 'r') as file:
    config = json.load(file)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Layer(nn.Module):
    def __init__(self,
                 in_dim, 
                 out_dim, 
                 dropout_p = None
                 ):
        super().__init__()
        if dropout_p == None:
            self.net = nn.Sequential(nn.Linear(in_features=in_dim, out_features=out_dim),
                                    #  nn.BatchNorm1d(num_features=out_dim),
                                     nn.ReLU())
        else:
            self.net = nn.Sequential(nn.Linear(in_features=in_dim, out_features=out_dim),
                                     nn.Dropout(p = dropout_p),
                                    #  nn.BatchNorm1d(num_features=out_dim),
                                     nn.ReLU())
    def forward(self, x):
        return self.net(x)

class Prob_Accessor(nn.Module):
    def __init__(self, 
                 search_r
                 ):
        super().__init__()
        self.search_r = search_r
        self.hidden_size = 50
        self.net = nn.Sequential(Layer(in_dim = 6, out_dim = self.hidden_size),
                                 Layer(in_dim = self.hidden_size, out_dim = self.hidden_size),
                                 Layer(in_dim = self.hidden_size, out_dim = (self.search_r * 2 + 1) ** 2),
                                 nn.Softmax(dim = 1)
                                 )
    
    def forward(self, 
                delta_pos,      # (batch_size, 3)  
                ue_rot          # (batch_size, 3)
                ):
        batch_size = delta_pos.shape[0]
        x = torch.cat((delta_pos, ue_rot), dim = 1)     # (batch_size, 6)
        prob = self.net(x)
        # length = 2 * self.search_r + 1
        # prob = prob.reshape(batch_size, length, length)
        return prob

    def save(self, 
             model_path
             ):
        torch.save(self.state_dict(), model_path)
        print("Prob_Accessor's params been saved to %s" % model_path)
    
    def load(self, 
             model_path
             ):
        if(str(device) == 'cpu'):
            model = torch.load(model_path, map_location=torch.device('cpu'))
            self.load_state_dict(model)
        else:
            self.load_state_dict(torch.load(model_path, weights_only=True))
        
        print("Prob_Accessor has loaded parameters from %s" % model_path)

# class Accessor_Loss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.ce_loss = nn.CrossEntropyLoss()

#     # prob:     (batch_size, 2 * r + 1, 2 * r + 1)
#     # label:    (batch_size, 2 * r + 1, 2 * r + 1)
#     def forward(self, 
#                 prob, 
#                 label
#                 ):
#         batch_size = prob.shape[0]
#         prob = prob.reshape(batch_size, -1)
#         label = label.reshape(batch_size, -1)
#         label = torch.argmax(label, dim = 1)
#         loss = self.ce_loss(prob, label)
#         return loss

# ------------------------------------- beam tracking network -------------------------------------
class Tracker_SNR_single(nn.Module):
    def __init__(self, 
                 window_input = 5, 
                 window_output = 5,
                 ):
        super().__init__()
        self.hidden_size = 50
        self.linear_net = nn.Sequential(Layer(in_dim = 1, out_dim = self.hidden_size), 
                                        Layer(in_dim = self.hidden_size, out_dim = self.hidden_size // 2),
                                        Layer(in_dim = self.hidden_size // 2, out_dim = 1)
                                        )
        self.projection_net = nn.Linear(in_features = window_input, out_features = window_output)
    def forward(self,
                SNR
                ):
        if SNR.dim() == 2:
            SNR = SNR.unsqueeze(2)      # (batch_size, window_size, 1)
        x = self.linear_net(SNR)
        x = x.squeeze(-1)
        mu_hat = self.projection_net(x).squeeze(-1)
        return mu_hat
    
    def save(self, model_path):
        torch.save(self.state_dict(), model_path)
        print("Tracker_SNR_single's params been saved to %s" % model_path)
    
    def load(self, model_path):
        if(str(device) == 'cpu'):
            model = torch.load(model_path, map_location=torch.device('cpu'))
            self.load_state_dict(model)
        else:
            self.load_state_dict(torch.load(model_path, weights_only=True))
        
        print("Tracker_SNR_single's loaded parameters from %s" % model_path)


# input: 
#   SNR:        (batch_size, window)
#   delta_pos:  (batch_size, window, 3)
#   ue_rot:     (batch_size, window, 3) 
class Tracker_Linear(nn.Module):
    def __init__(self, 
                 window_input = 5, 
                 window_output = 5
                 ):
        super().__init__()
        self.mu_net = Prob_tracker(window_input = window_input, window_output = window_output, type = "mu")
        self.var_net = Prob_tracker(window_input = window_input, window_output = window_output, type = "var")

    def forward(self, 
                SNR,                # (batch_size, window_size)
                delta_pos,          # (batch_size, window_size, 3)
                ue_rot,             # (batch_size, window_size, 3)
                train_phase = None  # train_phase: 1: 只训练mu_net, 寻找中心
                ):                  #              2：mu_net不传递梯度, 训练方差网络
        if train_phase == 1:
            for param in self.mu_net.parameters():
                param.requires_grad_(True)
            for param in self.var_net.parameters():
                param.requires_grad_(False)
            mu_hat = self.mu_net(SNR, delta_pos, ue_rot)
            var_hat = None
        elif train_phase == 2:
            for param in self.mu_net.parameters():
                param.requires_grad_(True)
            for param in self.var_net.parameters():
                param.requires_grad_(True)
            mu_hat = self.mu_net(SNR, delta_pos, ue_rot)
            var_hat = self.var_net(SNR, delta_pos, ue_rot)
        else:
            mu_hat = self.mu_net(SNR, delta_pos, ue_rot)
            var_hat = self.var_net(SNR, delta_pos, ue_rot)
        return mu_hat, var_hat           # (batch_size, window_output) 
    
    def save(self, 
             train_phase = None
             ):
        if train_phase == 1:
            self.mu_net.save(model_dict["Linear"]["mu_net_pth"])
        elif train_phase == 2:
            self.var_net.save(model_dict["Linear"]["var_net_pth"])
        else:
            raise RuntimeError("train_phase must be chosen from (1, 2)!")
    
    def load(self,
             train_phase = None
             ):
        self.mu_net.load(model_dict["Linear"]["mu_net_pth"])
        if train_phase == 2 or train_phase == None:
            if os.path.exists(model_dict["Linear"]["var_net_pth"]):
                self.var_net.load(model_dict["Linear"]["var_net_pth"])

class Prob_tracker(nn.Module):
    def __init__(self, 
                 window_input = 5, 
                 window_output = 5,
                 type = "mu"
                 ):
        super().__init__()
        self.hidden_size = 30
        self.type = type
        self.prob_net = nn.Sequential(Layer(in_dim = 7, out_dim = self.hidden_size, dropout_p = config["Network_Config"]["dropout"]),
                                      Layer(in_dim = self.hidden_size, out_dim = self.hidden_size // 2, dropout_p = config["Network_Config"]["dropout"]),
                                      Layer(in_dim = self.hidden_size // 2, out_dim = 1, dropout_p = config["Network_Config"]["dropout"])
                                      )
        if type == "mu":
            self.projection_net = nn.Linear(in_features = window_input, out_features = window_output)
        elif type == "var":
            self.projection_net = nn.Sequential(nn.Linear(in_features = window_input, out_features = window_output),
                                                nn.Sigmoid())
        else:
            raise RuntimeError("Network Type must be choosen from (\"mu\", \"var\")!")
        
    def forward(self, 
                SNR,            # (batch_size, window_size)
                delta_pos,      # (batch_size, window_size, 3)
                ue_rot          # (batch_size, window_size, 3)
                ):
        if SNR.dim() == 2:
            SNR = SNR.unsqueeze(2)
        x = torch.cat((SNR, delta_pos, ue_rot), dim = 2)    # (batch_size, window_input, 7)
        x = self.prob_net(x).squeeze(-1)          # (batch_size, window_input, 7) -> (batch_size, window_input, 1) 
        x = self.projection_net(x)
        return x           # (batch_size, window_output) 
    
    def save(self, model_path):
        torch.save(self.state_dict(), model_path)
        print("Params of %s_network(Linear) been saved to %s" % (self.type, model_path))
    
    def load(self, model_path):
        if(str(device) == 'cpu'):
            model = torch.load(model_path, map_location=torch.device('cpu'))
            self.load_state_dict(model)
        else:
            self.load_state_dict(torch.load(model_path, weights_only=True))
        print("Params of %s_network(Linear) been loaded from %s" % (self.type, model_path))
        
    
class Tracker_Conv1d(nn.Module):
    def __init__(self, 
                 window_size = 4
                 ):
        super().__init__()
        self.chan_size = 20
        self.patch_embedding = PatchEmbedding(chan_size = self.chan_size, 
                                              window_size = window_size
                                              )
        self.mu_net = nn.Sequential(nn.Conv1d(self.chan_size, self.chan_size,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         dilation=1,
                                         groups=1),
                                     torch.nn.Conv1d(self.chan_size, self.chan_size,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         dilation=1,
                                         groups=1),
                                     DNNHead(chan_size = self.chan_size)
                                     )
            
    def forward(self, 
                SNR_input,
                delta_pos,
                ue_rot
                ):
        mu_input = self.patch_embedding(SNR_input, delta_pos, ue_rot)
        mu_hat = self.mu_net(mu_input)
        return mu_hat
    
    def save(self, model_path):
        torch.save(self.state_dict(), model_path)
        print("Tracker_Conv1d's params been saved to %s" % model_path)
    
    def load(self, model_path):
        if(str(device) == 'cpu'):
            model = torch.load(model_path, map_location=torch.device('cpu'))
            self.load_state_dict(model)
        else:
            self.load_state_dict(torch.load(model_path, weights_only=True))
        
        print("Tracker_Conv1d's loaded parameters from %s" % model_path)

class PatchEmbedding(nn.Module):
    def __init__(self, 
                 chan_size, 
                 window_size
                 ):
        super().__init__()
        self.depthwise_mu = torch.nn.Conv1d(8, chan_size,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         dilation=1,
                                         groups=1)
        
        
        self.positions_mu = nn.Parameter(torch.randn(chan_size, window_size))
        
        self.cls_token_mu = nn.Parameter(torch.randn(1, window_size, 1))
    
    def forward(self, 
                SNR_input,
                delta_pos,
                ue_rot
                ):
        if SNR_input.dim() == 2:
            SNR_input = SNR_input.unsqueeze(2)
        batch_size = SNR_input.shape[0]

        cls_token_mu_batch = self.cls_token_mu.repeat(batch_size, 1, 1)
        # (batch_size, window_size, 1 + 3 + 3 + 1) -> exchange dimmension : (batch_size, 8, window_size)
        mu_input = torch.cat((SNR_input, delta_pos, ue_rot, cls_token_mu_batch), dim = 2).permute(0, 2, 1)      # exchange dimmension  
        mu_input = self.depthwise_mu(mu_input)      # (batch_size, chan_size, window_size)
        mu_input += self.positions_mu

        return mu_input

class DNNHead(nn.Module):
    def __init__(self, 
                 chan_size
                 ):
        super().__init__()
        self.depthwise = torch.nn.Conv1d(chan_size, 1,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         dilation=1,
                                         groups=1
                                         )
        
        self.linear = nn.Linear(in_features = config["Track_Dataset_Config"]["window_input"],
                                out_features = 1
                                )
    def forward(self, x):
        x = self.depthwise(x).squeeze(1)
        x = self.linear(x).squeeze(1)
        return x


class Tracker_LSTM(nn.Module):
    def __init__(self, 
                 window_input = 5,
                 window_output = 5
                 ):
        super().__init__()    
        self.hidden_size = 50
        self.num_layers = 5
        self.mu_lstm = nn.LSTM(input_size = 7,
                          hidden_size = self.hidden_size,
                          num_layers = self.num_layers
                          ) 
        # self.var_lstm = nn.LSTM(input_size = 7,
        #                   hidden_size = self.hidden_size,
        #                   num_layers = self.num_layers
        #                   ) 
        self.mu_projection = nn.Linear(self.hidden_size, window_output)
        # self.var_projection = nn.Sequential(nn.Linear(self.hidden_size, 1),
        #                                     nn.ReLU()
        #                                     )
        
    def forward(self, 
                SNR_input,
                delta_pos,
                ue_rot
                ):
        if SNR_input.dim() == 2:
            SNR_input = SNR_input.unsqueeze(2)

        x = torch.cat((SNR_input, delta_pos, ue_rot), dim = 2).permute(1, 0, 2)     # (window_size, batch_size, 7)

        h0 = torch.zeros(self.num_layers, SNR_input.shape[0], self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, SNR_input.shape[0], self.hidden_size).to(x.device)

        mu_hat, _ = self.mu_lstm(x, (h0, c0))               # mu_hat: (window_size, batch_size, hidden_size)
        # var_hat, _ = self.var_lstm(x, (h0, c0))
        mu_hat = self.mu_projection(mu_hat[-1, :, :])       # (batch_size, window_output)
        # var_hat = self.var_projection(var_hat[-1, :, :]).squeeze(1)
        return mu_hat#, var_hat
    
    def save(self, model_path):
        torch.save(self.state_dict(), model_path)
        print("Tracker_LSTM's params been saved to %s" % model_path)
    
    def load(self, model_path):
        if(str(device) == 'cpu'):
            model = torch.load(model_path, map_location=torch.device('cpu'))
            self.load_state_dict(model)
        else:
            self.load_state_dict(torch.load(model_path, weights_only=True))
        
        print("Tracker_LSTM's loaded parameters from %s" % model_path)
        
class Track_Loss(nn.Module):
    def __init__(self, 
                 train_phase = 1
                 ):
        super().__init__()
        if train_phase == 1 or train_phase == 2:
            self.train_phase = train_phase
        else:
            raise RuntimeError("train phase must be in (1, 2)!")
    
    def forward(self,
                mu_hat,         # (batch_size, window_output)
                var_hat,        # (batch_size, window_output)
                SNR_label,
                SNR_base        # (batch_size, )       
                ):
        if self.train_phase == 1:
            loss = self.calculate_mse_loss(mu_hat, SNR_label)
        else:
            loss = self.calculate_log_loss(mu_hat, var_hat, SNR_label, SNR_base)
        return loss
    def calculate_log_loss(self,
                           mu_hat,         # (batch_size, window_output)
                           var_hat,     # (batch_size, window_output)
                           SNR_label,
                           SNR_base
                           ):
        mu_hat = (1 + mu_hat) * SNR_base[:, None]
        SNR_label = (1 + SNR_label) * SNR_base[:, None]
        epsilon = 1e-5
        normx = SNR_label - mu_hat
        sx = var_hat + epsilon
        z = (normx / sx) ** 2
        pdf = torch.exp(-z / 2) / (np.sqrt(2 * np.pi) * sx)
        log_pdf = torch.log(torch.clamp(pdf, min=epsilon))

        loss = -torch.mean(torch.sum(log_pdf, dim = 1), dim = 0)
        return loss
    
    def calculate_mae_loss(self, 
                           mu_hat,
                           SNR_label
                           ):
        # mu_hat: (batch_size, window_output)
        # SNR_label: (batch_size, window_output)
        loss = torch.abs(mu_hat - SNR_label)
        loss = torch.mean(torch.sum(loss, dim = 1), dim = 0)
        return loss
    
    def calculate_mse_loss(self, 
                           mu_hat,
                           SNR_label
                           ):
        # mu_hat: (batch_size, window_output)
        # SNR_label: (batch_size, window_output)
        loss = (mu_hat - SNR_label) ** 2
        loss = torch.mean(torch.sum(loss, dim = 1), dim = 0)
        return loss

model_dict = {
    "Prob_Accessor": {
        "bs_model_pth": "./model/bs_prob_accessor.pth",
        "ue_model_pth": "./model/ue_prob_accessor.pth",
        "bs_predictor": Prob_Accessor(search_r = config["Access_Dataset_Config"]["bs_search_r"]),
        "ue_predictor": Prob_Accessor(search_r = config["Access_Dataset_Config"]["ue_search_r"])
    },

    "Linear": {
        "mu_net_pth": "./model/Linear_mu_net_%din%dout.pth" % (config["Track_Dataset_Config"]["window_input"],
                                                               config["Track_Dataset_Config"]["window_output"]
                                                               ),
        "var_net_pth": "./model/Linear_var_net_%din%dout.pth" % (config["Track_Dataset_Config"]["window_input"],
                                                                 config["Track_Dataset_Config"]["window_output"]
                                                                 ),
        "predictor": Tracker_Linear(window_input = config["Track_Dataset_Config"]["window_input"],
                                    window_output = config["Track_Dataset_Config"]["window_output"])
    },
    "SNR_single": {
        "model_pth": "./model/track_model_SNR_single.pth",
        "predictor": Tracker_SNR_single(window_input = config["Track_Dataset_Config"]["window_input"])
    }
}

if __name__ == "__main__":
    batch_size = 10
    # window_size = config["Track_Dataset_Config"]["window_input"]

    # SNR = torch.rand(size = (batch_size, window_size))
    # delta_pos = torch.rand(size = (batch_size, window_size, 3))
    # ue_rot = torch.rand(size = (batch_size, window_size, 3))

    # # mu_hat = torch.rand(size = (batch_size, ))
    # # sigma2_hat = torch.rand(size = (batch_size, ))
    # SNR_label = torch.rand(size = (batch_size, ))

    # model = model_dict["Linear"]["predictor"]
    # model.save(2)
    # mu, var = model(SNR, delta_pos, ue_rot, 2)
    # print(mu)
    # print(var)

    model = Prob_Accessor(search_r = 3)
    delta_pos = torch.rand((batch_size, 3))
    ue_rot = torch.rand((batch_size, 3))
    prob = model(delta_pos, ue_rot)
    # print(torch.sum(prob, dim = 1))

    criterion = nn.CrossEntropyLoss()
    label = torch.zeros((batch_size), dtype = torch.long)
    loss = criterion(prob, label)
    print(loss)
    print(prob.shape)

    
    
