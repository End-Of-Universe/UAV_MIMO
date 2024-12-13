import numpy as np
import json
itype = np.int32
ftype = np.float32
ctype = np.complex64

with open("./config.json", 'r') as file:
    config = json.load(file)

def GCS_to_LCS(theta, 
               phi, 
               alpha = 0, 
               beta = 0, 
               gamma = 0
               ):
    '''
    input:  phi and theta in GCS
    output: phi and theta in LCS
    '''
    theta_ = np.arccos(np.cos(beta) * np.cos(gamma) * np.cos(theta) + 
                       (np.sin(beta) * np.cos(gamma) * np.cos(phi - alpha) -
                        np.sin(gamma) * np.sin(phi - alpha)) * np.sin(theta))
    
    phi_ = np.angle(np.cos(beta) * np.sin(theta) * np.cos(phi - alpha) - np.sin(beta) * np.cos(theta) +
                    1j * (np.cos(beta) * np.sin(gamma) * np.cos(theta) + 
                        (np.sin(beta) * np.sin(gamma) * np.cos(phi - alpha) +
                        np.cos(gamma) * np.sin(phi - alpha)) * np.sin(theta)))
    
    return theta_, phi_

# def LCS_to_GCS(theta, 
#                phi, 
#                alpha = 0, 
#                beta = 0, 
#                gamma = 0
#                ):
#     '''
#     input:  phi and theta in LCS
#     output: phi and theta in GCS
#     '''
#     rho_ = np.array([[np.sin(theta)*np.cos(phi)], 
#                      [np.sin(theta)*np.sin(phi)], 
#                      [np.cos(theta)]], dtype = ftype)

#     r_z = R_z(alpha)
#     r_y = R_y(beta)
#     r_x = R_x(gamma)
#     R_ = np.matmul(np.matmul(r_z, r_y), r_x)

#     R_rho_ = np.matmul(R_, rho_)

#     tmp_1 = np.array([[0.], 
#                       [0.], 
#                       [1.]], dtype = ftype)
    
#     tmp_2 = np.array([[1.], 
#                       [1j], 
#                       [0.]], dtype = ctype)

#     theta_ = np.arccos(np.matmul(tmp_1.T, R_rho_))[0, 0]
#     phi_ = np.angle(np.matmul(tmp_2.T, R_rho_))[0, 0]
#     return theta_, phi_

def generate_complex_gaussian(size = ()):
    # return np.ones(shape=size) + 1j * np.zeros(shape=size)
    return (np.random.normal(size = size) + 1j * np.random.normal(size = size)) / 8 #np.sqrt(2.0)


def R_z(a):
    r_z = np.array([[np.cos(a), -np.sin(a), 0.], 
                    [np.sin(a),  np.cos(a), 0.], 
                    [       0.,         0., 1.]], dtype = ftype)
    return r_z
    
def R_y(a):
    r_y = np.array([[ np.cos(a), 0., np.sin(a)], 
                    [        0., 1.,        0.], 
                    [-np.sin(a), 0., np.cos(a)]], dtype = ftype)
    return r_y
    
def R_x(a):
    r_x = np.array([[1.,        0.,         0.], 
                    [0., np.cos(a), -np.sin(a)], 
                    [0., np.sin(a),  np.cos(a)]], dtype = ftype)
    return r_x

def cartesian_to_polar(delta_pos):
    (x, y, z) = delta_pos
    phi = np.arctan(y / x)
    theta = np.arctan(np.sqrt(x ** 2 + y ** 2) / z)
    if x < 0:
        phi += np.pi
    if z < 0:
        theta += np.pi
    return theta, phi

def convert_to_L1RSRP(x, power):
    high_bound = 140 # -40 dB
    low_bound = -140 # -140 dB
    x_quan = 20 * np.log10(x)
    x_quan += power
    x_quan = np.minimum(np.maximum(x_quan, low_bound), high_bound)
    x_quan = np.round(x_quan)
    # x_quan = (x_quan - low_bound)/(high_bound - low_bound)
    return x_quan

def calculate_overall_noise():
    noise_figure = config["UAV_Scenario_Config"]["ue_rx_noise_figure"]
    sigma2_dBm = config["UAV_Scenario_Config"]["sigma2_dBm"] + 10 * np.log10(config["UAV_Scenario_Config"]["band_width"] * 1e6) + noise_figure 
    return sigma2_dBm

## 根据相对位置和UAV姿态计算LCS坐标系下的zod, aod, zoa, aoa
def calculate_los_angle(delta_pos,
                  rot
                  ):
    ## 传入的必须为一维向量
    if delta_pos.shape[0] != 3:
        delta_pos = delta_pos[0]

    theta_zod_los, phi_aod_los = cartesian_to_polar(delta_pos)

    phi_aoa_los = np.pi + phi_aod_los
    theta_zoa_los = np.pi - theta_zod_los
    
    theta_zod_los, phi_aod_los = GCS_to_LCS(theta_zod_los, phi_aod_los, 
                                        alpha = np.pi / 2., 
                                        beta = -config["UAV_Scenario_Config"]["bs_tilt"], 
                                        gamma = 0.
                                        )
    
    theta_zoa_los, phi_aoa_los = GCS_to_LCS(theta_zoa_los, phi_aoa_los, 
                                        alpha = -np.pi / 2 + rot[0], 
                                        beta = config["UAV_Scenario_Config"]["bs_tilt"] + rot[1], 
                                        gamma = rot[2]
                                        )
    return theta_zod_los, phi_aod_los, theta_zoa_los, phi_aoa_los


## 根据波束的索引求los方向角，返回LCS坐标系下的方向角
def calculate_upa_angle(k, 
                  z_num, 
                  y_num
                  ):
    # y = k // z_num
    # z = k - y * z_num
    (y, z) = np.unravel_index(k, [y_num, z_num])
    est_theta = np.arccos(1 - 2 * z / z_num)
    sin_phi = max(min((1 - 2 * y / y_num) / np.sin(est_theta), 1), -1)
    est_phi = np.arcsin(sin_phi)
    return est_theta, est_phi

## 根据入射角计算波束的2D索引
def calculate_upa_index(theta,
                        phi,
                        z_num,
                        y_num
                        ):
    z_index = (1 - np.cos(theta)) * z_num / 2
    y_index = (1 - np.sin(theta) * np.sin(phi)) * y_num / 2
    
    z_index = round(z_index)
    y_index = round(y_index)
    
    return z_index, y_index

## 角度 -> 弧度
def degree_2_radian(*angles_d):
    angles_r = list()
    for angle_d in angles_d:
        angle_r = angle_d / 180 * np.pi
        angles_r.append(angle_r)
    return angles_r
## 弧度 -> 角度
def radian_2_degree(*angles_r):
    angles_d = list()
    for angle_r in angles_r:
        angle_d = angle_r / np.pi * 180
        angles_d.append(angle_d)
    return angles_d

def calculate_topk_acc(p_hat,           ## 估计概率         (Ns, N)
                       beam_label,      ## 真实波束索引     (Ns, )
                       k = 5, 
                       ):
    batch_size = p_hat.shape[0]
    beam_space_size = p_hat.shape[1]
    
    p_hat = p_hat.cpu().detach().numpy()
    beam_label = beam_label.cpu().detach().numpy()
    p_hat_topk = np.argsort(p_hat, axis = 1)[:, beam_space_size - k : ]
    count = 0
    for i in range(batch_size):
        if beam_label[i] in p_hat_topk[i]:
            count += 1
    return count / p_hat.shape[0]

def calculate_SNR(codebook,
                  h,
                  beam_bs,  # index 
                  beam_ue   # index
                  ):
    P_bs = np.power(10, config["UAV_Scenario_Config"]["bs_tx_power_max"] / 10) / 1000      # W
    sigma2_dBm = config["UAV_Scenario_Config"]["sigma2_dBm"] + 10 * np.log10(config["UAV_Scenario_Config"]["band_width"] * 1e6) + config["UAV_Scenario_Config"]["ue_rx_noise_figure"]
    sigma2 = sigma2_dBm - 30    
    sigma2 = np.power(10, sigma2 / 10)                              # mW转化为W     

    bs_codebook = codebook.bs_dft_codebook
    ue_codebook = codebook.ue_dft_codebook

    hr = np.matmul(h, bs_codebook[:, beam_bs])
    br = np.matmul(np.conjugate(ue_codebook[:, beam_ue]).T, hr)
    
    snr = P_bs * np.abs(br) ** 2 / sigma2
    return 10 * np.log10(snr)

if __name__ == "__main__":
    # gcs_theta_zod = np.pi / 3
    # gcs_phi_aod = np.pi / 2

    # lcs_theta_zod, lcs_phi_aod = GCS_to_LCS(gcs_theta_zod, gcs_phi_aod,
    #                                         alpha=-np.pi/2,
    #                                         beta=0,
    #                                         gamma=0
    #                                         )
    # print(lcs_theta_zod * 180 / np.pi, lcs_phi_aod * 180 / np.pi)

    zod_r, aod_r, zoa_r, aoa_r = np.pi / 2, np.pi / 3, np.pi / 6, -np.pi / 3
    zod_d, aod_d, zoa_d, aoa_d = radian_2_degree(zod_r, aod_r, zoa_r, aoa_r)
    print(zod_d, aod_d, zoa_d, aoa_d)


