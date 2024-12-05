# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 09:59:12 2021

@author: mengxiaomaomao
E-mail: mengfan@pmlabs.com.cn or mengxiaomaomao@outlook.com
"""
import numpy as np

ftype = np.float32
ctype = np.complex64

def generate_array_response(y_antenna_num, 
                            z_antenna_num
                            ):
    antenna_num = z_antenna_num * y_antenna_num
    u_mat = np.tile(np.tile(np.linspace(0, z_antenna_num - 1, z_antenna_num), [y_antenna_num, 1]).T, [y_antenna_num, z_antenna_num]) - z_antenna_num / 2
    v_mat = np.tile(np.tile(np.linspace(0, y_antenna_num - 1, y_antenna_num), [z_antenna_num, 1]).T.reshape(-1, 1), [1, antenna_num]) - y_antenna_num / 2
    z_mat = np.tile(np.tile(np.linspace(0, z_antenna_num - 1, z_antenna_num), [y_antenna_num, 1]), [z_antenna_num, y_antenna_num]) - z_antenna_num / 2
    y_mat = np.tile(np.tile(np.linspace(0, y_antenna_num - 1, y_antenna_num), [z_antenna_num, 1]).T.reshape(1, -1), [antenna_num, 1]) - y_antenna_num / 2
    dft_codebook = 1. / np.sqrt(antenna_num) * np.exp(-1j * 2 * np.pi * (u_mat * z_mat / z_antenna_num + v_mat * y_mat / y_antenna_num))      
    return dft_codebook

class Dft_codebook:
    def __init__(self, 
                 bs_z_antenna_num = 64, 
                 bs_y_antenna_num = 32, 
                 ue_z_antenna_num = 64, 
                 ue_y_antenna_num = 32
                 ):
        self.bs_z_antenna_num = bs_z_antenna_num
        self.bs_y_antenna_num = bs_y_antenna_num
        self.ue_z_antenna_num = ue_z_antenna_num
        self.ue_y_antenna_num = ue_y_antenna_num

        self.Nt = self.bs_z_antenna_num * self.bs_y_antenna_num # number of antennas of the transmitter array
        self.Nr = self.ue_z_antenna_num * self.ue_y_antenna_num # number of antennas of the receiver array

        self.generate_dft_codebook()

    def __generate_bs_codebook(self):
        self.bs_dft_codebook = generate_array_response(self.bs_y_antenna_num, self.bs_z_antenna_num)

    def __generate_ue_codebook(self):
        self.ue_dft_codebook = generate_array_response(self.ue_y_antenna_num, self.ue_z_antenna_num)

    def generate_dft_codebook(self):
        self.__generate_bs_codebook()
        
        self.__generate_ue_codebook()



if __name__ == "__main__":
    bs_z_antenna_num = 64
    bs_y_antenna_num = 32
    Nt = bs_z_antenna_num * bs_y_antenna_num

    codebook = Dft_codebook(bs_z_antenna_num, bs_y_antenna_num, 8, 8)

    bs_codebook = codebook.bs_dft_codebook

    phi_aod = -20 / 180 * np.pi      # phi:      (-pi/2, +pi/2), 在0 degree附近最准
    theta_zod = 70 / 180 * np.pi    # theta:    (0, pi), 在90 degree附近最准
    kd = np.pi
        
    # generate transmitter phase array
    z_mat = np.tile(np.linspace(0, bs_z_antenna_num - 1, bs_z_antenna_num), [bs_y_antenna_num, 1]).T - bs_z_antenna_num // 2
    y_mat = np.tile(np.linspace(0, bs_y_antenna_num - 1, bs_y_antenna_num), [bs_z_antenna_num, 1]) - bs_y_antenna_num // 2

    phase_array = y_mat * np.sin(phi_aod) * np.sin(theta_zod) + z_mat * np.cos(theta_zod)
    a_tx = 1. / np.sqrt(Nt) * np.exp(1j * kd * phase_array)
    a_tx = np.reshape(a_tx.T, [Nt])

    psd = np.abs(np.matmul(np.conj(bs_codebook).T, a_tx))
    psd = np.reshape(psd, (bs_y_antenna_num, bs_z_antenna_num))

    from matplotlib import pyplot as plt
    import seaborn as sns
    plt.close('all')    

    fig = plt.figure(1, figsize=(10, 8))
    cmap = sns.heatmap(psd)
    plt.xlabel('phi_aod')
    plt.ylabel('theta_zod')

    
    ## 匹配最优波束
    k = np.argmax(psd.reshape(Nt))
    # y = k // bs_z_antenna_num
    # z = k - y * bs_z_antenna_num
    (y, z) = np.unravel_index(k, [bs_y_antenna_num, bs_z_antenna_num])
    
    est_theta_zod = np.arccos(1 - 2 * z / bs_z_antenna_num)
    sin_phi_aod = (1 - 2 * y / bs_y_antenna_num) / np.sin(est_theta_zod)
    est_phi_aod = np.arcsin( min(sin_phi_aod, 1) ) 

    est_phi_aod = est_phi_aod * 180 / np.pi
    est_theta_zod = est_theta_zod * 180 / np.pi
    plt.text(z, y, "est_phi_aod:%.1f\nest_theta_zod:%.1f" % (est_phi_aod, est_theta_zod), color="red")
    plt.show()
    