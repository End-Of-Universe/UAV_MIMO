"""
scenario: 3-D UAV
one cell, MU-MIMO
consider commercial mmwave antenna configuration
BS: 32 x 16 MIMO UPA
BS location: [0, 0, 2.5]
BS tilt: -30 degree

UE: 32 x 16 MIMO UPA
UE tilt: 30 degree

LOS Scenario

This is a .py file to generate training data.
"""

import numpy as np
import json
import h5py
from tqdm import tqdm
import multiprocessing
import torch
import os

import utils
from dft_codebook import Dft_codebook
from access_and_track import initial_access_3
from network import model_dict
itype = np.int32
ftype = np.float32
ctype = np.complex64

os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with open("./config.json", 'r') as file:
    config = json.load(file)

class UAV_environment:
    def __init__(self):
        self.bs_z_antenna_num = config["UAV_Scenario_Config"]["bs_z_antenna_num"]
        self.bs_y_antenna_num = config["UAV_Scenario_Config"]["bs_y_antenna_num"]
        self.ue_z_antenna_num = config["UAV_Scenario_Config"]["ue_z_antenna_num"]
        self.ue_y_antenna_num = config["UAV_Scenario_Config"]["ue_y_antenna_num"]

        self.Nt = self.bs_z_antenna_num * self.bs_y_antenna_num # number of antennas of the transmitter array
        self.Nr = self.ue_z_antenna_num * self.ue_y_antenna_num # number of antennas of the receiver array
        

        self.c = config["UAV_Scenario_Config"]["c"]                    # light speed
        self.f_c = config["UAV_Scenario_Config"]["f_c"]                # Hz carrier frequency
        self.band_width = config["UAV_Scenario_Config"]["band_width"]  # MHz
        self.lamd = self.c / self.f_c           # wavelength
        self.d = self.lamd / 2                  # distance between elements of planar array( we consider half-wavelength arrays)
        self.kd = 2 * np.pi / self.lamd * self.d

        self.t_csi = config["UAV_Scenario_Config"]["t_csi"]                    # periodicity of CSI-RS, second 信道状态更新频率

        self.bs_tilt = config["UAV_Scenario_Config"]["bs_tilt"]

        self.bs_rx_noise_figure = config["UAV_Scenario_Config"]["bs_rx_noise_figure"]  # dB
        self.ue_rx_noise_figure = config["UAV_Scenario_Config"]["ue_rx_noise_figure"]  # dB
        self.sigma2_dBm = config["UAV_Scenario_Config"]["sigma2_dBm"]  # dBm/Hz environment noise power
        
        
    def __generate_bs_location(self):
        '''
        bs 3D location
        '''
        self.bs_loc = np.array([0, 0, 2.5]) # shape [1, 3]

    def __generate_initial_ue_location(self):
        '''
        ue 3D location distribution
        z-axis height range (100, 300) meter
        y-axis range (500, 1,500) meter
        x-axis lies in +- 30 degree
        '''
        ini_pos_y_low = config["UAV_Scenario_Config"]["ini_pos_y_low"]
        ini_pos_y_high = config["UAV_Scenario_Config"]["ini_pos_y_high"]
        ue_angle = np.random.uniform(low = -np.pi / 6, high = np.pi / 6, size = (self.ue_num))
        ue_z = np.random.uniform(low = 100., high = 200., size = (self.ue_num))
        ue_y = np.random.uniform(low = ini_pos_y_low, high = ini_pos_y_high, size = (self.ue_num))

        ue_x = ue_y * np.tan(ue_angle)
        self.ue_ini_loc = np.vstack((ue_x, ue_y, ue_z)).T # shape [ue_num, 3]

    def __generate_initial_ue_rotation(self):
        self.ue_rot = np.zeros((self.ue_num, 3, self.times_pilot), dtype = ftype)
        r_max = np.pi / 2
        ue_r_z = np.random.uniform(low = -r_max, high = r_max, size = (self.ue_num)) * self.t_csi
        ue_r_y = np.random.uniform(low = -r_max, high = r_max, size = (self.ue_num)) * self.t_csi
        ue_r_x = np.random.uniform(low = -r_max, high = r_max, size = (self.ue_num)) * self.t_csi
        self.ue_rot[:, :, 0] = np.vstack((ue_r_x, ue_r_y, ue_r_z)).T # shape [ue_num, 3]
        self.ue_rot_prev = None
    
    def __update_ue_rotation(self):
        '''
        follow 1-order Markov process
        '''
        if self.ue_rot_prev is not None:
            self.ue_rot[:, :, self.t] = self.ue_rot_prev + self.t_csi * self.rot_speed + 0.5 * self.rot_acc * self.t_csi ** 2
        self.ue_rot_prev = self.ue_rot[:, :, self.t]

    def __generate_initial_ue_speed(self):
        '''
        ue 3D speed distribution
        z-axis speed (-3, 3) m/s
        xy-axis speed 15 m/s
        
        ue 3D acceleration distribution
        z-axis acc (-1, 1) m/s^2
        y-axis acc (-1, 1) m/s^2
        x-axis acc (-1, 1) m/s^2
        '''
        ue_speed_xy_high = config["UAV_Scenario_Config"]["move_speed_xy_high"]
        ue_speed = np.random.uniform(low = -ue_speed_xy_high, high = ue_speed_xy_high, size = (self.ue_num))
        ue_angle = np.random.uniform(low = 0, high = np.pi * 2, size = (self.ue_num))
        # ue_speed = np.random.uniform(low = 0, high = 0, size = (self.ue_num))
        # ue_angle = np.random.uniform(low = 0, high = 0, size = (self.ue_num))

        ## 3
        ue_speed_z_high = config["UAV_Scenario_Config"]["move_speed_z_high"]
        ue_speed_z = np.random.uniform(low = -ue_speed_z_high, high = ue_speed_z_high, size = (self.ue_num))
        # ue_speed_z = np.random.uniform(low = 0, high = 0, size = (self.ue_num))
        ue_speed_y = ue_speed * np.cos(ue_angle)
        ue_speed_x = ue_speed * np.sin(ue_angle)
        self.ue_speed = np.vstack((ue_speed_x, ue_speed_y, ue_speed_z)).T # shape [ue_num, 3]

        ue_acc_z = np.random.uniform(low = -1, high = 1, size = (self.ue_num))
        ue_acc_y = np.random.uniform(low = -1, high = 1, size = (self.ue_num))
        ue_acc_x = np.random.uniform(low = -1, high = 1, size = (self.ue_num))
        # ue_acc_z = np.random.uniform(low = -0.01, high = 0.01, size = (self.ue_num))
        # ue_acc_y = np.random.uniform(low = -0.01, high = 0.01, size = (self.ue_num))
        # ue_acc_x = np.random.uniform(low = -0.01, high = 0.01, size = (self.ue_num))
        self.ue_acc = np.vstack((ue_acc_x, ue_acc_y, ue_acc_z)).T # shape [ue_num, 3]

    def __generate_location_set(self):
        ## (ue_num, 3) -> (ue_num, 3, times_pilot)
        ue_loc = np.tile(np.expand_dims(self.ue_ini_loc, axis = -1), [1, 1, self.times_pilot])
        ue_spd = np.tile(np.expand_dims(self.ue_speed, axis = -1), [1, 1, self.times_pilot])
        ue_acc = np.tile(np.expand_dims(self.ue_acc, axis = -1), [1, 1, self.times_pilot])
        
        tick = np.linspace(0, self.times_pilot - 1, self.times_pilot)
        ## mat_A: (1, 1, times_pilot) -> (ue_num, 3, times_pilot)
        mat_A = np.tile(self.t_csi * tick.reshape(1, 1, -1), [self.ue_num, 3, 1])

        self.ue_loc = ue_loc + ue_spd * mat_A + ue_acc * 0.5 * mat_A ** 2           # [ue_num, 3, times_pilot]

        self.d_2D = np.sqrt(np.square(self.ue_loc[:, 0, :] - self.bs_loc[0]) + 
                            np.square(self.ue_loc[:, 1, :] - self.bs_loc[1]))       # [ue_num, times_pilot]
        self.d_3D = np.sqrt(np.square(self.d_2D) + 
                            np.square(self.ue_loc[:, 2, :] - self.bs_loc[2]))       # [ue_num, times_pilot]

    def __generate_ue_rotation_speed(self):
        ## 每个方向的最大角速度
        rot_speed_max = np.pi / config["UAV_Scenario_Config"]["rot_speed_max_div"]
        
        ## 高转速
        if np.random.rand() >= 0.5:
            rot_speed_alpha = np.random.uniform(low = rot_speed_max / 1.5, high = rot_speed_max, size = (self.ue_num, ))
        else:
            rot_speed_alpha = np.random.uniform(high = -rot_speed_max / 1.5, low = -rot_speed_max, size = (self.ue_num, ))
        
        if np.random.rand() >= 0.5:
            rot_speed_beta = np.random.uniform(low = rot_speed_max / 1.5, high = rot_speed_max, size = (self.ue_num, ))
        else:
            rot_speed_beta = np.random.uniform(high = -rot_speed_max / 1.5, low = -rot_speed_max, size = (self.ue_num, ))

        if np.random.rand() >= 0.5:
            rot_speed_gamma = np.random.uniform(low = rot_speed_max / 1.5, high = rot_speed_max, size = (self.ue_num, ))
        else:
            rot_speed_gamma = np.random.uniform(high = -rot_speed_max / 1.5, low = -rot_speed_max, size = (self.ue_num, ))
        
        self.rot_speed = np.vstack((rot_speed_alpha, rot_speed_beta, rot_speed_gamma)).T

        rot_acc_max = np.pi / config["UAV_Scenario_Config"]["rot_acc_max_div"]
        rot_acc_alpha = np.random.uniform(low = -rot_acc_max, high = rot_acc_max, size = (self.ue_num, ))
        rot_acc_beta = np.random.uniform(low = -rot_acc_max, high = rot_acc_max, size = (self.ue_num, ))
        rot_acc_gamma = np.random.uniform(low = -rot_acc_max, high = rot_acc_max, size = (self.ue_num, ))
        self.rot_acc = np.vstack((rot_acc_alpha, rot_acc_beta, rot_acc_gamma)).T


    def __update_ue_rotation_acc(self, user):   
        if np.random.rand() >= 0.5 or self.t == 0:
            rot_acc_max = np.pi / config["UAV_Scenario_Config"]["rot_acc_max_div"]
            rot_acc_alpha = np.random.uniform(low = -rot_acc_max, high = rot_acc_max)#, size = 0)
            rot_acc_beta = np.random.uniform(low = -rot_acc_max, high = rot_acc_max)#, size = 0)
            rot_acc_gamma = np.random.uniform(low = -rot_acc_max, high = rot_acc_max)#, size = 0)
        else:
            rot_acc_max = np.pi / config["UAV_Scenario_Config"]["rot_acc_max_div"] * 10
            if np.random.rand() >= 0.5:
                rot_acc_alpha = np.random.uniform(low = 0.8 * rot_acc_max, high = rot_acc_max)#, size = 0)
            else:
                rot_acc_alpha = np.random.uniform(low = -rot_acc_max, high = -0.8 * rot_acc_max)#, size = 0)

            if np.random.rand() >= 0.5:
                rot_acc_beta = np.random.uniform(low = 0.8 * rot_acc_max, high = rot_acc_max)#, size = 0)
            else:
                rot_acc_beta = np.random.uniform(low = -rot_acc_max, high = -0.8 * rot_acc_max)#, size = 0)

            if np.random.rand() >= 0.5:
                rot_acc_gamma = np.random.uniform(low = 0.8 * rot_acc_max, high = rot_acc_max)#, size = 0)
            else:
                rot_acc_gamma = np.random.uniform(low = -rot_acc_max, high = -0.8 * rot_acc_max)#, size = 0)
        
        self.rot_acc[user, 0] = rot_acc_alpha
        self.rot_acc[user, 1] = rot_acc_beta
        self.rot_acc[user, 2] = rot_acc_gamma
        
    def __update_path_loss(self):
        '''
        路径损耗，只和距离有关
        Consider urban macro UMa 
        LOS always exists.
        NLOS exists, and satisfies Possion distribution.
        LOS_flag: 'True' for LOS and 'False' for NLOS.
        following 3GPP TR 38.901 path_loss/penetration_loss 
        '''

        ue_height = self.ue_loc[:, -1, :]  # [ue_num, n_tic]
        bs_height = self.bs_loc[-1]

        h_e = 1.0 # m
        d_bp = 4 * (bs_height - h_e) * (ue_height - h_e) * self.f_c / self.c  # [ue_num, n_tic]
        for u in range(self.ue_num):
            if self.d_2D[u, self.t] <= d_bp[u, self.t]:
                self.pl[u, self.t] = 28 + 22 * np.log10(self.d_3D[u, self.t]) + 20 * np.log10(self.f_c / 1e9)
            elif self.d_2D > d_bp[u, self.t]:
                self.pl[u, self.t] = 28 + 40 * np.log10(self.d_3D[u, self.t]) + 20 * np.log10(self.f_c / 1e9) - 9 * np.log10(d_bp ** 2 + (bs_height - ue_height) ** 2)
        
    def __update_shadow_fading(self):
        if self.sf_prev is None:
            self.sf = self.__calculate_shadow_fading()
        else:
            self.__calculate_correlation_coff()
            self.sf = self.sf_prev * self.R_delta + np.sqrt(1. - self.R_delta ** 2) * self.__calculate_shadow_fading() / 40
        self.sf_prev = self.sf

    def __calculate_shadow_fading(self):
        # consider only LoS
        sigma_sf = 4        # 假定阴影衰落的标准差为4 dB
        sf = np.random.normal(loc = 0, scale = sigma_sf, size = (self.ue_num, self.cl_num, self.ray_num))
        # sf[abs(sf) > 5] = 0
        return sf

    def __calculate_correlation_coff(self):
        '''
        3GPP TR 38.901 Table 7.6.4.1-4
        '''
        d_corr = 10
        delta_d = np.linalg.norm(x = self.d_3D[:, self.t] - self.d_3D[:, self.t - 1], ord = 2)  # [ue_num]
        t_corr = d_corr / (delta_d / self.t_csi)  # [ue_num]
        self.R_delta = np.exp(-(delta_d / d_corr + self.t_csi / t_corr))
        # self.R_delta = np.exp(-(self.t_csi / 10))

    def __generate_cluster_number(self):
        self.ray_num = 10           # consider 20 rays in a cluster
        self.cl_num = 12

        ## 泊松分布生成簇的个数
        # self.lamd_cluster = 1.8     ## poisson分布的平均值参数
        # n_cl = np.random.poisson(self.lamd_cluster, size = self.ue_num)
        # n_cl[n_cl < 1] = 1          ## 确保一定会有一个簇
        # self.cl_num = n_cl          # number of clusters (ue_num, )
        # self.cl_num = np.ones((self.ue_num, ), dtype = np.int32) * 12       # UMa cluster num: 12 
        

    def __generate_K_factor(self):
        '''
        3GPP TR 38.901 Table 7.5-6 Part-1
        Rician K factor in dB
        '''
        mu_K = 9
        sigma_K = 3.5
        K = np.random.normal(loc = mu_K, scale = sigma_K)
        return K
    
    def __generate_delay_spread(self):
        mu_lgDS = -6.955 - 0.0963 * np.log10(self.f_c) 
        sigma_lgDS = 0.66
        DS = np.random.normal(loc = mu_lgDS, scale = sigma_lgDS)
        return DS 

    def __generate_cluster_delay(self, 
                                 r_tau, 
                                 DS, 
                                 K  
                                 ):
        X_n = np.random.uniform(low = 0, high = 1, size = (self.cl_num, ))
        tau_n_ = -r_tau * DS * np.log(X_n)     ## 升序排列
        tau_n = np.sort(tau_n_ - np.min(tau_n_))
        C_tau = 0.7705 - 0.0433 * K + 0.0002 * K ** 2 + 0.000017 * K ** 3
        tau_n /= C_tau       ## 对LOS的时延进行修正
        return tau_n

    def __generate_cluster_power_fraction(self):
        '''
        LOS always exists.
        K_R in linear value
        '''
        r_tau = 2.5
        zeta = 3
        
        self.p_cluster = list()         ## (ue_num, cl_num[i])    p_cluster[0]为los的功率分布, 叠加随机path的功率分布
        self.p_los_random = list()      ## (ue_num, cl_num[i])    随机生成cluster中的所有path的功率分布, 不叠加los
        self.p_los = list()             ## (ue_num)               los的功率分布

        for i in range(self.ue_num):
            DS = self.__generate_delay_spread()
            K = self.__generate_K_factor()
            Zn = np.random.normal(loc=0., scale=zeta, size=self.cl_num)
            tau_n = self.__generate_cluster_delay(r_tau, DS, K)
            
            # Pn = np.power(u, r_tau - 1) * np.power(10, -0.1 * Zn)
            Pn = np.exp(-tau_n * (r_tau - 1) / (r_tau * DS)) * np.power(10, -Zn / 10)

            regularization = np.sum(Pn)
            
            K_R = np.power(10, K / 10)
            P_Los = K_R / (K_R + 1)
            Pn = 1. / (K_R + 1) * Pn / regularization
            self.p_los_random.append(Pn)     # exclude the LOS
            Pn[0] = Pn[0] + P_Los
            self.p_los.append(P_Los)  # only include the LOS
            self.p_cluster.append(Pn)        # include the LOS

    def __update_ZOA_and_ZOD(self):
        '''
        generate both the ZOAs and ZODs of one cluster
        '''
        self.theta_zoa = self.__update_ZOA()
        self.theta_zod = self.__update_ZOD()

    def __update_ZOD(self):
        '''
        See TR 38.901 Table 7.5-7
        '''
        # zsd: Zero Spreading Delay
        mu_lg10_zsd = np.maximum([-0.5] * self.ue_num, -2.1 * self.d_2D[:, self.t] / 1000 - 0.01 * (self.ue_loc[:, -1, self.t] - 1.5) + 0.75)
        sigma_lg10_zsd = 0.4
        mu_zsd = 10 ** mu_lg10_zsd
        sigma_zsd = 10 ** sigma_lg10_zsd
        mu_offset_zoa = 0
        c_theta = 0.6
        alpha = 0.5129

        theta_zod_list = list()
        for u in range(self.ue_num):
            zsd = np.random.normal(loc = mu_zsd[u], scale = sigma_zsd)
            theta_zod = zsd * np.log(np.max(self.p_cluster[u]) / self.p_cluster[u]) / c_theta          
            random_sign = 2 * np.random.randint(1, size = self.cl_num) - 1
            y = np.random.normal(loc = 0, scale = np.abs(zsd / 7), size = (self.cl_num, self.ray_num))
            theta_zod = np.tile(random_sign * theta_zod, [self.ray_num, 1]).T
            theta_zod = theta_zod + y
            
            # ray offset
            theta_ray_offset = alpha * 3/8 * np.power(10., mu_lg10_zsd[u]) * \
                                np.random.lognormal(mean = mu_lg10_zsd[u], sigma = sigma_lg10_zsd, size = [self.cl_num, self.ray_num])
            theta_zod = theta_zod + mu_offset_zoa + theta_ray_offset
            
            theta_zod[0, :] = 0.                    # no zos in the LOS
            theta_zod = theta_zod / 180 * np.pi     # transform degree to radian; relateive to the LOS
            
            # the absolute radian
            theta_zod = theta_zod + np.tile(self.theta_zod_los[u, self.t], [self.cl_num, self.ray_num])
            theta_zod_list.append(theta_zod)
        return theta_zod_list
            
    def __update_ZOA(self):
        '''
        generate the ZOAs of one cluster
        '''
        mu_lg10_zsa = 0.95
        sigma_lg10_zsa = 0.16
        mu_zsa = 10**mu_lg10_zsa
        sigma_zsa = 10**sigma_lg10_zsa
        
        c_theta = 0.6
        c_zsa = 7 # degree
        alpha = 0.5129
        
        theta_zoa_list = list()
        for u in range(self.ue_num):
            zsa = np.random.normal(loc = mu_zsa, scale = sigma_zsa)
            theta_zoa = zsa*np.log(np.max(self.p_cluster[u])/self.p_cluster[u])/c_theta
            random_sign = 2*np.random.randint(1, size = self.cl_num)-1
            y = np.random.normal(loc = 0, scale = np.abs(zsa/7), size = (self.cl_num, self.ray_num))
            theta_zoa = np.tile(random_sign*theta_zoa, [self.ray_num, 1]).T
            theta_zoa = theta_zoa + y
            
            # ray offset
            theta_ray_offset = alpha*np.random.laplace(loc = 0, scale = c_zsa, \
                                                       size = [self.cl_num, self.ray_num])
            theta_zoa = theta_zoa+theta_ray_offset
            
            theta_zoa[0, :] = 0. # no zos in the LOS
            theta_zoa = theta_zoa/180*np.pi # transform degree to radian; relateive to the LOS
            
            # the absolute radian
            theta_zoa = theta_zoa + np.tile(self.theta_zoa_los[u, self.t], [self.cl_num, self.ray_num])
            theta_zoa_list.append(theta_zoa)
        return theta_zoa_list

    def __update_AOA_and_AOD(self):
        '''
        generate both the AOAs and AODs of one cluster
        '''
        self.phi_aoa = self.__update_azimuth_angle(self.phi_aoa_los[:, self.t])
        self.phi_aod = self.__update_azimuth_angle(self.phi_aod_los[:, self.t])

    def __update_azimuth_angle(self, phi_los):
        mu_lg10_asa = 1.81
        sigma_lg10_asa = 0.2
        mu_asa = 10**mu_lg10_asa
        sigma_asa = 10**sigma_lg10_asa

        c_phi = 0.7
        c_asa = 11 # degree
        alpha = 0.5129
        
        phi_list = list()
        for u in range(self.ue_num):
            asa = np.random.normal(loc = mu_asa, scale = sigma_asa)
            phi = 2*(asa/1.4)*np.sqrt(np.log(np.max(self.p_cluster[u])/self.p_cluster[u]))/c_phi
            random_sign = 2*np.random.randint(1, size = self.cl_num)-1
            y = np.random.normal(loc = 0, scale = np.abs(asa/7), size = (self.cl_num, self.ray_num))
            phi = np.tile(random_sign*phi, [self.ray_num, 1]).T
            phi = phi+y
            
            # ray offset
            phi_ray_offset = alpha*np.random.laplace(loc = 0, scale = c_asa, size = [self.cl_num, self.ray_num])
            phi = phi+phi_ray_offset
            
            phi[0, :] = 0. # no aos in the LOS
            phi = phi / 180 * np.pi # transform degree to radian; relateive to the LOS
            
            # the absolute radian
            phi = phi + np.tile(phi_los[u], [self.cl_num, self.ray_num])
            phi_list.append(phi)
        return phi_list

    def __update_angles(self):
        ## 更新当前时隙self.t的los角度
        bs_loc = np.tile(np.expand_dims(self.bs_loc, axis = 0), [self.ue_num, 1]) # shape [ue_num, 3]
        ue2bs_pos = self.ue_loc[:, :, self.t] - bs_loc

        for u in range(self.ue_num):
            # cartesian_to_polar -> d_3D, d_2D, theta, phi
            
            self.theta_zod_los[u, self.t], self.phi_aod_los[u, self.t], \
            self.theta_zoa_los[u, self.t], self.phi_aoa_los[u, self.t] = \
            utils.calculate_los_angle(ue2bs_pos[u, :], self.ue_rot[u, :, self.t])

    def update_cross_polarization_power_ratio(self):
        '''
        See TR 38.901 Table 7.5-6
        函数 update_cross_polarization_power_ratio 用于更新交叉极化功率比(Cross Polarization Power Ratio, XPR)。
        这个比率是无线通信中描述天线性能的一个重要参数，特别是在使用双极化天线时。
        交叉极化功率比描述了一个天线在正交极化平面上接收或发送信号的能力，通常用来衡量天线对不同极化信号的隔离度。
        '''
        mu_XPR = 8 # dB
        sigma_XPR = 4 # dB
        mu_XPR = np.power(10., mu_XPR / 10.)
        sigma_XPR = np.power(10., sigma_XPR / 10.)
        XPR_list = list()
        for u in range(self.ue_num):
            X = np.random.randn(self.cl_num, self.ray_num) * sigma_XPR + mu_XPR
            XPR = np.power(10., X/10.)
            XPR_list.append(XPR)
        # XPR_list: (ue_num, cl_num[u], ray_num) )
        return XPR_list
        
    def __update_cluster_gain(self):
        '''
        generate the amplitudes of one cluster
        consider path loss, penetration loss, shadow fading, small-scale fading (rician fading), doppler effect
        '''
        ## cluster_gain: (ue_num, cl_num[i], ray_num)
        self.cluster_gain = list()
        for u in range(self.ue_num):
            ## p_los_random:    (ue_num, cl_num[i], ray_num)            随机生成cluster中的所有path的功率分布, 不叠加los
            pf_random = np.tile(self.p_los_random[u] / self.ray_num, [self.ray_num, 1]).T
            ## p_los:           (ue_num, cl_num[i], ray_num)            los的功率分布
            pf_deterministic = np.sqrt(self.p_los[u])
            ## self.pl: ndarray: (ue_num, times_pilot)
            pl = np.tile(self.pl[u, self.t], [self.cl_num, self.ray_num]) + self.sf[u, :, :]

            ## gain: ndarray: (cl_num[u], ray_num)
            gain = np.sqrt(pf_random) #* utils.generate_complex_gaussian(size = pl.shape)

            gain[0, :] += pf_deterministic

            # path loss is the power loss.
            gain = gain * np.power(10, -pl / 20)

            self.cluster_gain.append(gain)

    ## Multi Processing generating channel matrix h
    def __update_channel_response_multiple_processing(self):
        XPR = self.update_cross_polarization_power_ratio()
        pool = multiprocessing.Pool(processes = 20)
        results = [pool.apply_async(update_channel_response_for_user, args=(self, XPR, user)) for user in range(self.ue_num)]
        pool.close()
        pool.join
        h_t_results = [result.get() for result in results]
        h_t = np.concatenate(h_t_results, axis = 0)
        print("Multi Process: CSI period-%d's channel has been generated" % (self.t + 1))
        return h_t
        
    # Single Processing generating channel matrix h
    def __update_channel_response_single_processing(self):
        '''
        generate the channel matrix of one instant
        '''
        ## XPR_list: (ue_num, cl_num[u], ray_num) )
        XPR = self.update_cross_polarization_power_ratio()
        h_t = np.zeros((self.ue_num, self.Nr, self.Nt), dtype = np.complex64)
        for u in tqdm(range(self.ue_num), desc = "generating the channel of CSI period-%d" % (self.t)):
            for ncl in range(self.cl_num): #  
                for nray in range(self.ray_num): # 
                    if ncl == 0:
                        xpr = 10000.        ## 假设LoS路径的XPR非常高
                    else:
                        xpr = XPR[u][ncl][nray]

                    cl_gain = self.cluster_gain[u][ncl][nray]
                    phi_aod = self.phi_aod[u][ncl][nray]
                    theta_zod = self.theta_zod[u][ncl][nray]
                    phi_aoa = self.phi_aoa[u][ncl][nray]
                    theta_zoa = self.theta_zoa[u][ncl][nray]
                    at = self.generate_array_response(phi_aod, theta_zod, phi_aoa, theta_zoa)

                    gain = np.sqrt(self.Nt * self.Nr / self.ray_num) * np.square(1. + 1. / np.sqrt(xpr)) * cl_gain
                    # gain = np.sqrt(self.Nt * self.Nr / self.ray_num) * cl_gain
                    h_t[u, :, :] += gain * at
        return h_t

    def generate_array_response(self, aod, zod, aoa, zoa):
        ## 摆于X-Y平面的UPA阵列

        # generate transmitter phase array
        z_mat = np.tile(np.linspace(0, self.bs_z_antenna_num - 1, self.bs_z_antenna_num), [self.bs_y_antenna_num, 1]).T - self.bs_z_antenna_num // 2
        y_mat = np.tile(np.linspace(0, self.bs_y_antenna_num - 1, self.bs_y_antenna_num), [self.bs_z_antenna_num, 1]) - self.bs_y_antenna_num // 2
        
        phase_array = y_mat * np.sin(aod) * np.sin(zod) + z_mat * np.cos(zod)

        a_tx = 1. / np.sqrt(self.Nt) * np.exp(1j * self.kd * phase_array)
        a_tx = np.reshape(a_tx.T, [self.Nt, 1])
        
        # generate receiver phase array
        z_mat = np.tile(np.linspace(0, self.ue_z_antenna_num - 1, self.ue_z_antenna_num), [self.ue_y_antenna_num, 1]).T - self.ue_z_antenna_num // 2
        y_mat = np.tile(np.linspace(0, self.ue_y_antenna_num - 1, self.ue_y_antenna_num), [self.ue_z_antenna_num, 1]) - self.ue_y_antenna_num // 2
        
        phase_array = y_mat * np.sin(aoa) * np.sin(zoa) + z_mat * np.cos(zoa)

        a_rx = 1. / np.sqrt(self.Nr) * np.exp(1j * self.kd * phase_array)
        a_rx = np.reshape(a_rx.T, [self.Nr, 1])
        
        A = np.matmul(a_rx, np.conjugate(a_tx).T)   ## A: (self.Nr, self.Nt)
        return A
    
    def __initialize_environment(self):
        self.sf_prev = None
        self.pl = np.zeros((self.ue_num, self.times_pilot), dtype = ftype)

        self.__generate_bs_location()                   ## self.bs_loc:     (3, )
        self.__generate_initial_ue_location()           ## self.ue_ini_loc: (ue_num, 3)
        self.__generate_initial_ue_rotation()           ## self.ue_rot:     (ue_num, 3, self.times_pilot) 并更新第0个时隙的
        self.__generate_initial_ue_speed()              ## self.ue_speed:   (ue_num, 3)
                                                        ## self.ue_acc:     (ue_num, 3)
        self.__generate_cluster_number()                ## self.ray_num
                                                        ## self.cl_num:     (ue_num, )
        self.__generate_cluster_power_fraction()        ## self.p_cluster, self.p_los_random, self.p_los
        self.__calculate_overall_noise()
        self.__generate_location_set()                  ## self.ue_loc      (ue_num, n_tic)
        self.__generate_ue_rotation_speed()

        self.phi_aoa_los = np.zeros((self.ue_num, self.times_pilot), dtype = ftype)
        self.phi_aod_los = np.zeros((self.ue_num, self.times_pilot), dtype = ftype)
        self.theta_zoa_los = np.zeros((self.ue_num, self.times_pilot), dtype = ftype)
        self.theta_zod_los = np.zeros((self.ue_num, self.times_pilot), dtype = ftype)
        
    def __calculate_overall_noise(self):
        noise_figure = self.ue_rx_noise_figure
        sigma2_dBm = self.sigma2_dBm + 10 * np.log10(config["UAV_Scenario_Config"]["band_width"] * 1e6) + noise_figure 
        sigma2 = sigma2_dBm - 30
        self.sigma2 = np.power(10, sigma2 / 10)
        return sigma2_dBm

    ## 每隔10个CSI周期track一次
    def test(self, 
            model_type = "Linear",
            ):
        self.ue_num = config["Test_Config"]["ue_num"]
        track_period_baseline = config["Test_Config"]["track_period_baseline"]
        self.times_pilot = config["Test_Config"]["num_csi"]
        threshold = config["Test_Config"]["retrack_SNR_threshold"]
        self.__initialize_environment()
        print("start UAV testing...")

        window_input = config["Dataset_Config"]["window_input"]
        window_output = config["Dataset_Config"]["window_output"]

        bs_search_r = 1
        ue_search_r = 1

        ## 每个user当前跟踪周期内使用的波束对(基站波束, 终端波束)
        beam_track_baseline = [[None, None] for _ in range(self.ue_num)]     ## 跟踪的波束 (bs_beam, ue_beam)
        beam_track_model = [[None, None] for _ in range(self.ue_num)] 

        stage_baseline = np.zeros((self.ue_num, self.times_pilot), dtype = np.int16)
        stage_t_baseline = np.zeros((self.ue_num), dtype = np.int16)
        stage_model = np.zeros((self.ue_num, self.times_pilot), dtype = np.int16)
        stage_t_model = np.zeros((self.ue_num), dtype = np.int16)
        
        codebook = Dft_codebook(self.bs_z_antenna_num, self.bs_y_antenna_num, 
                                self.ue_z_antenna_num, self.ue_y_antenna_num)
        
        model = model_dict[model_type]["predictor"].to(device)
        model.load()

        SNR_input_window = np.zeros((self.ue_num, window_input), dtype = np.float32)
        delta_pos_window = np.zeros((self.ue_num, window_input, 3), dtype = np.float32)
        ue_rot_window = np.zeros((self.ue_num, window_input, 3), dtype = np.float32)

        SNR_baseline = np.zeros((self.ue_num, self.times_pilot))
        SNR_model = np.zeros((self.ue_num, self.times_pilot))

        count_baseline = np.zeros((self.ue_num), dtype = np.int16)  ## baseline CSI计数器，累计10次track一次
        count_model = np.zeros((self.ue_num), dtype = np.int16)

        prediction_start_t = window_input * np.ones((self.ue_num), dtype = np.int16)
        track_start_t = np.zeros((self.ue_num), dtype = np.int16)

        track_point_baseline = [[] for _ in range(self.ue_num)]
        track_point_model = [[] for _ in range(self.ue_num)]

        delta_expense = 1#(1 + bs_search_r) ** 2 * (1 + ue_search_r) ** 2
        track_expense = {
            "baseline":{},
            "model":{}
        }

        count_rot = np.zeros((self.ue_num), dtype = np.int16)
        for self.t in range(self.times_pilot):
            for user in range(self.ue_num):
                if count_rot[user] == 0:
                    self.__update_ue_rotation_acc(user)
                    count_rot[user] = np.random.randint(low = 80, high = 150)

            self.__update_path_loss()
            self.__update_shadow_fading()
            self.__update_ue_rotation()
            self.__update_angles()
            self.__update_AOA_and_AOD()
            self.__update_ZOA_and_ZOD()
            self.__update_cluster_gain()
            # h_t = self.__update_channel_response_single_processing()
            h_t = self.__update_channel_response_multiple_processing()

            count_rot -= 1

            delta_pos = self.ue_loc[:, :, self.t] - np.tile(self.bs_loc[None, :], (self.ue_num, 1))
            ue_rot = self.ue_rot[:, :, self.t]
            
            SNR_input_window[:, 0:-1] = SNR_input_window[:, 1:]
            ue_rot_window[:, 0:-1, :] = ue_rot_window[:, 1:, :]
            ue_rot_window[:, -1, :] = ue_rot
            delta_pos_window[:, 0:-1, :] = delta_pos_window[:, 1:, :]
            delta_pos_window[:, -1, :] = delta_pos
            
            for user in range(self.ue_num):
                ## Baseline: 每隔track_period_baseline个CSI周期track一次
                if beam_track_baseline[user][0] == None and beam_track_baseline[user][1] == None:
                    beam_track_baseline[user][0], beam_track_baseline[user][1], _ = initial_access_3(h_t[user], delta_pos[user], 
                                                                                                     ue_rot[user], codebook, bs_search_r, ue_search_r)
                    stage_t_baseline[user] += 1
                    track_point_baseline[user].append(self.t)
                    if len(track_point_baseline[user]) > 1:
                        track_t_now = self.t
                        track_t_last = track_point_baseline[user][-2]
                        track_period = track_t_now - track_t_last   ## 此次跟踪周期
                        if track_period not in track_expense["baseline"]:   
                            track_expense["baseline"][track_period] = delta_expense
                        else:
                            track_expense["baseline"][track_period] += delta_expense

                snr_baseline = utils.calculate_SNR(codebook, h_t[user], beam_track_baseline[user][0], beam_track_baseline[user][1])
                SNR_baseline[user, self.t] = snr_baseline

                count_baseline[user] += 1
                stage_baseline[user, self.t] = stage_t_baseline[user]
                if count_baseline[user] >= track_period_baseline:# or snr_baseline < threshold:
                    count_baseline[user] = 0
                    beam_track_baseline[user][0] = None
                    beam_track_baseline[user][1] = None
                    
                ## Data-driven Track
                # if beam_track_model[user][0] == None and beam_track_model[user][1] == None:
                if self.t == track_start_t[user]:
                    beam_track_model[user][0], beam_track_model[user][1], _ = initial_access_3(h_t[user], delta_pos[user], 
                                                                                               ue_rot[user], codebook, bs_search_r, ue_search_r)
                    stage_t_model[user] += 1
                    track_point_model[user].append(self.t)
                    if len(track_point_model[user]) > 1:
                        track_t_now = track_point_model[user][-1]
                        track_t_last = track_point_model[user][-2]
                        track_period = track_t_now - track_t_last   ## 此次跟踪周期
                        if track_period not in track_expense["model"]:   
                            track_expense["model"][track_period] = delta_expense
                        else:
                            track_expense["model"][track_period] += delta_expense

                snr_model = utils.calculate_SNR(codebook, h_t[user], beam_track_model[user][0], beam_track_model[user][1])
                SNR_model[user, self.t] = snr_model

                count_model[user] += 1
                stage_model[user, self.t] = stage_t_model[user]
                SNR_input_window[user, -1] = snr_model

                if self.t == prediction_start_t[user] and count_model[user] >= window_input:
                    delta_pos_diff = delta_pos_window[user, :, :] / np.reshape(delta_pos_window[user, 0, :], (1, 3)) - 1
                    ue_rot_diff = ue_rot_window[user, :, :] / np.reshape(ue_rot_window[user, 0, :], (1, 3)) - 1
                    SNR_base = SNR_input_window[user, 0]
                    SNR_input_diff = SNR_input_window[user, :] / SNR_base - 1

                    delta_pos_diff = torch.tensor(delta_pos_diff, device = device).unsqueeze(0)
                    ue_rot_diff = torch.tensor(ue_rot_diff, device = device).unsqueeze(0)
                    SNR_input_diff = torch.tensor(SNR_input_diff, device = device).unsqueeze(0)
                    SNR_output_diff, var_output = model(SNR_input_diff, delta_pos_diff, ue_rot_diff)

                    SNR_output_diff = SNR_output_diff.squeeze()
                    var_output = var_output.squeeze()

                    snr_hat = (SNR_output_diff + 1) * SNR_base

                    ## SNR之间的差如果过大的话, 时间预测的间隔需要减小
                    max_delta_SNR = np.min(SNR_input_window[user, 1:] - SNR_input_window[user, 0:-1])
                    if max_delta_SNR < -0.5:
                        interval_prediction = 1
                    else:
                        interval_prediction = window_output

                    for t in range(window_output):
                        if snr_hat[t] - 2.326 * var_output[t] < threshold:
                            track_start_t[user] = self.t + t + 1
                            prediction_start_t[user] = track_start_t[user] + window_input - 1
                            break
                        if t == window_output - 1:
                            prediction_start_t[user] = self.t + interval_prediction
      
        dataset_pth = config["Test_Config"]["dataset_pth"]
        test_dataset_h5 = h5py.File(dataset_pth, "w")
        test_dataset_h5.create_dataset("times_pilot", data = self.times_pilot)
        test_dataset_h5.create_dataset("ue_num", data = self.ue_num)

        ## 存储baseline信息
        group_baseline = test_dataset_h5.create_group("baseline")
        for user in range(self.ue_num):
            group_user = group_baseline.create_group("user_%d" % user)
            group_user.create_dataset("SNR", data = SNR_baseline[user])
            group_user.create_dataset("stage", data = stage_baseline[user])
            group_user.create_dataset("track_point", data = np.array(track_point_baseline[user]), dtype = np.int16)  

        expense_baseline = track_expense["baseline"]
        track_period_list = expense_baseline.keys()
        length = max(track_period_list) + 1
        expense = np.zeros((length, ), dtype = np.int32)
        for track_period in track_period_list:
            expense[track_period] = expense_baseline[track_period]
        group_baseline.create_dataset("track_expense", data = expense)
        
        ## 存储data_driven
        group_model = test_dataset_h5.create_group("data_driven")
        for user in range(self.ue_num):
            group_user = group_model.create_group("user_%d" % user)
            group_user.create_dataset("SNR", data = SNR_model[user])
            group_user.create_dataset("stage", data = stage_model[user])
            group_user.create_dataset("track_point", data = np.array(track_point_model[user]), dtype = np.int16)  

        expense_model = track_expense["model"]
        track_period_list = expense_model.keys()
        length = max(track_period_list) + 1
        expense = np.zeros((length, ), dtype = np.int32)
        for track_period in track_period_list:
            expense[track_period] = expense_model[track_period]
        group_model.create_dataset("track_expense", data = expense)

        test_dataset_h5.close()

def plot_track_curve(user = 0):
    dataset_pth = config["Test_Config"]["dataset_pth"]
    threshold = config["Test_Config"]["retrack_SNR_threshold"]
    test_dataset_h5 = h5py.File(dataset_pth, "r")

    times_pilot = test_dataset_h5["times_pilot"][()]

    SNR_baseline = test_dataset_h5["baseline"]["user_%d" % user]["SNR"][:]
    stage_baseline = test_dataset_h5["baseline"]["user_%d" % user]["stage"][:]
    track_point_baseline = test_dataset_h5["baseline"]["user_%d" % user]["track_point"][:]

    SNR_model = test_dataset_h5["data_driven"]["user_%d" % user]["SNR"][:]
    stage_model = test_dataset_h5["data_driven"]["user_%d" % user]["stage"][:]
    track_point_model = test_dataset_h5["data_driven"]["user_%d" % user]["track_point"][:]

    t_csi = config["UAV_Scenario_Config"]["t_csi"] * 1000

    test_dataset_h5.close()

    from matplotlib import pyplot as plt
    plt.rcParams['font.sans-serif'] = "Times New Roman"

    # draw th baseline track curve
    plt.figure(figsize=(18, 15), dpi=100)
    # plt.title("User-%d" % user)
    plt.subplot(211)
    start_t = 0
    for t in range(1, times_pilot):
        if t == times_pilot - 1:
            label = "SNR curve"
        else:
            label = None
        if t == times_pilot - 1 or (t + 1 <= times_pilot - 1 and stage_baseline[t] != stage_baseline[t + 1]):
            plt.plot(np.arange(start_t, t + 1) * t_csi, SNR_baseline[start_t : t + 1], marker = 'o', markersize = 2,
                     label = label, color = 'orange')
            start_t = t + 1
    plt.plot(track_point_baseline * t_csi, SNR_baseline[track_point_baseline], linestyle = '', marker = 'o', 
             color = 'b', label = 'track point')
    # draw the threshold line
    plt.plot(np.arange(times_pilot) * t_csi, threshold * np.ones((times_pilot)), '--', color = 'gray', label = "track threshold")
    # mark the low Qos point
    low_Qos_point_baseline = []
    for t in range(times_pilot):
        if SNR_baseline[t] < threshold:
            low_Qos_point_baseline.append(t)
    low_Qos_point_baseline = np.array(low_Qos_point_baseline, dtype = np.int16)
    plt.plot(low_Qos_point_baseline * t_csi, SNR_baseline[low_Qos_point_baseline], 
                linestyle = '', marker = 'x', color = 'r', label = 'low QoS point')

    plt.ylabel("SNR (dB)", fontsize = 12)
    plt.xlabel("t (ms)", fontsize = 12)
    plt.grid(True, linestyle='--')
    plt.axis([-10, times_pilot * t_csi + 10, min(35, np.min(SNR_baseline) - 1), max(70, np.max(SNR_baseline) + 1)])
    plt.legend()
    plt.title("SNR Curve of Baseline Tracking(user-%d)" % user, fontsize = 15)

    ## draw Data-driven track curve
    plt.subplot(212)
    start_t = 0
    for t in range(1, times_pilot):
        if t == times_pilot - 1:
            label = "SNR curve"
        else:
            label = None
        if t == times_pilot - 1 or (t + 1 <= times_pilot - 1 and stage_model[t] != stage_model[t + 1]):
            plt.plot(np.arange(start_t, t + 1) * t_csi, SNR_model[start_t : t + 1], marker = 'o', markersize = 2,
                        label = label, color = 'green')
            start_t = t + 1
    plt.plot(track_point_model * t_csi, SNR_model[track_point_model], linestyle = '', marker = 'o', 
                color = 'b', label = 'track point')
    # draw the threshold line
    plt.plot(np.arange(times_pilot) * t_csi, threshold * np.ones((times_pilot)), '--', color = 'gray', label = "track threshold")
    # mark the low Qos point
    low_Qos_point_model = []
    for t in range(times_pilot):
        if SNR_model[t] < threshold:
            low_Qos_point_model.append(t)
    low_Qos_point_model = np.array(low_Qos_point_model, dtype = np.int16)
    plt.plot(low_Qos_point_model * t_csi, SNR_model[low_Qos_point_model], 
                linestyle = '', marker = 'x', color = 'r', label = 'low QoS point')
    plt.ylabel("SNR (dB)", fontsize = 12)
    plt.xlabel("t (ms)", fontsize = 12)
    plt.grid(True, linestyle='--')
    plt.axis([-10, times_pilot * t_csi + 10, min(35, np.min(SNR_model) - 1), max(70, np.max(SNR_model) + 1)])
    plt.legend()
    plt.title("SNR Curve of Data-driven Tracking(user-%d)" % user, fontsize = 15)

    plt.show()

def plot_expense_bar():
    dataset_pth = config["Test_Config"]["dataset_pth"]
    test_dataset_h5 = h5py.File(dataset_pth, "r")
    expense_baseline = test_dataset_h5["baseline"]["track_expense"][:]
    expense_model = test_dataset_h5["data_driven"]["track_expense"][:]
    test_dataset_h5.close()

    from matplotlib import pyplot as plt
    plt.rcParams['font.sans-serif'] = "Times New Roman"

    plt.figure(figsize=(15, 7), dpi = 100)
    plt.subplot(121)
    plt.bar(np.arange(expense_baseline.shape[0]), expense_baseline)
    plt.xlabel("track period (number of CSI period)", fontsize = 12)
    plt.ylabel("sum of expense", fontsize = 12)
    plt.grid(True, linestyle='--')
    plt.title("Track Expense Distribution of Baseline Tracking", fontsize = 15)

    plt.subplot(122)
    plt.bar(np.arange(expense_model.shape[0]), expense_model)
    plt.xlabel("track period (number of CSI period)", fontsize = 12)
    plt.ylabel("sum of expense", fontsize = 12)
    plt.grid(True, linestyle='--')
    plt.title("Track Expense Distribution of Data-driven Tracking", fontsize = 15)

    plt.show()


def update_channel_response_for_user(env, 
                                     XPR,
                                     user
                                     ):
    
    ## XPR_list: (ue_num, cl_num[u], ray_num) )
    h_t = np.zeros((1, env.Nr, env.Nt), dtype = np.complex64)
    for ncl in range(env.cl_num): #  
        for nray in range(env.ray_num): # 
            if ncl == 0:
                xpr = 10000.        ## 假设LoS路径的XPR非常高
            else:
                xpr = XPR[user][ncl][nray]
            cl_gain = env.cluster_gain[user][ncl][nray]
            phi_aod = env.phi_aod[user][ncl][nray]
            theta_zod = env.theta_zod[user][ncl][nray]
            phi_aoa = env.phi_aoa[user][ncl][nray]
            theta_zoa = env.theta_zoa[user][ncl][nray]
            at = env.generate_array_response(phi_aod, theta_zod, phi_aoa, theta_zoa)
            gain = np.sqrt(env.Nt * env.Nr / env.ray_num) * np.square(1. + 1. / np.sqrt(xpr)) * cl_gain
            h_t[0, :, :] += gain * at
    if user % 50 == 0:
        print("user-%d's channel has been generated" % user)
    return h_t

                            
if __name__ == "__main__":
    ## 生成场景
    # env = UAV_environment()
    # env.test(model_type="Linear")

    ## 绘图
    user = 117
    # for user in range(300):
    plot_track_curve(user = user)
    plot_expense_bar()

    