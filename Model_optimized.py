import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

column_num = 18
step = 100
delta = 0.001
z = 4

def Traindata(name_list, if_nor=True):
    np_D = np.zeros((1, column_num))
    for i in range(len(name_list)):
        dict_obj = io.loadmat(name_list[i])
        temp = dict_obj['ae_D']
        np_D = np.vstack((np_D, temp))
    np_D = np.delete(np_D, 0, axis=0)
    np_D = np_D[:, 4:]
    index = np.where(np_D[:, 3] < 10)[0]
    np_D = np.delete(np_D, index, axis=0)
    np_Dmax, np_Dmin = np_D.max(axis=0), np_D.min(axis=0)
    if if_nor:
        np_D = (np_D - np_Dmin) / (np_Dmax - np_Dmin)
        print('已归一化的训练集，大小为：', np_D.shape)
        return np_D, np_Dmax, np_Dmin
    else:
        print('未归一化的训练集，大小为：', np_D.shape)
        return np_D, np_Dmax, np_Dmin

def Testdata(name_string, np_Dmax, np_Dmin, if_nor=True):
    dict_obj = io.loadmat(name_string)
    np_Kobs = dict_obj['ae_Kobs2']
    np_Kobs = np_Kobs[:, 4:]
    if if_nor:
        np_Kobs = (np_Kobs - np_Dmin) / (np_Dmax - np_Dmin)
        return np_Kobs
    else:
        return np_Kobs

def Faultdata(name_string, np_Dmax, np_Dmin, if_nor=True):
    dict_obj = io.loadmat(name_string)
    np_Kobs = dict_obj['ae_ver_temp']
    np_Kobs = np_Kobs[:, 4:]
    if if_nor:
        np_Kobs = (np_Kobs - np_Dmin) / (np_Dmax - np_Dmin)
        return np_Kobs
    else:
        return np_Kobs

def normalization(np_Kobs, np_Dmax, np_Dmin):
    np_Kobs = (np_Kobs - np_Dmin) / (np_Dmax - np_Dmin)
    return np_Kobs

def MemoryMat_train(np_D, memorymat_name):
    memorymat = np.zeros((1, np_D.shape[1]))
    for i in range(np_D.shape[1]):
        for k in range(step):
            diff = np.abs(np_D[:, i] - k * (1 / step))
            idx = np.where(diff < delta)[0]
            if idx.size > 0:
                memorymat = np.vstack((memorymat, np_D[idx[0]]))
    memorymat = np.delete(memorymat, 0, axis=0)
    print('memorymat:', memorymat.shape)
    np.save(memorymat_name, memorymat)
    return memorymat

def MemoryMats_train(np_D):
    np_D1 = np.zeros((1, np_D.shape[1]))
    np_D2 = np.zeros((1, np_D.shape[1]))
    np_D3 = np.zeros((1, np_D.shape[1]))
    col_D = np_D.shape[1]
    thres1 = 1 / 3
    thres2 = 2 / 3
    for t in range(np_D.shape[0]):
        if np_D[t, col_D - 1] < thres1:
            np_D1 = np.vstack((np_D1, np_D[t]))
        elif np_D[t, col_D - 1] > thres2:
            np_D3 = np.vstack((np_D3, np_D[t]))
        else:
            np_D2 = np.vstack((np_D2, np_D[t]))
    np_D1 = np.delete(np_D1, 0, axis=0)
    np_D2 = np.delete(np_D2, 0, axis=0)
    np_D3 = np.delete(np_D3, 0, axis=0)
    print('D1,D2,D3:', np_D1.shape, np_D2.shape, np_D3.shape)
    def build_memorymat(np_Dx):
        memorymat = np.zeros((1, np_Dx.shape[1]))
        for i in range(np_Dx.shape[1]):
            for k in range(step):
                diff = np.abs(np_Dx[:, i] - k * (1 / step))
                idx = np.where(diff < delta)[0]
                if idx.size > 0:
                    memorymat = np.vstack((memorymat, np_Dx[idx[0]]))
        memorymat = np.delete(memorymat, 0, axis=0)
        return memorymat
    memorymat1 = build_memorymat(np_D1)
    print('memorymat1:', memorymat1.shape)
    memorymat2 = build_memorymat(np_D2)
    print('memorymat2:', memorymat2.shape)
    memorymat3 = build_memorymat(np_D3)
    print('memorymat3:', memorymat3.shape)
    return memorymat1, memorymat2, memorymat3

def Temp_MemMat(memorymat, Temp_name):
    # 批量欧氏距离计算
    diff = memorymat[:, np.newaxis, :] - memorymat[np.newaxis, :, :]
    Temp = np.linalg.norm(diff, axis=2)
    np.save(Temp_name, Temp)

def MSET_batch(memorymat_name, Kobs, Temp_name):
    """
    批量MSET计算，Kobs为(batch, feature)
    """
    memorymat = np.load(memorymat_name)
    Temp = np.load(Temp_name)
    # 计算 memorymat 与 Kobs 的距离矩阵 (memorymat_row, Kobs_row)
    diff = memorymat[:, np.newaxis, :] - Kobs[np.newaxis, :, :]
    Temp1 = np.linalg.norm(diff, axis=2)
    # Kest = (memorymat.T @ pinv(Temp)) @ Temp1
    Kest = np.dot(np.dot(memorymat.T, np.linalg.pinv(Temp)), Temp1)
    Kest = Kest.T
    return Kest

def MSETs_batch(memorymat1_name, memorymat2_name, memorymat3_name, Kobs):
    """
    批量MSETs计算，Kobs为(batch, feature)
    """
    row_Kobs = Kobs.shape[0]
    col_Kobs = Kobs.shape[1]
    Kest = np.zeros((row_Kobs, col_Kobs))
    idx_low = np.where(Kobs[:, col_Kobs - 1] < 1 / 3)[0]
    idx_high = np.where(Kobs[:, col_Kobs - 1] > 2 / 3)[0]
    idx_med = np.where((Kobs[:, col_Kobs - 1] >= 1 / 3) & (Kobs[:, col_Kobs - 1] <= 2 / 3))[0]
    if len(idx_low) > 0:
        Kest[idx_low] = MSET_batch(memorymat1_name, Kobs[idx_low], 'Temp_low.npy')
    if len(idx_med) > 0:
        Kest[idx_med] = MSET_batch(memorymat2_name, Kobs[idx_med], 'Temp_med.npy')
    if len(idx_high) > 0:
        Kest[idx_high] = MSET_batch(memorymat3_name, Kobs[idx_high], 'Temp_hig.npy')
    return Kest

def Cal_sim(Kobs, Kest):
    # 向量化相似度计算
    dist_norm = np.linalg.norm(Kobs - Kest, axis=1, keepdims=True)
    dot = np.sum(Kobs * Kest, axis=1, keepdims=True)
    norm_prod = np.linalg.norm(Kobs, axis=1, keepdims=True) * np.linalg.norm(Kest, axis=1, keepdims=True)
    dist_cos = dot / (norm_prod + 1e-8)
    dist_cos = dist_cos * 0.5 + 0.5
    sim = (1 / (1 + dist_norm / (dist_cos + 1e-8)))
    return sim

def Cal_thres(sim):
    mu = np.zeros((sim.shape[0], 1))
    sigma = np.zeros((sim.shape[0], 1))
    index = np.empty((1,), dtype=int)
    for i in range(sim.shape[0]):
        if i == 0:
            mu[i] = sim[i]
        else:
            if sim[i - 1] >= (mu[i - 1] - z * sigma[i - 1]) and sim[i - 1] >= 0.8:
                mu[i] = 1 / (i + 1) * sim[i] + i / (i + 1) * sim[i - 1]
                sigma[i] = np.sqrt((i - 1) / i * (sigma[i - 1] ** 2) + ((sim[i] - mu[i - 1]) ** 2 / (i + 1)))
            elif sim[i - 1] < (mu[i - 1] - z * sigma[i - 1]) or \
                    (sim[i - 1] >= (mu[i - 1] - z * sigma[i - 1]) and sim[i - 1] < 0.8):
                mu[i] = mu[i - 1]
                sigma[i] = sigma[i - 1]
                index = np.append(index, i)
    index = np.delete(index, 0)
    thres = mu - z * sigma
    return thres, index

# 其它可视化和误差分析函数可直接复用原有实现 