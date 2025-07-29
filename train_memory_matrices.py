"""
记忆矩阵训练脚本
用于生成MSET异常检测所需的记忆矩阵和临时矩阵文件
"""

import numpy as np
import os
import logging
from MSET_python.Model_optimized import MemoryMats_train, Temp_MemMat

def train_memory_matrices(training_data, output_dir='.'):
    """
    训练记忆矩阵
    
    参数:
        training_data: 训练数据，形状为 (N, features)
        output_dir: 输出目录
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("开始训练记忆矩阵...")
    print(f"训练数据形状: {training_data.shape}")
    
    # 训练三个记忆矩阵（低、中、高负荷）
    memorymat1, memorymat2, memorymat3 = MemoryMats_train(training_data)
    
    # 保存记忆矩阵
    np.save(os.path.join(output_dir, 'memorymat1.npy'), memorymat1)
    np.save(os.path.join(output_dir, 'memorymat2.npy'), memorymat2)
    np.save(os.path.join(output_dir, 'memorymat3.npy'), memorymat3)
    
    print(f"记忆矩阵已保存:")
    print(f"  memorymat1.npy: {memorymat1.shape}")
    print(f"  memorymat2.npy: {memorymat2.shape}")
    print(f"  memorymat3.npy: {memorymat3.shape}")
    
    # 生成临时矩阵
    print("生成临时矩阵...")
    Temp_MemMat(memorymat1, os.path.join(output_dir, 'Temp_low.npy'))
    Temp_MemMat(memorymat2, os.path.join(output_dir, 'Temp_med.npy'))
    Temp_MemMat(memorymat3, os.path.join(output_dir, 'Temp_hig.npy'))
    
    print("临时矩阵已保存:")
    print(f"  Temp_low.npy")
    print(f"  Temp_med.npy")
    print(f"  Temp_hig.npy")
    
    print("训练完成！")


def create_sample_training_data(n_samples=10000, n_features=1):
    """
    创建示例训练数据（用于测试）
    
    参数:
        n_samples: 样本数量
        n_features: 特征数量
        
    返回:
        training_data: 训练数据
    """
    print(f"创建示例训练数据: {n_samples} 样本, {n_features} 特征")
    
    # 生成正常数据（添加一些噪声）
    np.random.seed(42)
    base_signal = np.sin(np.linspace(0, 4*np.pi, n_samples))
    noise = np.random.normal(0, 0.1, n_samples)
    training_data = (base_signal + noise).reshape(-1, 1)
    
    # 归一化到[0,1]
    training_data = (training_data - training_data.min()) / (training_data.max() - training_data.min())
    
    return training_data


if __name__ == '__main__':
    # 创建示例训练数据
    training_data = create_sample_training_data(n_samples=10000, n_features=1)
    
    # 训练记忆矩阵
    train_memory_matrices(training_data)
    
    print("\n现在可以运行 test.py 进行异常检测了！") 