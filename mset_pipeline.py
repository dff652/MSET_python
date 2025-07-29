"""
MSET异常检测流水线
模块化的训练和测试流程
"""

import numpy as np
import pandas as pd
import os
import logging
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

from MSET_python.Model_optimized import MemoryMats_train, Temp_MemMat, MSETs_batch, Cal_sim, Cal_thres


class MSETPipeline:
    """MSET异常检测流水线类"""
    
    def __init__(self, output_dir: str = '.'):
        """
        初始化MSET流水线
        
        参数:
            output_dir: 输出目录，用于保存模型文件
        """
        self.output_dir = output_dir
        self.memory_files = ['memorymat1.npy', 'memorymat2.npy', 'memorymat3.npy']
        self.temp_files = ['Temp_low.npy', 'Temp_med.npy', 'Temp_hig.npy']
        self.model_info = {}
        
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def split_data(self, data: pd.DataFrame, test_size: float = 0.3, 
                   random_state: int = 42, split_method: str = 'ratio', time_split_point: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        将数据划分为训练集和测试集
        
        参数:
            data: 输入数据
            test_size: 测试集比例（split_method='ratio'时生效）
            random_state: 随机种子
            split_method: 'ratio'（比例划分）或 'time'（按时间点划分）
            time_split_point: 时间字符串，split_method='time'时生效
            
        返回:
            train_data, test_data: 训练集和测试集
        """
        self.logger.info(f"开始数据划分，总数据量: {len(data)}，划分方式: {split_method}")
        
        # 确保数据按时间排序
        data = data.sort_index()
        
        if split_method == 'time' and time_split_point is not None:
            # 按时间点划分
            split_time = pd.to_datetime(time_split_point)
            train_data = data[data.index < split_time]
            test_data = data[data.index >= split_time]
            self.logger.info(f"按时间点划分，split_time: {split_time}")
        else:
            # 默认按比例划分
            split_idx = int(len(data) * (1 - test_size))
            train_data = data.iloc[:split_idx]
            test_data = data.iloc[split_idx:]
            self.logger.info(f"按比例划分，训练集: {len(train_data)}，测试集: {len(test_data)}")
        
        self.logger.info(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")
        
        return train_data, test_data
    
    def prepare_training_data(self, train_data: pd.DataFrame, 
                            column: str) -> np.ndarray:
        """
        准备训练数据
        
        参数:
            train_data: 训练数据DataFrame
            column: 目标列名
            
        返回:
            training_array: 训练数据数组
        """
        self.logger.info("准备训练数据...")
        
        # 提取目标列数据
        series_data = train_data[column].values.reshape(-1, 1)
        
        # 归一化到[0,1]
        data_min = series_data.min()
        data_max = series_data.max()
        normalized_data = (series_data - data_min) / (data_max - data_min)
        
        # 保存归一化参数
        self.model_info['normalization'] = {
            'data_min': float(data_min),
            'data_max': float(data_max)
        }
        
        self.logger.info(f"训练数据形状: {normalized_data.shape}")
        return normalized_data
    
    def train_model(self, training_data: np.ndarray) -> bool:
        """
        训练MSET模型
        
        参数:
            training_data: 训练数据数组
            
        返回:
            success: 训练是否成功
        """
        try:
            self.logger.info("开始训练MSET记忆矩阵...")
            
            # 训练三个记忆矩阵（低、中、高负荷）
            memorymat1, memorymat2, memorymat3 = MemoryMats_train(training_data)
            
            # 保存记忆矩阵
            np.save(os.path.join(self.output_dir, 'memorymat1.npy'), memorymat1)
            np.save(os.path.join(self.output_dir, 'memorymat2.npy'), memorymat2)
            np.save(os.path.join(self.output_dir, 'memorymat3.npy'), memorymat3)
            
            # 生成临时矩阵
            Temp_MemMat(memorymat1, os.path.join(self.output_dir, 'Temp_low.npy'))
            Temp_MemMat(memorymat2, os.path.join(self.output_dir, 'Temp_med.npy'))
            Temp_MemMat(memorymat3, os.path.join(self.output_dir, 'Temp_hig.npy'))
            
            # 保存模型信息
            self.model_info['memory_matrices'] = {
                'memorymat1_shape': memorymat1.shape,
                'memorymat2_shape': memorymat2.shape,
                'memorymat3_shape': memorymat3.shape
            }
            self.model_info['training_time'] = datetime.now().isoformat()
            
            self.logger.info("MSET模型训练完成！")
            return True
            
        except Exception as e:
            self.logger.error(f"训练过程中出现错误: {str(e)}")
            return False
    
    def load_model(self) -> bool:
        """
        加载已训练的模型
        
        返回:
            success: 加载是否成功
        """
        try:
            # 检查所有必需文件是否存在
            missing_files = []
            for file in self.memory_files + self.temp_files:
                file_path = os.path.join(self.output_dir, file)
                if not os.path.exists(file_path):
                    missing_files.append(file)
            
            if missing_files:
                self.logger.error(f"缺少模型文件: {missing_files}")
                return False
            
            self.logger.info("模型文件加载成功")
            return True
            
        except Exception as e:
            self.logger.error(f"加载模型时出现错误: {str(e)}")
            return False
    
    def prepare_test_data(self, test_data: pd.DataFrame, 
                         column: str) -> np.ndarray:
        """
        准备测试数据
        
        参数:
            test_data: 测试数据DataFrame
            column: 目标列名
            
        返回:
            test_array: 测试数据数组
        """
        self.logger.info("准备测试数据...")
        
        # 提取目标列数据
        series_data = test_data[column].values.reshape(-1, 1)
        
        # 使用训练时的归一化参数
        if 'normalization' in self.model_info:
            data_min = self.model_info['normalization']['data_min']
            data_max = self.model_info['normalization']['data_max']
            normalized_data = (series_data - data_min) / (data_max - data_min)
        else:
            # 如果没有保存归一化参数，重新计算
            data_min = series_data.min()
            data_max = series_data.max()
            normalized_data = (series_data - data_min) / (data_max - data_min)
        
        self.logger.info(f"测试数据形状: {normalized_data.shape}")
        return normalized_data
    
    def detect_anomalies(self, test_data: np.ndarray, 
                        time_index: pd.DatetimeIndex,
                        sensor_name: str) -> Dict[str, Any]:
        """
        执行异常检测
        
        参数:
            test_data: 测试数据数组
            time_index: 时间索引
            sensor_name: 传感器名称
            
        返回:
            results: 检测结果字典
        """
        try:
            self.logger.info("开始异常检测...")
            
            # MSET估计
            Kest = MSETs_batch(
                os.path.join(self.output_dir, 'memorymat1.npy'),
                os.path.join(self.output_dir, 'memorymat2.npy'),
                os.path.join(self.output_dir, 'memorymat3.npy'),
                test_data
            )
            
            # 计算相似度
            sim = Cal_sim(test_data, Kest)
            
            # 计算动态阈值
            thres, warning_indices = Cal_thres(sim)
            
            # 异常检测
            anomaly_indices = np.where(sim < thres)[0]
            warning_indices = np.where((sim >= thres) & (sim < 0.8))[0]
            
            # 生成结果
            results = {
                'sensor_name': sensor_name,
                'anomaly_count': len(anomaly_indices),
                'warning_count': len(warning_indices),
                'total_points': len(test_data),
                'anomaly_rate': len(anomaly_indices) / len(test_data),
                'sim_mean': float(np.mean(sim)),
                'sim_std': float(np.std(sim)),
                'thres_mean': float(np.mean(thres)),
                'anomaly_indices': anomaly_indices.tolist(),
                'warning_indices': warning_indices.tolist(),
                'anomaly_times': [str(time_index[i]) for i in anomaly_indices if i < len(time_index)],
                'warning_times': [str(time_index[i]) for i in warning_indices if i < len(time_index)]
            }
            
            self.logger.info(f"异常检测完成，检测到 {len(anomaly_indices)} 个异常点")
            return results
            
        except Exception as e:
            self.logger.error(f"异常检测过程中出现错误: {str(e)}")
            return {}
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """
        保存检测结果
        
        参数:
            results: 检测结果
            filename: 文件名（可选）
        """
        if filename is None:
            filename = f"{results.get('sensor_name', 'unknown')}_results.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"结果已保存到: {filepath}")
        except Exception as e:
            self.logger.error(f"保存结果时出现错误: {str(e)}")
    
    def save_model_info(self, filename: str = 'model_info.json'):
        """
        保存模型信息
        
        参数:
            filename: 文件名
        """
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.model_info, f, ensure_ascii=False, indent=2)
            self.logger.info(f"模型信息已保存到: {filepath}")
        except Exception as e:
            self.logger.error(f"保存模型信息时出现错误: {str(e)}")


def run_mset_pipeline(data: pd.DataFrame, column: str, sensor_name: str,
                     test_size: float = 0.3, output_dir: str = '.',
                     split_method: str = 'ratio', time_split_point: str = None) -> Dict[str, Any]:
    """
    运行完整的MSET异常检测流水线
    
    参数:
        data: 输入数据
        column: 目标列名
        sensor_name: 传感器名称
        test_size: 测试集比例（split_method='ratio'时生效）
        output_dir: 输出目录
        split_method: 'ratio' 或 'time'
        time_split_point: 时间字符串，split_method='time'时生效
        
    返回:
        results: 检测结果
    """
    # 创建流水线实例
    pipeline = MSETPipeline(output_dir)
    
    # 数据划分
    train_data, test_data = pipeline.split_data(data, test_size, split_method=split_method, time_split_point=time_split_point)
    
    # 准备训练数据
    training_array = pipeline.prepare_training_data(train_data, column)
    
    # 训练模型
    if not pipeline.train_model(training_array):
        return {}
    
    # 保存模型信息
    pipeline.save_model_info()
    
    # 准备测试数据
    test_array = pipeline.prepare_test_data(test_data, column)
    
    # 异常检测
    results = pipeline.detect_anomalies(test_array, test_data.index, sensor_name)
    
    # 保存结果
    if results:
        pipeline.save_results(results)
    
    return results 