"""
模块化MSET异常检测测试脚本
使用mset_pipeline模块进行训练和测试
"""

import argparse
import logging
import numpy as np
import pandas as pd
import sys
from typing import List
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

from tsdownsample import MinMaxLTTBDownsampler, M4Downsampler
from MSET_python.mset_pipeline import MSETPipeline, run_mset_pipeline

# 导入原有的数据读取和预处理函数
from test import read_iotdb, get_fulldata, adaptive_downsample, detect_anomalies, generate_alarm, visualize_results


def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # 配置参数
    muti_var_list = ["AC6401B_FM1.PV"]
    path = 'root.zhlh_202307_202412.ZHLH_4C_1216'
    st = "2023-07-18 12:00:00"
    et = "2024-11-05 23:59:59"
    
    # 创建输出目录
    output_dir = 'mset_results'
    os.makedirs(output_dir, exist_ok=True)
    
    for sensor_id in muti_var_list:
        logger.info(f"开始处理传感器: {sensor_id}")
        logger.info(f"时间范围: {st} 到 {et}")

        # 1. 读取数据
        logger.info("开始读取IoTDB数据...")
        data = read_iotdb(
            target_column=sensor_id,
            path=path,
            st=st,
            et=et)
        
        raw_data = data.copy()
        columns_to_check = raw_data.columns
        column = columns_to_check[0]

        # 2. 数据预处理
        logger.info("开始数据预处理...")
        data = get_fulldata(raw_data, column) 
        logger.info(f"数据预处理完成，数据形状: {data.shape}")

        # 3. 降采样
        logger.info("开始数据降采样...")
        downsampled_data, time_index = adaptive_downsample(
            data, downsampler='m4', sample_param=2, min_threshold=200000)
        logger.info(f"降采样完成，降采样后数据形状: {downsampled_data.shape}")

        # 4. 使用模块化流水线进行训练和测试
        logger.info("开始MSET异常检测流水线...")
        
        # 方法1：使用完整的流水线（推荐）
        results = run_mset_pipeline(
            data=downsampled_data,
            column=column,
            sensor_name=sensor_id,
            # test_size=0.3,  # 30%作为测试集
            split_method='time',
            time_split_point='2024-08-01 00:00:00',
            output_dir=output_dir
        )
        
        
        
        if results:
            logger.info("异常检测完成！")
            logger.info(f"检测到 {results['anomaly_count']} 个异常点")
            logger.info(f"异常率: {results['anomaly_rate']*100:.2f}%")
            
            # 打印详细结果
            print(f"\n{'='*60}")
            print(f"传感器 {sensor_id} 异常检测结果:")
            print(f"总数据点: {results['total_points']}")
            print(f"异常点数量: {results['anomaly_count']}")
            print(f"警告点数量: {results['warning_count']}")
            print(f"异常率: {results['anomaly_rate']*100:.2f}%")
            print(f"平均相似度: {results['sim_mean']:.4f}")
            print(f"相似度标准差: {results['sim_std']:.4f}")
            
            if len(results['anomaly_times']) > 0:
                print(f"前5个异常时间点: {results['anomaly_times'][:5]}")
            
            if len(results['warning_times']) > 0:
                print(f"前5个警告时间点: {results['warning_times'][:5]}")
            print(f"{'='*60}\n")

            # 1. 还原测试集
            test_data = downsampled_data[downsampled_data.index >= pd.to_datetime('2024-08-01 00:00:00')]
            # 2. 归一化参数
            # 如果你保存了归一化参数，可以从 model_info.json 读取
            # 这里假设没有归一化，直接用原始数据
            Kobs = test_data[column].values.reshape(-1, 1)

            # 3. 加载估计值（可选：如果你在 detect_anomalies 里返回了 Kest，可以直接用）
            # 否则需要在 mset_pipeline.py 里让 detect_anomalies 返回 Kest
            # 这里假设 results 里没有 Kest，建议你在 mset_pipeline.py 里加上返回 Kest

            # 4. 可视化
            # 你需要准备 Kobs, Kest, sim, thres, anomaly_indices, warning_indices, time_index, sensor_name
            visualize_results(
                Kobs=Kobs,
                Kest=Kest,  # 需要从 detect_anomalies 返回
                sim=np.array(results['sim']),  # 需要在 detect_anomalies 返回 sim
                thres=np.array(results['thres']),  # 需要在 detect_anomalies 返回 thres
                anomaly_indices=np.array(results['anomaly_indices']),
                warning_indices=np.array(results['warning_indices']),
                time_index=test_data.index,
                sensor_name=sensor_id
            )
        
        # 方法2：分步骤使用流水线（可选，用于更精细的控制）
        """
        # 创建流水线实例
        pipeline = MSETPipeline(output_dir)
        
        # 数据划分
        train_data, test_data = pipeline.split_data(downsampled_data, test_size=0.3)
        
        # 准备训练数据
        training_array = pipeline.prepare_training_data(train_data, column)
        
        # 训练模型
        if pipeline.train_model(training_array):
            # 保存模型信息
            pipeline.save_model_info()
            
            # 准备测试数据
            test_array = pipeline.prepare_test_data(test_data, column)
            
            # 异常检测
            results = pipeline.detect_anomalies(test_array, test_data.index, sensor_id)
            
            # 保存结果
            if results:
                pipeline.save_results(results)
        """
        
        logger.info(f"传感器 {sensor_id} 处理完成\n")


def compare_methods():
    """比较不同方法的性能"""
    logger = logging.getLogger(__name__)
    
    # 配置参数
    sensor_id = "AC6401B_FM1.PV"
    path = 'root.zhlh_202307_202412.ZHLH_4C_1216'
    st = "2023-07-18 12:00:00"
    et = "2024-11-05 23:59:59"
    
    # 读取和预处理数据
    data = read_iotdb(target_column=sensor_id, path=path, st=st, et=et)
    raw_data = data.copy()
    column = raw_data.columns[0]
    data = get_fulldata(raw_data, column)
    downsampled_data, time_index = adaptive_downsample(
        data, downsampler='m4', sample_param=2, min_threshold=200000)
    
    # 比较不同的测试集比例
    test_sizes = [0.2, 0.3, 0.4]
    results_comparison = {}
    
    for test_size in test_sizes:
        logger.info(f"测试测试集比例: {test_size}")
        
        output_dir = f'mset_results_test_{int(test_size*100)}'
        os.makedirs(output_dir, exist_ok=True)
        
        results = run_mset_pipeline(
            data=downsampled_data,
            column=column,
            sensor_name=sensor_id,
            test_size=test_size,
            output_dir=output_dir
        )
        
        if results:
            results_comparison[test_size] = {
                'anomaly_rate': results['anomaly_rate'],
                'sim_mean': results['sim_mean'],
                'sim_std': results['sim_std'],
                'total_points': results['total_points']
            }
    
    # 打印比较结果
    print("\n不同测试集比例的比较结果:")
    print("="*60)
    for test_size, metrics in results_comparison.items():
        print(f"测试集比例 {test_size*100}%:")
        print(f"  异常率: {metrics['anomaly_rate']*100:.2f}%")
        print(f"  平均相似度: {metrics['sim_mean']:.4f}")
        print(f"  相似度标准差: {metrics['sim_std']:.4f}")
        print(f"  测试数据点: {metrics['total_points']}")
        print()


if __name__ == '__main__':
    # 运行主函数
    main()
    
    # 可选：运行比较实验
    # compare_methods() 