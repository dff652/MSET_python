
import argparse
from dataclasses import dataclass, asdict, field
import json
import numpy as np
import pandas as pd
import sys
from typing import List
import logging

from iotdb.Session import Session
import json

import os

import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

from tsdownsample import MinMaxLTTBDownsampler, M4Downsampler
import MSET_python.Model_optimized as Model

def read_iotdb(
        host="192.168.199.185",
        port="6667",
        user='root',
        password='root',
        path='root.supcon.nb.whlj.LJSJ',
        target_column='*',
        st='2023-06-01 00:00:00',
        et='2024-08-01 00:00:00',
        limit=1000000000,

):
    print('-' * 40 + '开始' + '-' * 40)
    print(f'\n开始读取数据{target_column},时间段{st}到{et}')
    
    session = Session(host, port, user, password, fetch_size=2000000)
    session.open(False)

    ststring = ">=" + st.replace(' ', 'T')
    etstring = "<=" + et.replace(' ', 'T')

    query = f"select `{target_column}` from {path}"

    if st:
        ststring = ">=" + st.replace(' ', 'T')

    if et:
        etstring = "<=" + et.replace(' ', 'T')

    if st and not et:
        query = query + f" where time {ststring}"
    if et and not st:
        query = query + f" where time {etstring}"
    if st and et:
        query = query + f" where time {ststring} and time {etstring}"

    if limit:
        query = query + f" limit {limit}"

    result = session.execute_query_statement(query)
    result.set_fetch_size(3000000)

    df = result.todf()

    df.set_index('Time', inplace=True)

    df.index = pd.to_datetime(df.index.astype('int64')).tz_localize('UTC').tz_convert('Asia/Shanghai')

    column_rename = {}
    for column in df.columns:
        if column.endswith('`'):
            column_new = column[(column.rindex('`', 0, len(column) - 2) + 1): len(column) - 1]
        else:
            column_new = column.split('.')[-1].replace('`', '')
        column_rename[column] = column_new

    df = df.rename(columns=column_rename)
    return df

def check_time_continuity(data, discontinuity_threshold=None):
    """
    检查时间序列的连续性
    Parameters
    ----------
    data : DataFrame
        输入的数据集，索引为时间戳
    sLength : int
        采样间隔，以秒为单位，默认为1秒
    Returns
    -------
    continuity_ratio : float
        时间戳中断的比例（相邻两个时间戳之差大于标准采样频率）
    continuity : Series
        布尔类型序列，标记每个间隔是否超过采样频率
    missing_timestamps : DatetimeIndex
        缺失的时间戳
    missing_ratio : float
        缺失时间戳占完整模板时间戳的比例
    """
    # score_df = pd.DataFrame(index=['time_continuity_ratio'],columns=data.columns)
    ts = 'ts'
    time_index = pd.DataFrame(columns=[ts], index=data.index)
    time_index[ts] = data.index
    interval = (time_index[ts] - time_index[ts].shift(1))
    interval_seconds = interval.dt.total_seconds()  # .values.ravel()
    # print('interval_seconds.mode :', interval_seconds.mode())
    if discontinuity_threshold is None or discontinuity_threshold == '':
        # 根据数据中的时间间隔推断采样频率
        # print(interval_seconds.mode())
        if len(interval_seconds) > 1:
            discontinuity_threshold = interval_seconds.mode()[0]
        else:
            return pd.DataFrame({
                'missing_ratio': [0],
                'missing_timestamps_count': [0]
            }).transpose(), pd.Series(False, index=data.index)
    else:
        discontinuity_threshold = int(discontinuity_threshold)

    continuity = interval_seconds > int(discontinuity_threshold)
    continuity_ratio = continuity.sum() / len(interval_seconds)

    # score = np.round((continuity_ratio) * 100, 2)
    # score = np.vectorize(lambda x: "{:.2f}%".format(x))(score)
    # score_df.loc[:,:] = score
    # discontinuity_index=np.where(interval_seconds > 1)

    start_time = time_index[ts].min()
    end_time = time_index[ts].max()
    # freq = pd.infer_freq(time_index[ts])
    freq = f'{int(discontinuity_threshold)}s'

    if freq is None:
        raise ValueError("无法推断时间序列的频率，请确保输入是有序时间戳序列")

    # 构造完整的时间戳序列
    full_timestamps = pd.date_range(start=start_time, end=end_time, freq=freq)

    # 找出缺少的时间戳
    missing_timestamps = full_timestamps.difference(time_index[ts])

    # 创建布尔序列以标记缺失和不连续的时间戳
    continuity = pd.Series(False, index=full_timestamps)
    continuity[missing_timestamps] = True  # 标记缺失的时间戳为 True

    # existing_continuity = pd.Series(interval_seconds > discontinuity_threshold, index=data.index)
    # continuity.update(existing_continuity)

    missing_count = continuity.sum()

    # 计算缺少时间戳的比例
    missing_ratio = missing_count / len(data)

    result_df = pd.DataFrame({
        f'missing_ratio': [missing_ratio],
        f'missing_timestamps_count': [missing_count]
    }).transpose()

    return result_df, continuity

def get_fulldata(data,
                 col_name, ):
  

    result_df, continuity = check_time_continuity(data)
 
    df = data.copy()

    missing_timestamps = continuity[continuity > 0].index
 

    full_time_index = df.index.append(pd.DatetimeIndex(missing_timestamps)).unique()
    df = df.reindex(full_time_index)

    df.sort_index(inplace=True)

    # 首先进行后向填充
    df[col_name] = df[col_name].bfill()

    # 然后进行前向填充
    df[col_name] = df[col_name].ffill()
 

    return df


def ts_downsample(data, downsampler='m4', n_out=100000):
    """
    Downsample time series data
    :param data: pd.Series
    :param downsampler: str
    :return: numpy.array, numpy.array
    """

    if downsampler == 'm4':
        s_ds = M4Downsampler().downsample(data, n_out=n_out)
    elif downsampler == 'minmax':
        s_ds = MinMaxLTTBDownsampler().downsample(data, n_out=n_out)

    downsampled_data = data.iloc[s_ds]
    downsampled_time = data.index[s_ds]

    return downsampled_data, downsampled_time


def adaptive_downsample(data, downsampler='m4', sample_param=0.1, min_threshold=1000):
    """
    自适应降采样函数，支持按比例或固定数量降采样
    
    参数:
        data: pd.Series 或带有单列的 pd.DataFrame - 输入数据
        downsampler: str - 降采样方法，支持 'm4', 'minmax', 'none'或None，为None或'none'时不降采样
        sample_param: float 或 None 
                     - 0到1之间表示按比例降采样
                     - 大于1时自动转为None，表示使用min_threshold作为固定降采样数量
        min_threshold: int - 两个用途:
                     - 当数据量小于此值时不进行降采样
                     - 当sample_param > 1或为None时，作为固定降采样数量
                     
    返回:
        downsampled_data: 降采样后的数据
        downsampled_ts: 降采样后的时间戳
    """
    # 如果输入是DataFrame，提取第一列作为数据
    if isinstance(data, pd.DataFrame):
        col_name = data.columns[0]
        series_data = data[col_name].copy()
    else:
        series_data = data.copy()
    
    # 获取原始数据长度
    data_length = len(series_data)
    
    # 如果数据长度小于阈值或降采样方法为None或'none'，则不进行降采样
    if data_length < min_threshold or downsampler is None or downsampler.lower() == 'none':
        if isinstance(data, pd.DataFrame):
            return data.copy(), data.index.copy()
        return series_data.copy(), series_data.index.copy()
    
    # 内部处理sample_param参数
    if sample_param is None or sample_param > 1:
        # 如果sample_param为None或大于1，使用min_threshold作为固定降采样数量
        n_out = min_threshold
    elif 0 < sample_param <= 1:  
        # 按比例降采样
        n_out = int(data_length * sample_param)
    else:
        # 不支持的参数值，使用min_threshold作为默认
        logging.warning(f"不支持的sample_param值: {sample_param}，应该在0-1之间或大于1。使用min_threshold作为默认值。")
        n_out = min_threshold
    
    # 确保n_out不超过原始数据长度
    n_out = min(n_out, data_length)
    
    # 如果是M4降采样，确保n_out是4的倍数
    if downsampler.lower() == 'm4':
        n_out = n_out + (4 - n_out % 4) if n_out % 4 != 0 else n_out
    
    # 执行降采样
    return ts_downsample(series_data, downsampler, n_out)


def detect_anomalies(sim, thres, warning_threshold=0.8):
    """
    异常检测函数
    
    参数:
        sim: 相似度数组
        thres: 动态阈值数组
        warning_threshold: 警告阈值，默认0.8
        
    返回:
        anomaly_indices: 异常点索引
        warning_indices: 警告点索引
    """
    anomaly_indices = []
    warning_indices = []
    
    for i in range(len(sim)):
        # 如果相似度低于动态阈值，标记为异常
        if sim[i] < thres[i]:
            anomaly_indices.append(i)
        # 如果相似度低于警告阈值，标记为警告
        elif sim[i] < warning_threshold:
            warning_indices.append(i)
    
    return np.array(anomaly_indices), np.array(warning_indices)


def generate_alarm(anomaly_indices, warning_indices, time_index, sensor_name):
    """
    生成报警信息
    
    参数:
        anomaly_indices: 异常点索引
        warning_indices: 警告点索引
        time_index: 时间索引
        sensor_name: 传感器名称
        
    返回:
        alarm_info: 报警信息字典
    """
    alarm_info = {
        'sensor_name': sensor_name,
        'anomaly_count': len(anomaly_indices),
        'warning_count': len(warning_indices),
        'anomaly_times': [],
        'warning_times': [],
        'total_points': len(time_index)
    }
    
    # 记录异常时间点
    for idx in anomaly_indices:
        if idx < len(time_index):
            alarm_info['anomaly_times'].append(str(time_index[idx]))
    
    # 记录警告时间点
    for idx in warning_indices:
        if idx < len(time_index):
            alarm_info['warning_times'].append(str(time_index[idx]))
    
    # 打印报警信息
    print(f"\n{'='*60}")
    print(f"传感器 {sensor_name} 异常检测结果:")
    print(f"总数据点: {alarm_info['total_points']}")
    print(f"异常点数量: {alarm_info['anomaly_count']}")
    print(f"警告点数量: {alarm_info['warning_count']}")
    print(f"异常率: {alarm_info['anomaly_count']/alarm_info['total_points']*100:.2f}%")
    
    if len(anomaly_indices) > 0:
        print(f"前5个异常时间点: {alarm_info['anomaly_times'][:5]}")
    
    if len(warning_indices) > 0:
        print(f"前5个警告时间点: {alarm_info['warning_times'][:5]}")
    print(f"{'='*60}\n")
    
    return alarm_info


def error_contribution_analysis(Kobs, Kest, anomaly_indices, sensor_name, time_index):
    """
    误差贡献率分析
    
    参数:
        Kobs: 观测值
        Kest: 估计值
        anomaly_indices: 异常点索引
        sensor_name: 传感器名称
        time_index: 时间索引
    """
    if len(anomaly_indices) == 0:
        print(f"传感器 {sensor_name} 未检测到异常点，跳过误差贡献率分析")
        return
    
    print(f"\n传感器 {sensor_name} 误差贡献率分析:")
    
    # 选择第一个异常点进行分析
    first_anomaly_idx = anomaly_indices[0]
    if first_anomaly_idx < len(Kobs):
        error = (Kobs[first_anomaly_idx] - Kest[first_anomaly_idx])**2
        error_cont = error / np.sum(error) if np.sum(error) > 0 else error
        
        print(f"异常时间点: {time_index[first_anomaly_idx]}")
        print(f"误差贡献率: {error_cont.flatten()}")
        print(f"最大误差贡献: {np.max(error_cont):.4f}")
        print(f"平均误差贡献: {np.mean(error_cont):.4f}")


def visualize_results(Kobs, Kest, sim, thres, anomaly_indices, warning_indices, 
                     time_index, sensor_name, np_Dmax=None, np_Dmin=None):
    """
    可视化异常检测结果
    
    参数:
        Kobs: 观测值
        Kest: 估计值
        sim: 相似度
        thres: 动态阈值
        anomaly_indices: 异常点索引
        warning_indices: 警告点索引
        time_index: 时间索引
        sensor_name: 传感器名称
        np_Dmax, np_Dmin: 归一化参数（可选）
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建子图
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # 1. 观测值vs估计值对比
    ax1 = axes[0]
    if np_Dmax is not None and np_Dmin is not None:
        # 反归一化
        Kobs_plot = Kobs * (np_Dmax - np_Dmin) + np_Dmin
        Kest_plot = Kest * (np_Dmax - np_Dmin) + np_Dmin
    else:
        Kobs_plot = Kobs
        Kest_plot = Kest
    
    ax1.plot(time_index, Kobs_plot, 'steelblue', label='观测值', linewidth=1.5)
    ax1.plot(time_index, Kest_plot, 'indianred', label='估计值', linewidth=1.5)
    
    # 标记异常点和警告点
    if len(anomaly_indices) > 0:
        ax1.scatter(time_index[anomaly_indices], Kobs_plot[anomaly_indices], 
                   color='red', s=50, label='异常点', zorder=5)
    if len(warning_indices) > 0:
        ax1.scatter(time_index[warning_indices], Kobs_plot[warning_indices], 
                   color='orange', s=30, label='警告点', zorder=5)
    
    ax1.set_title(f'{sensor_name} - 观测值vs估计值对比', fontsize=14)
    ax1.set_ylabel('数值', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 相似度和动态阈值
    ax2 = axes[1]
    ax2.plot(time_index, sim, 'blue', label='相似度', linewidth=1.5)
    ax2.plot(time_index, thres, 'red', label='动态阈值', linewidth=1.5)
    ax2.axhline(y=0.8, color='orange', linestyle='--', label='警告阈值(0.8)', alpha=0.7)
    ax2.set_title('相似度曲线和动态阈值', fontsize=14)
    ax2.set_ylabel('相似度', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 相对误差
    ax3 = axes[2]
    if np_Dmax is not None and np_Dmin is not None:
        relative_error = np.abs(Kobs_plot - Kest_plot) / (np.abs(Kobs_plot) + 1e-8) * 100
    else:
        relative_error = np.abs(Kobs - Kest) / (np.abs(Kobs) + 1e-8) * 100
    
    ax3.plot(time_index, relative_error, 'peru', linewidth=1)
    ax3.set_title('相对误差百分比', fontsize=14)
    ax3.set_xlabel('时间', fontsize=12)
    ax3.set_ylabel('相对误差 (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 保存图片
    plt.savefig(f'{sensor_name}_anomaly_detection.png', dpi=300, bbox_inches='tight')
    print(f"结果图片已保存为: {sensor_name}_anomaly_detection.png")


def main():
    muti_var_list =[
        "AC6401B_FM1.PV"
    ]
    path ='root.zhlh_202307_202412.ZHLH_4C_1216'

    result = pd.DataFrame()
    seg_dict = {}
    res =[]
    data_heatmap = {}
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    for sensor_id in muti_var_list:
        st: str = "2023-07-18 12:00:00"
        et: str = "2024-11-05 23:59:59"

        logging.info(f"开始处理传感器: {sensor_id}")
        logging.info(f"时间范围: {st} 到 {et}")

        # 读取数据
        data = read_iotdb(
            target_column=sensor_id,
            path=path,
            st=st,
            et=et)
        
        raw_data = data.copy()
        columns_to_check = raw_data.columns
        column = columns_to_check[0]

        # 数据预处理
        logging.info("开始数据预处理...")
        data = get_fulldata(raw_data, column) 
        logging.info(f"数据预处理完成，数据形状: {data.shape}")

        # 降采样
        logging.info("开始数据降采样...")
        downsampled_data, time_index = adaptive_downsample(data, downsampler='m4', sample_param=2, min_threshold=200000)
        logging.info(f"降采样完成，降采样后数据形状: {downsampled_data.shape}")

        # 数据归一化（如果需要）
        # 这里假设数据已经适合MSET处理，如果需要归一化可以添加
        Kobs = downsampled_data.values  # shape: (N, feature)
        
        # 检查记忆矩阵文件是否存在
        memory_files = ['memorymat1.npy', 'memorymat2.npy', 'memorymat3.npy']
        temp_files = ['Temp_low.npy', 'Temp_med.npy', 'Temp_hig.npy']
        
        missing_files = []
        for file in memory_files + temp_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            logging.error(f"缺少必要的文件: {missing_files}")
            logging.error("请确保以下文件存在:")
            logging.error("记忆矩阵文件: memorymat1.npy, memorymat2.npy, memorymat3.npy")
            logging.error("临时矩阵文件: Temp_low.npy, Temp_med.npy, Temp_hig.npy")
            continue
        
        # MSET异常检测
        logging.info("开始MSET异常检测...")
        try:
            Kest = Model.MSETs_batch('memorymat1.npy', 'memorymat2.npy', 'memorymat3.npy', Kobs)
            logging.info("MSET计算完成")
            
            # 计算相似度
            sim = Model.Cal_sim(Kobs, Kest)
            logging.info("相似度计算完成")
            
            # 计算动态阈值
            thres, index = Model.Cal_thres(sim)
            logging.info("动态阈值计算完成")
            
            # 1. 异常检测
            logging.info("开始异常检测...")
            anomaly_indices, warning_indices = detect_anomalies(sim, thres)
            logging.info(f"检测到 {len(anomaly_indices)} 个异常点，{len(warning_indices)} 个警告点")
            
            # 2. 生成报警信息
            alarm_info = generate_alarm(anomaly_indices, warning_indices, time_index, sensor_id)
            
            # 3. 误差贡献率分析
            error_contribution_analysis(Kobs, Kest, anomaly_indices, sensor_id, time_index)
            
            # 4. 可视化结果
            logging.info("开始生成可视化结果...")
            visualize_results(Kobs, Kest, sim, thres, anomaly_indices, warning_indices, 
                           time_index, sensor_id)
            logging.info("可视化完成")
            
            # 保存结果到文件
            results = {
                'sensor_name': sensor_id,
                'alarm_info': alarm_info,
                'anomaly_indices': anomaly_indices.tolist(),
                'warning_indices': warning_indices.tolist(),
                'sim_mean': float(np.mean(sim)),
                'sim_std': float(np.std(sim)),
                'thres_mean': float(np.mean(thres)),
                'data_shape': Kobs.shape
            }
            
            with open(f'{sensor_id}_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logging.info(f"结果已保存到 {sensor_id}_results.json")
            
        except Exception as e:
            logging.error(f"MSET处理过程中出现错误: {str(e)}")
            continue
        
        logging.info(f"传感器 {sensor_id} 处理完成\n")

if __name__ == '__main__':
    main()







