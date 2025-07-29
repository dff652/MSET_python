"""
模块化MSET异常检测示例
演示如何使用新的模块化系统
"""

import numpy as np
import pandas as pd
import logging
from mset_pipeline import MSETPipeline, run_mset_pipeline


def create_demo_data(n_samples=10000):
    """
    创建演示数据
    模拟真实的传感器数据，包含正常和异常模式
    """
    print("创建演示数据...")
    
    # 生成时间索引
    time_index = pd.date_range('2023-01-01', periods=n_samples, freq='1min')
    
    # 生成正常数据（正弦波 + 噪声）
    np.random.seed(42)
    t = np.linspace(0, 4*np.pi, n_samples)
    normal_signal = np.sin(t) + 0.1 * np.random.normal(0, 1, n_samples)
    
    # 添加一些异常点
    anomaly_indices = [2000, 3000, 4000, 5000, 6000, 7000, 8000]
    for idx in anomaly_indices:
        if idx < n_samples:
            normal_signal[idx] += 2.0  # 添加异常值
    
    # 创建DataFrame
    data = pd.DataFrame({
        'sensor_value': normal_signal,
        'load_level': np.random.uniform(0, 1, n_samples)  # 负荷水平
    }, index=time_index)
    
    print(f"演示数据创建完成，形状: {data.shape}")
    return data


def demo_basic_usage():
    """演示基本用法"""
    print("\n" + "="*60)
    print("演示1：基本用法 - 一键运行完整流水线")
    print("="*60)
    
    # 创建演示数据
    data = create_demo_data(5000)
    
    # 一键运行异常检测
    results = run_mset_pipeline(
        data=data,
        column='sensor_value',
        sensor_name='demo_sensor',
        test_size=0.3,
        output_dir='demo_results'
    )
    
    if results:
        print(f"\n检测结果:")
        print(f"  异常点数量: {results['anomaly_count']}")
        print(f"  异常率: {results['anomaly_rate']*100:.2f}%")
        print(f"  平均相似度: {results['sim_mean']:.4f}")
        print(f"  相似度标准差: {results['sim_std']:.4f}")
        
        if len(results['anomaly_times']) > 0:
            print(f"  前3个异常时间点:")
            for i, time in enumerate(results['anomaly_times'][:3]):
                print(f"    {i+1}. {time}")


def demo_step_by_step():
    """演示分步骤用法"""
    print("\n" + "="*60)
    print("演示2：分步骤用法 - 精细控制")
    print("="*60)
    
    # 创建演示数据
    data = create_demo_data(3000)
    
    # 创建流水线实例
    pipeline = MSETPipeline('demo_results_step')
    
    # 1. 数据划分
    print("步骤1：数据划分")
    train_data, test_data = pipeline.split_data(data, test_size=0.4)
    print(f"  训练集大小: {len(train_data)}")
    print(f"  测试集大小: {len(test_data)}")
    
    # 2. 准备训练数据
    print("步骤2：准备训练数据")
    training_array = pipeline.prepare_training_data(train_data, 'sensor_value')
    print(f"  训练数据形状: {training_array.shape}")
    
    # 3. 训练模型
    print("步骤3：训练模型")
    success = pipeline.train_model(training_array)
    if success:
        print("  模型训练成功！")
        pipeline.save_model_info()
    
    # 4. 准备测试数据
    print("步骤4：准备测试数据")
    test_array = pipeline.prepare_test_data(test_data, 'sensor_value')
    print(f"  测试数据形状: {test_array.shape}")
    
    # 5. 异常检测
    print("步骤5：异常检测")
    results = pipeline.detect_anomalies(test_array, test_data.index, 'demo_sensor_step')
    
    if results:
        print(f"  检测到 {results['anomaly_count']} 个异常点")
        pipeline.save_results(results)


def demo_model_reuse():
    """演示模型重用"""
    print("\n" + "="*60)
    print("演示3：模型重用 - 使用已训练的模型")
    print("="*60)
    
    # 创建新的测试数据
    new_data = create_demo_data(2000)
    
    # 创建流水线实例
    pipeline = MSETPipeline('demo_results')
    
    # 检查是否有已训练的模型
    if pipeline.load_model():
        print("找到已训练的模型，开始异常检测...")
        
        # 直接使用已训练的模型进行检测
        test_array = pipeline.prepare_test_data(new_data, 'sensor_value')
        results = pipeline.detect_anomalies(test_array, new_data.index, 'demo_sensor_reuse')
        
        if results:
            print(f"  检测到 {results['anomaly_count']} 个异常点")
            print(f"  异常率: {results['anomaly_rate']*100:.2f}%")
    else:
        print("未找到已训练的模型，请先运行演示1或演示2")


def demo_parameter_comparison():
    """演示参数比较"""
    print("\n" + "="*60)
    print("演示4：参数比较 - 不同测试集比例的影响")
    print("="*60)
    
    # 创建演示数据
    data = create_demo_data(6000)
    
    # 比较不同的测试集比例
    test_sizes = [0.2, 0.3, 0.4, 0.5]
    comparison_results = {}
    
    for test_size in test_sizes:
        print(f"\n测试测试集比例: {test_size*100}%")
        
        output_dir = f'demo_results_test_{int(test_size*100)}'
        
        results = run_mset_pipeline(
            data=data,
            column='sensor_value',
            sensor_name=f'demo_sensor_{int(test_size*100)}',
            test_size=test_size,
            output_dir=output_dir
        )
        
        if results:
            comparison_results[test_size] = {
                'anomaly_rate': results['anomaly_rate'],
                'sim_mean': results['sim_mean'],
                'total_points': results['total_points']
            }
    
    # 打印比较结果
    print(f"\n参数比较结果:")
    print(f"{'测试集比例':<12} {'异常率':<10} {'平均相似度':<12} {'测试点数':<10}")
    print("-" * 50)
    for test_size, metrics in comparison_results.items():
        print(f"{test_size*100:>8}%     {metrics['anomaly_rate']*100:>6.2f}%    "
              f"{metrics['sim_mean']:>10.4f}    {metrics['total_points']:>8}")


def main():
    """主函数"""
    # 设置日志级别
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("MSET异常检测模块化系统演示")
    print("="*60)
    
    try:
        # 演示1：基本用法
        demo_basic_usage()
        
        # 演示2：分步骤用法
        demo_step_by_step()
        
        # 演示3：模型重用
        demo_model_reuse()
        
        # 演示4：参数比较
        demo_parameter_comparison()
        
        print("\n" + "="*60)
        print("所有演示完成！")
        print("="*60)
        print("\n生成的文件:")
        print("- demo_results/: 基本用法结果")
        print("- demo_results_step/: 分步骤用法结果")
        print("- demo_results_test_*/: 参数比较结果")
        print("- *.json: 检测结果和模型信息")
        print("- *.npy: 记忆矩阵和临时矩阵")
        
    except Exception as e:
        print(f"演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 