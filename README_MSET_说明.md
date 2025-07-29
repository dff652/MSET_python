# MSET异常检测系统说明文档

## 1. 问题解答

### 1.1 训练数据和测试数据不一致的问题

**问题**：`create_sample_training_data` 生成的训练数据与 `test.py` 的测试数据不一致。

**原因**：
- `create_sample_training_data` 生成的是模拟的正弦波数据，用于测试和演示
- `test.py` 使用的是真实的IoTDB数据

**解决方案**：
- 使用模块化流水线 `mset_pipeline.py`，从真实的降采样数据中划分训练集和测试集
- 确保训练和测试数据来自同一数据源，保持一致性

### 1.2 从downsampled_data中划分训练和测试数据

**实现方式**：
```python
# 在 mset_pipeline.py 中
def split_data(self, data: pd.DataFrame, test_size: float = 0.3):
    # 按时间顺序划分，而不是随机划分
    split_idx = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    return train_data, test_data
```

**优势**：
- 保持时间序列的连续性
- 训练集使用历史数据，测试集使用未来数据
- 更符合实际应用场景

### 1.3 训练和测试的模块化

**模块化设计**：
- `MSETPipeline` 类封装了完整的训练和测试流程
- 支持分步骤执行或一键运行
- 自动保存模型文件和检测结果

**主要功能**：
- 数据划分和预处理
- 模型训练和保存
- 异常检测和结果分析
- 结果保存和可视化

### 1.4 MSET_batch vs MSETs_batch 的区别

#### MSET_batch（单记忆矩阵）
```python
def MSET_batch(memorymat_name, Kobs, Temp_name):
    """
    使用单个记忆矩阵进行MSET计算
    """
    memorymat = np.load(memorymat_name)
    Temp = np.load(Temp_name)
    # 计算距离矩阵
    diff = memorymat[:, np.newaxis, :] - Kobs[np.newaxis, :, :]
    Temp1 = np.linalg.norm(diff, axis=2)
    # MSET估计
    Kest = np.dot(np.dot(memorymat.T, np.linalg.pinv(Temp)), Temp1)
    return Kest.T
```

#### MSETs_batch（多记忆矩阵）
```python
def MSETs_batch(memorymat1_name, memorymat2_name, memorymat3_name, Kobs):
    """
    使用三个记忆矩阵进行MSET计算
    """
    # 根据负荷水平选择不同的记忆矩阵
    idx_low = np.where(Kobs[:, -1] < 1/3)[0]    # 低负荷
    idx_med = np.where((Kobs[:, -1] >= 1/3) & (Kobs[:, -1] <= 2/3))[0]  # 中负荷
    idx_high = np.where(Kobs[:, -1] > 2/3)[0]   # 高负荷
    
    # 分别使用对应的记忆矩阵
    if len(idx_low) > 0:
        Kest[idx_low] = MSET_batch(memorymat1_name, Kobs[idx_low], 'Temp_low.npy')
    if len(idx_med) > 0:
        Kest[idx_med] = MSET_batch(memorymat2_name, Kobs[idx_med], 'Temp_med.npy')
    if len(idx_high) > 0:
        Kest[idx_high] = MSET_batch(memorymat3_name, Kobs[idx_high], 'Temp_hig.npy')
```

### 1.5 三个记忆矩阵的作用

**设计原理**：
- 根据数据的负荷水平（通常用最后一列表示）将数据分为三类
- 每类数据使用专门的记忆矩阵，提高异常检测的准确性

**分类标准**：
- **低负荷**：负荷值 < 1/3，使用 `memorymat1`
- **中负荷**：1/3 ≤ 负荷值 ≤ 2/3，使用 `memorymat2`
- **高负荷**：负荷值 > 2/3，使用 `memorymat3`

**优势**：
- 针对不同工况优化记忆矩阵
- 提高异常检测的敏感性和准确性
- 减少误报和漏报

## 2. 模块化设计架构

### 2.1 文件结构
```
MSET_python/
├── mset_pipeline.py          # 模块化流水线
├── test_modular.py          # 模块化测试脚本
├── test.py                  # 原始测试脚本
├── train_memory_matrices.py # 记忆矩阵训练脚本
├── Model_optimized.py       # 优化的MSET模型
└── README_MSET_说明.md      # 说明文档
```

### 2.2 核心类和方法

#### MSETPipeline 类
```python
class MSETPipeline:
    def __init__(self, output_dir='.'):
        # 初始化流水线
    
    def split_data(self, data, test_size=0.3):
        # 数据划分
    
    def prepare_training_data(self, train_data, column):
        # 准备训练数据
    
    def train_model(self, training_data):
        # 训练模型
    
    def detect_anomalies(self, test_data, time_index, sensor_name):
        # 异常检测
```

#### 便捷函数
```python
def run_mset_pipeline(data, column, sensor_name, test_size=0.3, output_dir='.'):
    # 一键运行完整流水线
```

## 3. 使用指南

### 3.1 快速开始
```python
from mset_pipeline import run_mset_pipeline

# 准备数据
data = your_data_loading_function()
column = 'your_target_column'
sensor_name = 'your_sensor_name'

# 运行异常检测
results = run_mset_pipeline(
    data=data,
    column=column,
    sensor_name=sensor_name,
    test_size=0.3,
    output_dir='results'
)
```

### 3.2 分步骤使用
```python
from mset_pipeline import MSETPipeline

# 创建流水线实例
pipeline = MSETPipeline('results')

# 数据划分
train_data, test_data = pipeline.split_data(data, test_size=0.3)

# 训练模型
training_array = pipeline.prepare_training_data(train_data, column)
pipeline.train_model(training_array)

# 异常检测
test_array = pipeline.prepare_test_data(test_data, column)
results = pipeline.detect_anomalies(test_array, test_data.index, sensor_name)
```

### 3.3 结果分析
```python
# 查看检测结果
print(f"异常点数量: {results['anomaly_count']}")
print(f"异常率: {results['anomaly_rate']*100:.2f}%")
print(f"平均相似度: {results['sim_mean']:.4f}")

# 查看异常时间点
for time in results['anomaly_times'][:5]:
    print(f"异常时间: {time}")
```

## 4. 性能优化

### 4.1 数据预处理优化
- 使用降采样减少计算量
- 数据归一化提高数值稳定性
- 时间序列连续性检查

### 4.2 算法优化
- 向量化计算提高效率
- 批量处理减少内存占用
- 多记忆矩阵提高准确性

### 4.3 存储优化
- 自动保存模型文件
- 结果序列化存储
- 日志记录便于调试

## 5. 常见问题

### 5.1 内存不足
- 减少数据量或使用更激进的降采样
- 分批处理大数据集

### 5.2 训练时间过长
- 调整降采样参数
- 使用更小的记忆矩阵

### 5.3 检测结果不准确
- 检查训练数据的质量
- 调整异常检测阈值
- 验证数据预处理步骤

## 6. 扩展功能

### 6.1 多传感器支持
- 可以同时处理多个传感器
- 支持传感器间的关联分析

### 6.2 实时检测
- 支持流式数据处理
- 增量模型更新

### 6.3 可视化增强
- 交互式图表
- 异常点标注
- 趋势分析

## 7. 总结

模块化的MSET异常检测系统解决了以下问题：
1. **数据一致性问题**：从同一数据源划分训练和测试集
2. **模块化设计**：清晰的类和方法结构，便于维护和扩展
3. **性能优化**：多记忆矩阵设计提高检测准确性
4. **易用性**：提供便捷的一键运行接口

通过使用 `mset_pipeline.py` 和 `test_modular.py`，您可以轻松实现高质量的MSET异常检测。 