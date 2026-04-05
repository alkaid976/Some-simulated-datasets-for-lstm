"""
硬盘故障预测系统
使用LSTM模型预测硬盘剩余寿命
作者: AI助手
日期: 2024年
"""

# ==================== 第一部分: 导入库 ====================
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_recall_curve, average_precision_score, precision_score, recall_score)
import warnings

warnings.filterwarnings('ignore')

# 深度学习相关
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization,
                                     Input, Bidirectional, Conv1D, MaxPooling1D,
                                     Flatten, concatenate)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint, TensorBoard)
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.utils import to_categorical

# 系统相关
import os
import pickle
import json
import joblib
from pathlib import Path
import random
from collections import Counter

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 创建目录结构
def create_directories():
    """创建必要的目录结构"""
    directories = [
        'data/raw',
        'data/processed',
        'data/generated',
        'models',
        'results/plots',
        'results/reports',
        'logs',
        'deployment'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("目录结构创建完成")


create_directories()


# ==================== 第二部分: 数据生成器 ====================
class HardDriveDataGenerator:
    """硬盘故障数据生成器"""

    def __init__(self, n_drives=200, days_per_drive=365, failure_rate=0.3):
        """
        初始化生成器

        参数:
        n_drives: 硬盘数量
        days_per_drive: 每块硬盘监控的天数
        failure_rate: 故障率
        """
        self.n_drives = n_drives
        self.days_per_drive = days_per_drive
        self.failed_drives = int(n_drives * failure_rate)
        self.drive_types = ['HDD', 'SSD']
        self.hdd_models = ['ST1000DM010', 'WD1003FZEX', 'HGST_HUS724030ALE641']
        self.ssd_models = ['Samsung_870_EVO_1TB', 'Crucial_MX500_1TB', 'WD_Blue_SN550_1TB']

    def generate_smart_data(self, drive_id, drive_type, will_fail=True, failure_day=None):
        """
        为单块硬盘生成SMART数据
        """
        # 如果没有指定故障日，随机生成
        if will_fail and failure_day is None:
            failure_day = np.random.randint(150, self.days_per_drive - 14)
        elif not will_fail:
            failure_day = self.days_per_drive + 100  # 不会故障的硬盘

        data = []
        base_date = datetime(2023, 1, 1)

        for day in range(self.days_per_drive):
            date = base_date + timedelta(days=day)

            # 如果是故障硬盘且过了故障日，标记为故障
            if will_fail and day >= failure_day:
                break

            # 生成基础SMART数据
            if drive_type == "HDD":
                row = self._generate_hdd_data(day, failure_day, will_fail)
                model = np.random.choice(self.hdd_models)
            else:  # SSD
                row = self._generate_ssd_data(day, failure_day, will_fail)
                model = np.random.choice(self.ssd_models)

            # 计算剩余使用寿命(RUL)
            if will_fail:
                rul = failure_day - day
                status = "working"
                if rul <= 0:
                    status = "failed"
            else:
                rul = -1
                status = "working"

            # 添加通用信息
            row.update({
                'date': date.strftime('%Y-%m-%d'),
                'serial_number': f'{drive_type}_{drive_id:06d}',
                'drive_type': drive_type,
                'model': model,
                'capacity_bytes': np.random.choice([1000204886016, 2000398934016, 3000592982016]),
                'status': status,
                'days_to_failure': rul,
                'will_fail': 1 if will_fail else 0,
                'failure_day': failure_day if will_fail else -1,
                'day_index': day
            })

            data.append(row)

        return pd.DataFrame(data)

    def _generate_hdd_data(self, day, failure_day, will_fail):
        """生成HDD SMART数据"""
        # 基础值
        power_on_hours = 24 * day

        if not will_fail:
            # 正常硬盘
            reallocated = np.random.poisson(0.005 * day)
            uncorrectable = np.random.poisson(0.0005 * day)
            pending = np.random.poisson(0.002 * day)
            read_error_rate = np.random.exponential(0.05)
            temperature = 30 + np.random.normal(0, 1.5)
            seek_error_rate = np.random.exponential(0.01)
        else:
            # 故障硬盘
            if day < failure_day - 60:  # 早期
                reallocated = np.random.poisson(0.005 * day)
                uncorrectable = np.random.poisson(0.0005 * day)
                pending = np.random.poisson(0.002 * day)
                read_error_rate = np.random.exponential(0.05)
                temperature = 30 + np.random.normal(0, 1.5)
                seek_error_rate = np.random.exponential(0.01)

            elif day < failure_day - 14:  # 中期
                progress = (day - (failure_day - 60)) / 46
                reallocated = np.random.poisson(0.005 * day + 10 * progress)
                uncorrectable = np.random.poisson(0.0005 * day + 1 * progress)
                pending = np.random.poisson(0.002 * day + 5 * progress)
                read_error_rate = np.random.exponential(0.05 + 0.1 * progress)
                temperature = 30 + np.random.normal(0, 1.5) + 3 * progress
                seek_error_rate = np.random.exponential(0.01 + 0.05 * progress)

            else:  # 晚期
                days_to_fail = failure_day - day
                reallocated = np.random.poisson(0.005 * day + 10 + 50 * (14 - days_to_fail) / 14)
                uncorrectable = np.random.poisson(0.0005 * day + 1 + 10 * (14 - days_to_fail) / 14)
                pending = np.random.poisson(0.002 * day + 5 + 30 * (14 - days_to_fail) / 14)
                read_error_rate = np.random.exponential(0.05 + 0.1 + 0.5 * (14 - days_to_fail) / 14)
                temperature = 30 + np.random.normal(0, 1.5) + 3 + 7 * (14 - days_to_fail) / 14
                seek_error_rate = np.random.exponential(0.01 + 0.05 + 0.2 * (14 - days_to_fail) / 14)

        # 归一化值
        normalized_reallocated = max(1, 100 - reallocated * 1.5)
        normalized_uncorrectable = max(1, 100 - uncorrectable * 8)

        return {
            # SMART原始值
            'smart_5_raw': int(reallocated),  # 重分配扇区计数
            'smart_187_raw': int(uncorrectable),  # 无法修正错误
            'smart_197_raw': int(pending),  # 当前待处理扇区
            'smart_9_raw': int(power_on_hours),  # 通电时间
            'smart_1_raw': read_error_rate,  # 读取错误率
            'smart_7_raw': seek_error_rate,  # 寻道错误率

            # SMART归一化值
            'smart_5_normalized': int(normalized_reallocated),
            'smart_187_normalized': int(normalized_uncorrectable),
            'smart_197_normalized': max(1, 100 - pending * 4),
            'smart_9_normalized': max(1, 100 - power_on_hours / 15000),
            'smart_1_normalized': max(1, 100 - read_error_rate * 15),
            'smart_7_normalized': max(1, 100 - seek_error_rate * 20),

            # 性能指标
            'temperature': temperature,
            'read_iops': np.random.normal(150, 15),
            'write_iops': np.random.normal(100, 10),
            'io_latency': np.random.exponential(3),
            'throughput_read': np.random.normal(200, 20),
            'throughput_write': np.random.normal(150, 15),

            # 其他指标
            'power_cycles': np.random.poisson(0.1 * day),
            'uncorrectable_errors': np.random.poisson(0.001 * day),
            'command_timeout': np.random.poisson(0.0001 * day)
        }

    def _generate_ssd_data(self, day, failure_day, will_fail):
        """生成SSD SMART数据"""
        power_on_hours = 24 * day

        if not will_fail:
            # 正常SSD
            wear_leveling = max(1, 100 - day * 0.008)
            reallocated = np.random.poisson(0.0005 * day)
            uncorrectable = np.random.poisson(0.00005 * day)
            temperature = 35 + np.random.normal(0, 2)
            nand_writes = np.random.poisson(0.1 * day)
        else:
            # 故障SSD
            if day < failure_day - 90:  # 早期
                wear_leveling = max(1, 100 - day * 0.008)
                reallocated = np.random.poisson(0.0005 * day)
                uncorrectable = np.random.poisson(0.00005 * day)
                temperature = 35 + np.random.normal(0, 2)
                nand_writes = np.random.poisson(0.1 * day)

            elif day < failure_day - 21:  # 中期
                progress = (day - (failure_day - 90)) / 69
                wear_leveling = max(1, 100 - day * 0.008 - 40 * progress)
                reallocated = np.random.poisson(0.0005 * day + 3 * progress)
                uncorrectable = np.random.poisson(0.00005 * day + 0.3 * progress)
                temperature = 35 + np.random.normal(0, 2) + 4 * progress
                nand_writes = np.random.poisson(0.1 * day + 5 * progress)

            else:  # 晚期
                days_to_fail = failure_day - day
                wear_leveling = max(1, 100 - day * 0.008 - 40 - 50 * (21 - days_to_fail) / 21)
                reallocated = np.random.poisson(0.0005 * day + 3 + 10 * (21 - days_to_fail) / 21)
                uncorrectable = np.random.poisson(0.00005 * day + 0.3 + 2 * (21 - days_to_fail) / 21)
                temperature = 35 + np.random.normal(0, 2) + 4 + 8 * (21 - days_to_fail) / 21
                nand_writes = np.random.poisson(0.1 * day + 5 + 20 * (21 - days_to_fail) / 21)

        return {
            # SMART原始值
            'smart_5_raw': int(reallocated),
            'smart_187_raw': int(uncorrectable),
            'smart_197_raw': np.random.poisson(0.0002 * day),
            'smart_9_raw': int(power_on_hours),
            'smart_1_raw': np.random.exponential(0.02),
            'smart_233_raw': wear_leveling,  # SSD磨损度
            'smart_241_raw': nand_writes,  # 写入数据总量

            # SMART归一化值
            'smart_5_normalized': max(1, 100 - reallocated * 4),
            'smart_187_normalized': max(1, 100 - uncorrectable * 15),
            'smart_197_normalized': 100,
            'smart_9_normalized': max(1, 100 - power_on_hours / 20000),
            'smart_1_normalized': 100,
            'smart_233_normalized': wear_leveling,
            'smart_241_normalized': max(1, 100 - nand_writes * 0.001),

            # 性能指标
            'temperature': temperature,
            'read_iops': np.random.normal(300, 25),
            'write_iops': np.random.normal(200, 20),
            'io_latency': np.random.exponential(0.3),
            'throughput_read': np.random.normal(500, 50),
            'throughput_write': np.random.normal(400, 40),

            # 其他指标
            'power_cycles': np.random.poisson(0.2 * day),
            'uncorrectable_errors': np.random.poisson(0.0001 * day),
            'command_timeout': np.random.poisson(0.00005 * day),
            'media_wearout': wear_leveling
        }

    def generate_all_data(self, save_to_file=True):
        """生成所有硬盘的数据"""
        all_data = []

        print(f"开始生成 {self.n_drives} 块硬盘的数据...")
        print(f"故障硬盘数: {self.failed_drives} ({self.failed_drives / self.n_drives * 100:.1f}%)")

        # 生成故障硬盘
        for i in range(self.failed_drives):
            drive_type = np.random.choice(self.drive_types, p=[0.7, 0.3])
            failure_day = np.random.randint(150, self.days_per_drive - 14)
            drive_data = self.generate_smart_data(i, drive_type, will_fail=True, failure_day=failure_day)
            all_data.append(drive_data)

            if (i + 1) % 20 == 0:
                print(f"  已生成 {i + 1}/{self.failed_drives} 块故障硬盘...")

        # 生成正常硬盘
        for i in range(self.failed_drives, self.n_drives):
            drive_type = np.random.choice(self.drive_types, p=[0.7, 0.3])
            drive_data = self.generate_smart_data(i, drive_type, will_fail=False)
            all_data.append(drive_data)

            if (i + 1 - self.failed_drives) % 20 == 0:
                print(f"  已生成 {i + 1 - self.failed_drives}/{self.n_drives - self.failed_drives} 块正常硬盘...")

        # 合并所有数据
        full_data = pd.concat(all_data, ignore_index=True)

        # 保存数据
        if save_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated/hard_drive_data_{timestamp}.csv"
            full_data.to_csv(filename, index=False)
            print(f"\n数据已保存到: {filename}")
            print(f"文件大小: {os.path.getsize(filename) / 1024 / 1024:.2f} MB")

        return full_data

    def analyze_data(self, data):
        """分析生成的数据"""
        print("\n" + "=" * 60)
        print("数据统计分析")
        print("=" * 60)

        # 基础统计
        print(f"\n1. 基础统计:")
        print(f"   总记录数: {len(data):,}")
        print(f"   硬盘数量: {data['serial_number'].nunique()}")
        print(f"   数据时间跨度: {data['date'].min()} 到 {data['date'].max()}")

        # 故障统计
        failed_drives = data[data['will_fail'] == 1]['serial_number'].unique()
        normal_drives = data[data['will_fail'] == 0]['serial_number'].unique()

        print(f"\n2. 故障统计:")
        print(f"   故障硬盘数: {len(failed_drives)}")
        print(f"   正常硬盘数: {len(normal_drives)}")
        print(f"   故障率: {len(failed_drives) / (len(failed_drives) + len(normal_drives)) * 100:.1f}%")

        # 硬盘类型统计
        print(f"\n3. 硬盘类型统计:")
        for drive_type in ['HDD', 'SSD']:
            type_drives = data[data['drive_type'] == drive_type]['serial_number'].unique()
            type_failed = len([d for d in type_drives if d in failed_drives])
            print(
                f"   {drive_type}硬盘: {len(type_drives)}块 (故障: {type_failed}块, {type_failed / len(type_drives) * 100:.1f}%)")

        # SMART指标统计
        print(f"\n4. 关键SMART指标统计:")
        smart_features = ['smart_5_raw', 'smart_187_raw', 'smart_197_raw', 'smart_233_raw']
        for feature in smart_features:
            if feature in data.columns:
                failed_mean = data[data['will_fail'] == 1][feature].mean()
                normal_mean = data[data['will_fail'] == 0][feature].mean()
                print(f"   {feature}: 故障硬盘均值={failed_mean:.2f}, 正常硬盘均值={normal_mean:.2f}")

        # 故障时间统计
        failure_times = []
        for drive in failed_drives:
            drive_data = data[data['serial_number'] == drive]
            failure_day = drive_data['failure_day'].iloc[0]
            if failure_day > 0:
                failure_times.append(failure_day)

        if failure_times:
            print(f"\n5. 故障时间统计:")
            print(f"   平均故障时间: 第{np.mean(failure_times):.1f}天")
            print(f"   最早故障: 第{min(failure_times)}天")
            print(f"   最晚故障: 第{max(failure_times)}天")

        return {
            'total_records': len(data),
            'total_drives': data['serial_number'].nunique(),
            'failed_drives': len(failed_drives),
            'normal_drives': len(normal_drives),
            'failure_rate': len(failed_drives) / (len(failed_drives) + len(normal_drives))
        }


# ==================== 第三部分: 数据预处理 ====================
class DataPreprocessor:
    """数据预处理类"""

    def __init__(self, sequence_length=30, prediction_horizon=7,
                 features=None, target='will_fail'):
        """
        初始化预处理类

        参数:
        sequence_length: 序列长度
        prediction_horizon: 预测未来多少天
        features: 使用的特征列表
        target: 目标变量
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.target = target

        # 默认特征
        if features is None:
            self.features = [
                'smart_5_raw', 'smart_187_raw', 'smart_197_raw',
                'smart_9_raw', 'smart_233_raw', 'smart_1_raw',
                'temperature', 'read_iops', 'write_iops',
                'io_latency', 'throughput_read', 'throughput_write',
                'power_cycles', 'uncorrectable_errors'
            ]
        else:
            self.features = features

        self.scaler = StandardScaler()
        self.feature_scalers = {}  # 用于反标准化

    def prepare_sequences(self, data, task='classification'):
        """
        准备时间序列数据

        参数:
        data: 原始数据
        task: 任务类型 ('classification' 或 'regression')

        返回:
        X, y, drive_info
        """
        sequences = []
        labels = []
        drive_info_list = []

        print("正在准备时间序列数据...")

        # 按硬盘分组
        drive_groups = data.groupby('serial_number')
        total_drives = len(drive_groups)

        for idx, (serial, drive_data) in enumerate(drive_groups, 1):
            # 按日期排序
            drive_data = drive_data.sort_values('day_index')

            # 提取特征
            feature_data = drive_data[self.features].values

            # 获取标签信息
            will_fail = drive_data['will_fail'].iloc[0]
            failure_day = drive_data['failure_day'].iloc[0] if will_fail else len(drive_data) + 100

            # 创建序列
            for i in range(len(drive_data) - self.sequence_length - self.prediction_horizon + 1):
                sequence = feature_data[i:i + self.sequence_length]

                # 根据任务类型生成标签
                if task == 'classification':
                    # 分类任务: 预测未来prediction_horizon天内是否故障
                    current_day = drive_data.iloc[i + self.sequence_length - 1]['days_to_failure']
                    if will_fail and 0 <= current_day <= self.prediction_horizon:
                        label = 1
                    else:
                        label = 0
                elif task == 'regression':
                    # 回归任务: 预测剩余寿命
                    current_day = drive_data.iloc[i + self.sequence_length - 1]['days_to_failure']
                    if will_fail and current_day >= 0:
                        label = current_day
                    else:
                        label = 100  # 不会故障的硬盘给一个大值

                sequences.append(sequence)
                labels.append(label)

                # 保存硬盘信息
                info = {
                    'serial_number': serial,
                    'drive_type': drive_data['drive_type'].iloc[0],
                    'model': drive_data['model'].iloc[0],
                    'sequence_start_day': drive_data.iloc[i]['day_index'],
                    'sequence_end_day': drive_data.iloc[i + self.sequence_length - 1]['day_index'],
                    'days_to_failure_at_end': current_day if will_fail else -1,
                    'will_fail': will_fail
                }
                drive_info_list.append(info)

            if idx % 20 == 0 or idx == total_drives:
                print(f"  已处理 {idx}/{total_drives} 块硬盘...")

        # 转换为numpy数组
        X = np.array(sequences, dtype=np.float32)
        y = np.array(labels, dtype=np.float32 if task == 'regression' else np.int32)

        print(f"生成 {len(X)} 个序列样本")
        print(f"正样本(故障): {sum(y) if task == 'classification' else 'N/A'}")

        return X, y, pd.DataFrame(drive_info_list)

    def fit_transform(self, X, y=None):
        """拟合并转换数据"""
        n_samples, seq_len, n_features = X.shape

        # 重塑为2D用于标准化
        X_reshaped = X.reshape(-1, n_features)

        # 拟合和转换
        X_scaled = self.scaler.fit_transform(X_reshaped)

        # 重塑回3D
        X_scaled = X_scaled.reshape(n_samples, seq_len, n_features)

        return X_scaled

    def transform(self, X):
        """转换数据"""
        n_samples, seq_len, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, seq_len, n_features)
        return X_scaled

    def inverse_transform(self, X_scaled):
        """反标准化"""
        n_samples, seq_len, n_features = X_scaled.shape
        X_reshaped = X_scaled.reshape(-1, n_features)
        X_original = self.scaler.inverse_transform(X_reshaped)
        X_original = X_original.reshape(n_samples, seq_len, n_features)
        return X_original

    def train_test_split_by_drive(self, X, y, drive_info, test_size=0.2, random_state=42):
        """按硬盘划分训练集和测试集"""
        unique_drives = drive_info['serial_number'].unique()
        n_test_drives = int(len(unique_drives) * test_size)

        # 随机选择测试硬盘
        np.random.seed(random_state)
        test_drives = np.random.choice(unique_drives, n_test_drives, replace=False)

        # 创建掩码
        train_mask = ~drive_info['serial_number'].isin(test_drives)
        test_mask = drive_info['serial_number'].isin(test_drives)

        # 划分数据
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        train_info, test_info = drive_info[train_mask].reset_index(drop=True), drive_info[test_mask].reset_index(
            drop=True)

        # 打乱训练集
        shuffle_idx = np.random.permutation(len(X_train))
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]
        train_info = train_info.iloc[shuffle_idx].reset_index(drop=True)

        print(f"\n数据划分结果:")
        print(f"  训练集样本数: {len(X_train):,} (来自 {len(train_info['serial_number'].unique())} 块硬盘)")
        print(f"  测试集样本数: {len(X_test):,} (来自 {len(test_info['serial_number'].unique())} 块硬盘)")

        if len(y_train.shape) == 1:
            print(f"  训练集正样本: {sum(y_train) if y_train.dtype != np.float32 else 'N/A'}")
            print(f"  测试集正样本: {sum(y_test) if y_test.dtype != np.float32 else 'N/A'}")

        return X_train, X_test, y_train, y_test, train_info, test_info

    def save_preprocessor(self, filepath='models/preprocessor.pkl'):
        """保存预处理器"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon,
                'features': self.features,
                'target': self.target,
                'scaler': self.scaler
            }, f)
        print(f"预处理器已保存到: {filepath}")

    @classmethod
    def load_preprocessor(cls, filepath='models/preprocessor.pkl'):
        """加载预处理器"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        preprocessor = cls(
            sequence_length=data['sequence_length'],
            prediction_horizon=data['prediction_horizon'],
            features=data['features'],
            target=data['target']
        )
        preprocessor.scaler = data['scaler']

        print(f"预处理器已从 {filepath} 加载")
        return preprocessor


# ==================== 第四部分: LSTM模型 ====================
class LSTMFailurePredictor:
    """LSTM硬盘故障预测模型"""

    def __init__(self, input_shape, model_type='classification', model_architecture='standard'):
        """
        初始化模型

        参数:
        input_shape: 输入形状 (序列长度, 特征数)
        model_type: 模型类型 ('classification' 或 'regression')
        model_architecture: 模型架构 ('standard', 'bidirectional', 'cnn_lstm')
        """
        self.input_shape = input_shape
        self.model_type = model_type
        self.model_architecture = model_architecture
        self.model = None
        self.history = None

    def build_standard_lstm(self, lstm_units=[128, 64], dropout_rate=0.3,
                            dense_units=[32, 16], learning_rate=0.001):
        """构建标准LSTM模型"""
        model = Sequential(name="Standard_LSTM")

        # 第一层LSTM
        model.add(LSTM(units=lstm_units[0],
                       return_sequences=True if len(lstm_units) > 1 else False,
                       input_shape=self.input_shape,
                       kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

        # 后续LSTM层
        for i, units in enumerate(lstm_units[1:]):
            return_seq = (i < len(lstm_units) - 2)
            model.add(LSTM(units=units,
                           return_sequences=return_seq,
                           kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

        # 全连接层
        for units in dense_units:
            model.add(Dense(units, activation='relu',
                            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate / 2))

        # 输出层
        if self.model_type == 'classification':
            model.add(Dense(1, activation='sigmoid', name='output'))
        else:  # regression
            model.add(Dense(1, activation='relu', name='output'))

        # 编译模型
        if self.model_type == 'classification':
            loss = 'binary_crossentropy'
            metrics = ['accuracy',
                       tf.keras.metrics.AUC(name='auc'),
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.TruePositives(name='tp'),
                       tf.keras.metrics.FalsePositives(name='fp'),
                       tf.keras.metrics.TrueNegatives(name='tn'),
                       tf.keras.metrics.FalseNegatives(name='fn')]
        else:
            loss = 'mse'
            metrics = ['mae', 'mse']

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.model = model
        return model

    def build_bidirectional_lstm(self, lstm_units=[128, 64], dropout_rate=0.3,
                                 dense_units=[32, 16], learning_rate=0.001):
        """构建双向LSTM模型"""
        model = Sequential(name="Bidirectional_LSTM")

        # 双向LSTM层
        model.add(Bidirectional(
            LSTM(units=lstm_units[0], return_sequences=True if len(lstm_units) > 1 else False,
                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            input_shape=self.input_shape
        ))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

        # 后续双向LSTM层
        for i, units in enumerate(lstm_units[1:]):
            return_seq = (i < len(lstm_units) - 2)
            model.add(Bidirectional(
                LSTM(units=units, return_sequences=return_seq,
                     kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))
            ))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

        # 全连接层
        for units in dense_units:
            model.add(Dense(units, activation='relu',
                            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate / 2))

        # 输出层
        if self.model_type == 'classification':
            model.add(Dense(1, activation='sigmoid', name='output'))
        else:
            model.add(Dense(1, activation='relu', name='output'))

        # 编译模型
        if self.model_type == 'classification':
            loss = 'binary_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
        else:
            loss = 'mse'
            metrics = ['mae', 'mse']

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.model = model
        return model

    def build_cnn_lstm(self, conv_filters=[64, 32], kernel_size=3,
                       lstm_units=[64, 32], dropout_rate=0.3,
                       dense_units=[32, 16], learning_rate=0.001):
        """构建CNN-LSTM混合模型"""
        model = Sequential(name="CNN_LSTM")

        # CNN层
        model.add(Conv1D(filters=conv_filters[0], kernel_size=kernel_size,
                         activation='relu', input_shape=self.input_shape,
                         padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))

        if len(conv_filters) > 1:
            model.add(Conv1D(filters=conv_filters[1], kernel_size=kernel_size,
                             activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=2))

        # LSTM层
        for i, units in enumerate(lstm_units):
            return_seq = (i < len(lstm_units) - 1)
            model.add(LSTM(units=units, return_sequences=return_seq,
                           kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
            if i < len(lstm_units) - 1:  # 不是最后一层
                model.add(BatchNormalization())
                model.add(Dropout(dropout_rate))

        # 全连接层
        for units in dense_units:
            model.add(Dense(units, activation='relu',
                            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate / 2))

        # 输出层
        if self.model_type == 'classification':
            model.add(Dense(1, activation='sigmoid', name='output'))
        else:
            model.add(Dense(1, activation='relu', name='output'))

        # 编译模型
        if self.model_type == 'classification':
            loss = 'binary_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
        else:
            loss = 'mse'
            metrics = ['mae', 'mse']

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.model = model
        return model

    def build_model(self, **kwargs):
        """构建模型"""
        if self.model_architecture == 'standard':
            return self.build_standard_lstm(**kwargs)
        elif self.model_architecture == 'bidirectional':
            return self.build_bidirectional_lstm(**kwargs)
        elif self.model_architecture == 'cnn_lstm':
            return self.build_cnn_lstm(**kwargs)
        else:
            raise ValueError(f"未知的模型架构: {self.model_architecture}")

    def train(self, X_train, y_train, X_val, y_val,
              epochs=100, batch_size=32, class_weight=None,
              callbacks=None, validation_split=0.1):
        """训练模型"""

        # 默认回调函数
        if callbacks is None:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15,
                              restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=5, min_lr=1e-6, verbose=1),
                ModelCheckpoint('models/best_model.h5',
                                monitor='val_loss', save_best_only=True),
                TensorBoard(log_dir='logs', histogram_freq=1)
            ]

        # 训练
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )

        return self.history

    def evaluate(self, X_test, y_test, threshold=0.5):
        """评估模型"""
        # 预测
        y_pred_prob = self.model.predict(X_test, verbose=0)

        if self.model_type == 'classification':
            y_pred = (y_pred_prob > threshold).astype(int)

            # 计算评估指标
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_prob)

            # 分类报告
            report = classification_report(y_test, y_pred,
                                           target_names=['正常', '故障'],
                                           output_dict=True)

            # 混淆矩阵
            cm = confusion_matrix(y_test, y_pred)

            return {
                'accuracy': accuracy,
                'auc': auc,
                'classification_report': report,
                'confusion_matrix': cm,
                'y_pred_prob': y_pred_prob.flatten(),
                'y_pred': y_pred.flatten(),
                'threshold': threshold
            }
        else:  # regression
            mae = np.mean(np.abs(y_test - y_pred_prob.flatten()))
            mse = np.mean((y_test - y_pred_prob.flatten()) ** 2)
            rmse = np.sqrt(mse)

            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'predictions': y_pred_prob.flatten(),
                'actual': y_test
            }

    def predict(self, X, threshold=0.5):
        """预测"""
        y_pred_prob = self.model.predict(X, verbose=0)

        if self.model_type == 'classification':
            y_pred = (y_pred_prob > threshold).astype(int)
            return y_pred_prob.flatten(), y_pred.flatten()
        else:
            return y_pred_prob.flatten()

    def plot_training_history(self, save_path=None):
        """绘制训练历史"""
        if self.history is None:
            print("没有训练历史可用")
            return

        history = self.history.history

        # 确定要绘制的指标
        metrics = list(history.keys())
        val_metrics = [m for m in metrics if m.startswith('val_')]
        train_metrics = [m for m in metrics if not m.startswith('val_')]

        # 创建子图
        n_plots = min(len(train_metrics), 4)  # 最多4个子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(train_metrics[:n_plots]):
            ax = axes[idx]
            ax.plot(history[metric], label=f'训练{metric}')

            val_metric = f'val_{metric}'
            if val_metric in history:
                ax.plot(history[val_metric], label=f'验证{metric}')

            ax.set_title(metric)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for idx in range(n_plots, 4):
            axes[idx].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')

        plt.show()

    def save_model(self, filepath='models/hard_drive_predictor.h5'):
        """保存模型"""
        self.model.save(filepath)
        print(f"模型已保存到: {filepath}")

    @classmethod
    def load_model(cls, filepath='models/hard_drive_predictor.h5'):
        """加载模型"""
        model = load_model(filepath)
        predictor = cls(input_shape=model.input_shape[1:])
        predictor.model = model
        print(f"模型已从 {filepath} 加载")
        return predictor


# ==================== 第五部分: 模型评估工具 ====================
class ModelEvaluator:
    """模型评估工具类"""

    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['预测正常', '预测故障'],
                    yticklabels=['实际正常', '实际故障'])
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')

        plt.show()

    def plot_roc_curve(self, y_true, y_pred_prob, save_path=None):
        """绘制ROC曲线"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        auc_score = roc_auc_score(y_true, y_pred_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', label=f'ROC曲线 (AUC = {auc_score:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='随机猜测', alpha=0.5)
        plt.xlabel('假正率 (False Positive Rate)')
        plt.ylabel('真正率 (True Positive Rate)')
        plt.title('ROC曲线')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')

        plt.show()

        return auc_score

    def plot_precision_recall_curve(self, y_true, y_pred_prob, save_path=None):
        """绘制精确率-召回率曲线"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
        avg_precision = average_precision_score(y_true, y_pred_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, 'b-', label=f'PR曲线 (AP = {avg_precision:.3f})', linewidth=2)
        plt.xlabel('召回率 (Recall)')
        plt.ylabel('精确率 (Precision)')
        plt.title('精确率-召回率曲线')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')

        plt.show()

        return avg_precision

    def plot_threshold_analysis(self, y_true, y_pred_prob, save_path=None):
        """绘制阈值分析"""
        thresholds = np.arange(0.1, 1.0, 0.05)
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        for threshold in thresholds:
            y_pred = (y_pred_prob > threshold).astype(int)
            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            accuracies.append(acc)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, accuracies, 'b-', label='准确率', linewidth=2)
        plt.plot(thresholds, precisions, 'r-', label='精确率', linewidth=2)
        plt.plot(thresholds, recalls, 'g-', label='召回率', linewidth=2)
        plt.plot(thresholds, f1_scores, 'm-', label='F1分数', linewidth=2)
        plt.xlabel('阈值')
        plt.ylabel('分数')
        plt.title('阈值分析')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')

        plt.show()

    def analyze_predictions(self, X_test, y_test, test_info, n_examples=5):
        """分析预测结果"""
        y_pred_prob, y_pred = self.model.predict(X_test)

        # 计算基本统计
        print(f"\n预测分析:")
        print(f"  总样本数: {len(y_test)}")
        print(f"  正样本数: {sum(y_test)}")
        print(f"  预测正样本数: {sum(y_pred)}")
        print(f"  准确率: {accuracy_score(y_test, y_pred):.4f}")

        # 分析正确和错误的预测
        correct_idx = np.where(y_pred == y_test)[0]
        incorrect_idx = np.where(y_pred != y_test)[0]

        print(f"\n正确预测: {len(correct_idx)} ({len(correct_idx) / len(y_test) * 100:.1f}%)")
        print(f"错误预测: {len(incorrect_idx)} ({len(incorrect_idx) / len(y_test) * 100:.1f}%)")

        # 分析错误类型
        false_positives = np.where((y_pred == 1) & (y_test == 0))[0]
        false_negatives = np.where((y_pred == 0) & (y_test == 1))[0]

        print(f"  误报(False Positive): {len(false_positives)}")
        print(f"  漏报(False Negative): {len(false_negatives)}")

        # 显示示例
        if len(false_negatives) > 0:
            print(f"\n漏报示例 (应报故障但未报):")
            for i in false_negatives[:min(n_examples, len(false_negatives))]:
                info = test_info.iloc[i]
                print(f"  硬盘: {info['serial_number']}, "
                      f"结束时间距故障: {info['days_to_failure_at_end']}天, "
                      f"预测概率: {y_pred_prob[i]:.3f}")

        if len(false_positives) > 0:
            print(f"\n误报示例 (不应报故障但报了):")
            for i in false_positives[:min(n_examples, len(false_positives))]:
                info = test_info.iloc[i]
                print(f"  硬盘: {info['serial_number']}, "
                      f"预测概率: {y_pred_prob[i]:.3f}")

        return {
            'correct_idx': correct_idx,
            'incorrect_idx': incorrect_idx,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }


# ==================== 第六部分: 部署类 ====================
class HardDrivePredictorDeploy:
    """硬盘故障预测部署类"""

    def __init__(self, model_path='models/best_model.h5',
                 preprocessor_path='models/preprocessor.pkl'):
        """加载模型和预处理器"""
        self.model = load_model(model_path)
        self.preprocessor = DataPreprocessor.load_preprocessor(preprocessor_path)

        print(f"部署模型加载完成")
        print(f"  模型: {model_path}")
        print(f"  预处理器: {preprocessor_path}")
        print(f"  序列长度: {self.preprocessor.sequence_length}")
        print(f"  特征数: {len(self.preprocessor.features)}")

    def prepare_input_data(self, drive_history_df):
        """准备输入数据"""
        # 检查特征是否完整
        missing_features = [f for f in self.preprocessor.features
                            if f not in drive_history_df.columns]

        if missing_features:
            raise ValueError(f"缺少特征: {missing_features}")

        # 确保有足够的数据
        if len(drive_history_df) < self.preprocessor.sequence_length:
            raise ValueError(f"需要至少 {self.preprocessor.sequence_length} 天的数据, "
                             f"当前只有 {len(drive_history_df)} 天")

        # 提取最后sequence_length天的数据
        recent_data = drive_history_df.tail(self.preprocessor.sequence_length)

        # 提取特征
        X = recent_data[self.preprocessor.features].values

        # 标准化
        X_scaled = self.preprocessor.scaler.transform(X)

        # 重塑为LSTM输入格式
        X_input = X_scaled.reshape(1, self.preprocessor.sequence_length, -1)

        return X_input, recent_data

    def predict_single_drive(self, drive_history_df, threshold=0.5):
        """
        预测单块硬盘

        参数:
        drive_history_df: 硬盘历史数据DataFrame
        threshold: 分类阈值

        返回:
        预测结果字典
        """
        try:
            # 准备输入数据
            X_input, recent_data = self.prepare_input_data(drive_history_df)

            # 预测
            failure_prob = float(self.model.predict(X_input, verbose=0)[0][0])

            # 风险评估
            if failure_prob >= 0.8:
                risk_level = "极高风险"
                action = "立即更换硬盘"
                alert_level = "CRITICAL"
            elif failure_prob >= 0.6:
                risk_level = "高风险"
                action = "尽快安排更换(1-3天内)"
                alert_level = "HIGH"
            elif failure_prob >= 0.4:
                risk_level = "中等风险"
                action = "加强监控，准备更换"
                alert_level = "MEDIUM"
            elif failure_prob >= 0.2:
                risk_level = "低风险"
                action = "继续监控"
                alert_level = "LOW"
            else:
                risk_level = "正常"
                action = "常规监控"
                alert_level = "INFO"

            # 获取关键指标
            latest_data = recent_data.iloc[-1]
            key_metrics = {}
            for feature in ['smart_5_raw', 'smart_187_raw', 'smart_197_raw',
                            'smart_233_raw', 'temperature', 'io_latency']:
                if feature in latest_data:
                    key_metrics[feature] = float(latest_data[feature])

            return {
                'success': True,
                'serial_number': drive_history_df['serial_number'].iloc[
                    0] if 'serial_number' in drive_history_df.columns else 'unknown',
                'drive_type': drive_history_df['drive_type'].iloc[
                    0] if 'drive_type' in drive_history_df.columns else 'unknown',
                'failure_probability': failure_prob,
                'risk_level': risk_level,
                'recommended_action': action,
                'alert_level': alert_level,
                'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'last_data_date': recent_data['date'].iloc[-1] if 'date' in recent_data.columns else 'unknown',
                'key_metrics': key_metrics,
                'threshold': threshold
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

    def predict_batch(self, drives_data_dict, threshold=0.5):
        """
        批量预测

        参数:
        drives_data_dict: 字典，键为硬盘ID，值为历史数据DataFrame
        threshold: 分类阈值

        返回:
        预测结果字典
        """
        results = {}

        print(f"开始批量预测 {len(drives_data_dict)} 块硬盘...")

        for idx, (drive_id, drive_data) in enumerate(drives_data_dict.items(), 1):
            result = self.predict_single_drive(drive_data, threshold)
            results[drive_id] = result

            if idx % 10 == 0 or idx == len(drives_data_dict):
                print(f"  已预测 {idx}/{len(drives_data_dict)} 块硬盘...")

        # 统计结果
        successful = sum(1 for r in results.values() if r['success'])
        high_risk = sum(1 for r in results.values() if r.get('risk_level') in ['极高风险', '高风险'])

        print(f"\n批量预测完成:")
        print(f"  成功预测: {successful}/{len(results)}")
        print(f"  高风险硬盘: {high_risk}")

        return results

    def generate_report(self, predictions, output_file='results/reports/prediction_report.html'):
        """生成预测报告"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 统计
        total = len(predictions)
        successful = sum(1 for p in predictions.values() if p['success'])
        high_risk = sum(1 for p in predictions.values()
                        if p.get('risk_level') in ['极高风险', '高风险'])
        medium_risk = sum(1 for p in predictions.values()
                          if p.get('risk_level') == '中等风险')

        # 创建HTML报告
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>硬盘故障预测报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ background: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .risk-critical {{ background: #ffcccc; padding: 10px; margin: 5px 0; }}
                .risk-high {{ background: #ffebcc; padding: 10px; margin: 5px 0; }}
                .risk-medium {{ background: #fff3cd; padding: 10px; margin: 5px 0; }}
                .risk-low {{ background: #d4edda; padding: 10px; margin: 5px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>硬盘故障预测报告</h1>
                <p>生成时间: {timestamp}</p>
            </div>

            <div class="summary">
                <h2>预测摘要</h2>
                <p>总硬盘数: {total}</p>
                <p>成功预测: {successful}</p>
                <p>高风险硬盘: {high_risk}</p>
                <p>中等风险硬盘: {medium_risk}</p>
            </div>

            <h2>详细预测结果</h2>
            <table>
                <tr>
                    <th>硬盘ID</th>
                    <th>硬盘类型</th>
                    <th>故障概率</th>
                    <th>风险等级</th>
                    <th>建议措施</th>
                    <th>最后数据日期</th>
                </tr>
        """

        for drive_id, pred in predictions.items():
            if pred['success']:
                risk_class = f"risk-{pred['alert_level'].lower()}"
                html_content += f"""
                <tr>
                    <td>{drive_id}</td>
                    <td>{pred.get('drive_type', 'N/A')}</td>
                    <td>{pred['failure_probability']:.2%}</td>
                    <td class="{risk_class}">{pred['risk_level']}</td>
                    <td>{pred['recommended_action']}</td>
                    <td>{pred.get('last_data_date', 'N/A')}</td>
                </tr>
                """

        html_content += """
            </table>

            <h2>关键指标说明</h2>
            <ul>
                <li><strong>极高风险(>80%)</strong>: 立即更换硬盘</li>
                <li><strong>高风险(60-80%)</strong>: 尽快安排更换</li>
                <li><strong>中等风险(40-60%)</strong>: 加强监控，准备更换</li>
                <li><strong>低风险(20-40%)</strong>: 继续监控</li>
                <li><strong>正常(<20%)</strong>: 常规监控</li>
            </ul>
        </body>
        </html>
        """

        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"预测报告已生成: {output_file}")

        return output_file


# ==================== 第七部分: 主程序 ====================
def main():
    """主程序"""
    print("=" * 60)
    print("硬盘故障预测系统")
    print("=" * 60)

    # 创建目录
    create_directories()

    # 配置参数
    config = {
        'n_drives': 200,  # 硬盘数量
        'days_per_drive': 365,  # 每块硬盘监控天数
        'failure_rate': 0.3,  # 故障率
        'sequence_length': 30,  # 序列长度
        'prediction_horizon': 7,  # 预测未来天数
        'test_size': 0.2,  # 测试集比例
        'epochs': 50,  # 训练轮数
        'batch_size': 32,  # 批次大小
        'model_architecture': 'standard',  # 模型架构
        'task': 'classification'  # 任务类型
    }

    print(f"\n配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 步骤1: 生成数据
    print(f"\n{'=' * 60}")
    print("步骤1: 生成模拟数据")
    print('=' * 60)

    generator = HardDriveDataGenerator(
        n_drives=config['n_drives'],
        days_per_drive=config['days_per_drive'],
        failure_rate=config['failure_rate']
    )

    data = generator.generate_all_data(save_to_file=True)
    stats = generator.analyze_data(data)

    # 步骤2: 数据预处理
    print(f"\n{'=' * 60}")
    print("步骤2: 数据预处理")
    print('=' * 60)

    preprocessor = DataPreprocessor(
        sequence_length=config['sequence_length'],
        prediction_horizon=config['prediction_horizon'],
        target='will_fail'
    )

    # 准备序列数据
    X, y, drive_info = preprocessor.prepare_sequences(
        data,
        task=config['task']
    )

    # 标准化
    X_scaled = preprocessor.fit_transform(X)

    # 划分数据集
    X_train, X_test, y_train, y_test, train_info, test_info = preprocessor.train_test_split_by_drive(
        X_scaled, y, drive_info,
        test_size=config['test_size']
    )

    # 保存预处理器
    preprocessor.save_preprocessor()

    # 步骤3: 构建和训练模型
    print(f"\n{'=' * 60}")
    print("步骤3: 构建和训练模型")
    print('=' * 60)

    input_shape = (X_train.shape[1], X_train.shape[2])

    # 创建模型
    predictor = LSTMFailurePredictor(
        input_shape=input_shape,
        model_type=config['task'],
        model_architecture=config['model_architecture']
    )

    # 构建模型
    model = predictor.build_model(
        lstm_units=[128, 64],
        dropout_rate=0.3,
        dense_units=[32, 16],
        learning_rate=0.001
    )

    # 显示模型摘要
    model.summary()

    # 训练模型
    print("\n开始训练模型...")
    history = predictor.train(
        X_train, y_train,
        X_val=X_test,  # 使用测试集作为验证集
        y_val=y_test,
        epochs=config['epochs'],
        batch_size=config['batch_size']
    )

    # 绘制训练历史
    predictor.plot_training_history(save_path='results/plots/training_history.png')

    # 保存模型
    predictor.save_model('models/hard_drive_predictor.h5')

    # 步骤4: 模型评估
    print(f"\n{'=' * 60}")
    print("步骤4: 模型评估")
    print('=' * 60)

    # 评估模型
    results = predictor.evaluate(X_test, y_test, threshold=0.5)

    print(f"\n模型评估结果:")
    if config['task'] == 'classification':
        print(f"  准确率: {results['accuracy']:.4f}")
        print(f"  AUC: {results['auc']:.4f}")

        # 显示分类报告
        print(f"\n分类报告:")
        report_df = pd.DataFrame(results['classification_report']).transpose()
        print(report_df.to_string())

        # 创建评估器
        evaluator = ModelEvaluator(predictor, preprocessor)

        # 绘制评估图表
        evaluator.plot_confusion_matrix(y_test, results['y_pred'],
                                        save_path='results/plots/confusion_matrix.png')

        evaluator.plot_roc_curve(y_test, results['y_pred_prob'],
                                 save_path='results/plots/roc_curve.png')

        evaluator.plot_precision_recall_curve(y_test, results['y_pred_prob'],
                                              save_path='results/plots/pr_curve.png')

        # 分析预测结果
        analysis = evaluator.analyze_predictions(X_test, y_test, test_info, n_examples=5)

    else:  # regression
        print(f"  MAE: {results['mae']:.2f} 天")
        print(f"  RMSE: {results['rmse']:.2f} 天")

        # 绘制预测 vs 实际
        plt.figure(figsize=(10, 6))
        plt.scatter(results['actual'], results['predictions'], alpha=0.5)
        plt.plot([0, max(results['actual'])], [0, max(results['actual'])], 'r--')
        plt.xlabel('实际剩余寿命(天)')
        plt.ylabel('预测剩余寿命(天)')
        plt.title('预测 vs 实际')
        plt.grid(True, alpha=0.3)
        plt.savefig('results/plots/predictions_vs_actual.png', dpi=100, bbox_inches='tight')
        plt.show()

    # 步骤5: 部署演示
    print(f"\n{'=' * 60}")
    print("步骤5: 部署演示")
    print('=' * 60)

    # 加载部署模型
    try:
        deploy_predictor = HardDrivePredictorDeploy(
            model_path='models/hard_drive_predictor.h5',
            preprocessor_path='models/preprocessor.pkl'
        )

        # 从测试集中选择几块硬盘进行演示
        demo_drives = {}
        test_serials = test_info['serial_number'].unique()[:5]

        for serial in test_serials:
            drive_data = data[data['serial_number'] == serial]
            demo_drives[serial] = drive_data

        # 批量预测
        predictions = deploy_predictor.predict_batch(demo_drives, threshold=0.5)

        # 显示预测结果
        print(f"\n演示预测结果:")
        for drive_id, pred in predictions.items():
            if pred['success']:
                print(f"\n硬盘: {drive_id}")
                print(f"  故障概率: {pred['failure_probability']:.2%}")
                print(f"  风险等级: {pred['risk_level']}")
                print(f"  建议措施: {pred['recommended_action']}")

                # 显示关键指标
                if 'key_metrics' in pred:
                    print(f"  关键指标:")
                    for metric, value in pred['key_metrics'].items():
                        print(f"    {metric}: {value:.2f}")

        # 生成报告
        report_file = deploy_predictor.generate_report(
            predictions,
            output_file='results/reports/demo_prediction_report.html'
        )

        print(f"\n演示完成! 报告已生成: {report_file}")

    except Exception as e:
        print(f"部署演示失败: {e}")

    # 步骤6: 保存配置和结果
    print(f"\n{'=' * 60}")
    print("步骤6: 保存配置和结果")
    print('=' * 60)

    # 保存配置
    config['data_stats'] = stats
    if config['task'] == 'classification':
        config['model_results'] = {
            'accuracy': float(results['accuracy']),
            'auc': float(results['auc'])
        }
    else:
        config['model_results'] = {
            'mae': float(results['mae']),
            'rmse': float(results['rmse'])
        }

    config['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open('results/reports/config.json', 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"配置和结果已保存到: results/reports/config.json")

    # 生成最终报告
    print(f"\n{'=' * 60}")
    print("系统运行完成!")
    print('=' * 60)

    print(f"\n生成的文件:")
    print(f"  1. 原始数据: data/generated/hard_drive_data_*.csv")
    print(f"  2. 预处理数据: data/processed/")
    print(f"  3. 模型文件: models/hard_drive_predictor.h5")
    print(f"  4. 预处理器: models/preprocessor.pkl")
    print(f"  5. 评估图表: results/plots/")
    print(f"  6. 预测报告: results/reports/demo_prediction_report.html")
    print(f"  7. 配置和结果: results/reports/config.json")

    print(f"\n下一步:")
    print(f"  1. 使用真实数据替换模拟数据")
    print(f"  2. 调整模型参数以获得更好的性能")
    print(f"  3. 将模型集成到监控系统中")
    print(f"  4. 定期重新训练模型以适应新数据")


# ==================== 运行主程序 ====================
if __name__ == "__main__":
    # 检查TensorFlow版本
    print(f"TensorFlow版本: {tf.__version__}")

    # 运行主程序
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n硬盘故障预测系统运行结束!")