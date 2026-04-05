import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import matplotlib as mpl
import os

# ====================== 1. 彻底解决中文乱码：多字体兜底+全局配置 ======================
# 清除matplotlib字体缓存，避免旧配置影响
cache_dir = mpl.get_cachedir()
font_cache_path = os.path.join(cache_dir, 'fontlist-v330.json')
if os.path.exists(font_cache_path):
    os.remove(font_cache_path)

# 配置中文字体：多字体兜底，确保matplotlib一定能找到可用字体
plt.rcParams["font.family"] = ["Arial Unicode MS", "PingFang SC", "WenQuanYi Micro Hei"]
plt.rcParams['axes.unicode_minus'] = False
# 解决负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# 全局字体大小优化，确保中文清晰
plt.rcParams['font.size'] = 10
# 图表样式
plt.style.use('default')

warnings.filterwarnings('ignore')

# ====================== 2. 读取数据（保持原路径不变） ======================
file_path = '/Users/Apple/PycharmProjects/Some-simulated-datasets-for-lstm/SSD/generated/hard_drive_data_20260405_130457.csv'
df = pd.read_csv(file_path)

# 转换日期格式
df['date'] = pd.to_datetime(df['date'])

# ====================== 3. 创建综合分析图表（保持原逻辑不变） ======================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('硬盘监控数据集综合分析', fontsize=16, fontweight='bold', y=0.98)

# 1. 硬盘类型分布（饼图）
drive_type_counts = df['drive_type'].value_counts()
colors1 = ['#FF6B6B', '#4ECDC4']
axes[0,0].pie(drive_type_counts.values, labels=drive_type_counts.index, autopct='%1.1f%%',
              colors=colors1, startangle=90, textprops={'fontsize': 10})
axes[0,0].set_title('硬盘类型分布', fontsize=12, fontweight='bold', pad=20)

# 2. 硬盘状态分布（柱状图）
status_counts = df['status'].value_counts()
colors2 = ['#95E1D3', '#F38181']
bars = axes[0,1].bar(status_counts.index, status_counts.values, color=colors2, alpha=0.8)
axes[0,1].set_title('硬盘状态分布', fontsize=12, fontweight='bold', pad=20)
axes[0,1].set_ylabel('数量')
# 在柱子上添加数值标签
for bar in bars:
    height = bar.get_height()
    axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 100,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10)

# 3. 故障预测分布（will_fail）
fail_counts = df['will_fail'].value_counts().sort_index()
labels_fail = ['正常', '即将故障']
colors3 = ['#6C5CE7', '#FD79A8']
bars3 = axes[0,2].bar(labels_fail, fail_counts.values, color=colors3, alpha=0.8)
axes[0,2].set_title('硬盘故障预测分布', fontsize=12, fontweight='bold', pad=20)
axes[0,2].set_ylabel('数量')
# 添加数值标签
for bar in bars3:
    height = bar.get_height()
    axes[0,2].text(bar.get_x() + bar.get_width()/2., height + 100,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10)

# 4. 温度分布（直方图）
axes[1,0].hist(df['temperature'], bins=30, color='#FFB86C', alpha=0.7, edgecolor='black')
axes[1,0].set_title('硬盘温度分布', fontsize=12, fontweight='bold', pad=20)
axes[1,0].set_xlabel('温度 (°C)')
axes[1,0].set_ylabel('频次')
axes[1,0].axvline(df['temperature'].mean(), color='red', linestyle='--',
                  label=f'平均值: {df["temperature"].mean():.1f}°C')
axes[1,0].legend()

# 5. 天数故障分布（只显示有故障风险的数据）
failure_data = df[df['days_to_failure'] >= 0]
if len(failure_data) > 0:
    axes[1,1].hist(failure_data['days_to_failure'], bins=20, color='#74B9FF', alpha=0.7, edgecolor='black')
    axes[1,1].set_title('故障前剩余天数分布', fontsize=12, fontweight='bold', pad=20)
    axes[1,1].set_xlabel('故障前剩余天数')
    axes[1,1].set_ylabel('频次')
    axes[1,1].axvline(failure_data['days_to_failure'].mean(), color='darkblue', linestyle='--',
                      label=f'平均值: {failure_data["days_to_failure"].mean():.1f}天')
    axes[1,1].legend()
else:
    axes[1,1].text(0.5, 0.5, '无故障风险数据', ha='center', va='center',
                   transform=axes[1,1].transAxes, fontsize=12)
    axes[1,1].set_title('故障前剩余天数分布', fontsize=12, fontweight='bold', pad=20)

# 6. 硬盘型号分布（前5个型号）
top_models = df['model'].value_counts().head(5)
colors6 = plt.cm.Set3(np.linspace(0, 1, len(top_models)))
bars6 = axes[1,2].barh(range(len(top_models)), top_models.values, color=colors6, alpha=0.8)
axes[1,2].set_yticks(range(len(top_models)))
axes[1,2].set_yticklabels(top_models.index, fontsize=9)
axes[1,2].set_title('硬盘型号分布（前5名）', fontsize=12, fontweight='bold', pad=20)
axes[1,2].set_xlabel('数量')
# 添加数值标签
for i, bar in enumerate(bars6):
    width = bar.get_width()
    axes[1,2].text(width + 50, bar.get_y() + bar.get_height()/2.,
                   f'{int(width)}', ha='left', va='center', fontsize=9)

plt.tight_layout()
# ====================== 4. 修复保存路径：确保生成正确的PNG文件 ======================
save_path = '/Users/Apple/PycharmProjects/Some-simulated-datasets-for-lstm/hard_drive_analysis_fixed.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

# ====================== 5. 输出关键统计信息（保持原逻辑不变） ======================
print("=" * 70)
print("硬盘监控数据集关键统计信息")
print("=" * 70)

# 1. 基本信息
print(f"1. 数据规模:")
print(f"   - 总记录数: {len(df):,} 条")
print(f"   - 总字段数: {df.shape[1]} 个")
print(f"   - 监控时间范围: {df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')}")
print(f"   - 涉及硬盘数量: {df['serial_number'].nunique()} 个")

# 2. 硬盘类型分析
print(f"\n2. 硬盘类型分布:")
for drive_type, count in df['drive_type'].value_counts().items():
    percentage = (count / len(df)) * 100
    print(f"   - {drive_type}: {count:,} 条记录 ({percentage:.1f}%)")

# 3. 故障风险分析
print(f"\n3. 故障风险分析:")
fail_yes = len(df[df['will_fail'] == 1])
fail_no = len(df[df['will_fail'] == 0])
fail_percentage = (fail_yes / len(df)) * 100
print(f"   - 正常硬盘记录: {fail_no:,} 条 ({100-fail_percentage:.1f}%)")
print(f"   - 即将故障硬盘记录: {fail_yes:,} 条 ({fail_percentage:.1f}%)")

# 4. 温度统计
print(f"\n4. 硬盘温度统计:")
print(f"   - 平均温度: {df['temperature'].mean():.1f}°C")
print(f"   - 最高温度: {df['temperature'].max()}°C")
print(f"   - 最低温度: {df['temperature'].min()}°C")
print(f"   - 温度中位数: {df['temperature'].median():.1f}°C")

# 5. 性能指标统计
print(f"\n5. 主要性能指标统计:")
print(f"   - 平均读取IOPS: {df['read_iops'].mean():.1f}")
print(f"   - 平均写入IOPS: {df['write_iops'].mean():.1f}")
print(f"   - 平均IO延迟: {df['io_latency'].mean():.2f} ms")
print(f"   - 平均读取吞吐量: {df['throughput_read'].mean():.2f} MB/s")
print(f"   - 平均写入吞吐量: {df['throughput_write'].mean():.2f} MB/s")

# 6. SMART指标概况
smart_cols = [col for col in df.columns if 'smart_' in col]
print(f"\n6. SMART指标监控:")
print(f"   - 共监控 {len(smart_cols)} 个SMART指标")
print(f"   - 主要监控指标包括: smart_1 (读取错误率), smart_5 (重映射扇区数), "
      f"smart_9 (通电时间), smart_187 (不可纠正错误), smart_197 (坏道计数) 等")

# 7. 缺失值情况
print(f"\n7. 数据质量分析:")
missing_cols = df.columns[df.isnull().sum() > 0]
if len(missing_cols) > 0:
    print(f"   - 存在缺失值的字段数量: {len(missing_cols)} 个")
    print(f"   - 缺失最严重的字段:")
    missing_stats = df[missing_cols].isnull().sum().sort_values(ascending=False).head(3)
    for col, count in missing_stats.items():
        percentage = (count / len(df)) * 100
        print(f"     * {col}: {count:,} 个缺失值 ({percentage:.1f}%)")
else:
    print(f"   - 数据完整性良好，无缺失值")

print("\n" + "=" * 70)
print(f"图表已保存为: {save_path}")
print("=" * 70)