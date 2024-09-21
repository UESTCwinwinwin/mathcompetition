import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# 读取 Excel 数据
data = pd.read_excel('附件一（训练集）.xlsx')

# 提取特征（第1到第4列）和磁通密度列（第5到第1029列）
leixing = data.iloc[:, 0]
temperature = data.iloc[:, 1]
frequency = data.iloc[:, 2]
core_loss = data.iloc[:, 3]
excitation_waveform = data.iloc[:, 4]
flux_density_columns = data.columns[5:1030]
flux_density_data = data[flux_density_columns]

def extract_features(flux_density):
        features = {
            'max_value': flux_density.max(),
            'min_value': flux_density.min(),
            'mean_value': flux_density.mean(),
            'std_value': flux_density.std(),
            'peak_to_peak': flux_density.max() - flux_density.min(),
            'skewness': flux_density.skew(),
            'kurtosis': flux_density.kurtosis()
        }
        return pd.Series(features)
features = flux_density_data.apply(extract_features, axis=1)

# 将励磁波形类型（正弦波、三角波、梯形波）转换为数值编码
label_encoder = LabelEncoder()
excitation_waveform_encoded = label_encoder.fit_transform(excitation_waveform)
leixing_encoded = label_encoder.fit_transform(leixing)
features['temperature'] = temperature
features['frequency'] = frequency
features['excitation_waveform'] = excitation_waveform_encoded
features['leixing'] = leixing_encoded
features['core_loss'] = core_loss
# 构建特征矩阵 X 和目标变量 y
X = features
y = core_loss  # 目标变量是磁芯损耗

# 将 X 的所有列名转换为字符串
X.columns = X.columns.astype(str)
# 将 X 的所有列名转换为字符串
features.columns = features.columns.astype(str)


# 创建列联表，分析励磁波形类型与磁芯损耗的关系
contingency_table_waveform = pd.crosstab(index=features['excitation_waveform'], columns=features['core_loss'])
print("\n励磁波形类型与磁芯损耗的列联表:\n", contingency_table_waveform)

# 进行卡方检验
chi2_waveform, p_waveform, dof_waveform, expected_waveform = chi2_contingency(contingency_table_waveform)
print("\n励磁波形类型的卡方检验结果：")
print("Chi2值:", chi2_waveform)
print("p值:", p_waveform)
print("自由度:", dof_waveform)
print("期望频数:\n", expected_waveform)

# 创建列联表，分析温度与磁芯损耗的关系
contingency_table_temp = pd.crosstab(index=features['temperature'], columns=features['core_loss'])
print("\n温度与磁芯损耗的列联表:\n", contingency_table_temp)

# 进行卡方检验
chi2_temp, p_temp, dof_temp, expected_temp = chi2_contingency(contingency_table_temp)
print("\n温度的卡方检验结果：")
print("Chi2值:", chi2_temp)
print("p值:", p_temp)
print("自由度:", dof_temp)
print("期望频数:\n", expected_temp)

# 可以在此处加入更多的分析，例如频率与磁芯损耗的关系等
# 创建列联表，分析温度与磁芯损耗的关系
contingency_table_leixing = pd.crosstab(index=features['leixing'], columns=features['core_loss'])
print("\n材料与磁芯损耗的列联表:\n", contingency_table_leixing)

# 进行卡方检验
chi2_temp, p_temp, dof_temp, expected_temp = chi2_contingency(contingency_table_leixing)
print("\n材料的卡方检验结果：")
print("Chi2值:", chi2_temp)
print("p值:", p_temp)
print("自由度:", dof_temp)
print("期望频数:\n", expected_temp)
# 保存处理后的数据（如果需要）
#data.to_csv('combined_data.csv', index=False)
