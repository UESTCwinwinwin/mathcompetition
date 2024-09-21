import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# 从Excel表格中读取数据，假设文件名为 'core_loss_data.xlsx'
data = pd.read_excel('001.xlsx')
print("1")
# 提取温度、频率、核心损耗和励磁波形类型列
temperature = data.iloc[:, 0]
frequency = data.iloc[:, 1]
core_loss = data.iloc[:, 2]
excitation_waveform = data.iloc[:, 3]
print("2")
# 提取磁通密度列（第5列到第1029列，即磁通密度 flux_density_1 到 flux_density_1024）
flux_density_columns = data.columns[4:1029]
flux_density_data = data[flux_density_columns]
print("2")
# 特征提取函数
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

# 应用特征提取到磁通密度数据
features = flux_density_data.apply(extract_features, axis=1)
print("4")
# 合并温度、频率、核心损耗数据到提取的特征中
#features['temperature'] = temperature
#features['frequency'] = frequency
#features['core_loss'] = core_loss
print("5")
# 准备目标变量：励磁波形类型
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(excitation_waveform)
print("6")
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 训练SVM模型
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)
print("rbf7")
# 进行预测
y_pred1 = svm_model.predict(X_test)
# 保存模型到文件
joblib.dump(svm_model, 'svm_model.pkl')
# 输出分类报告和混淆矩阵
print('分类报告:')
print(classification_report(y_test, y_pred1, target_names=label_encoder.classes_))
print('混淆矩阵:')
print(confusion_matrix(y_test, y_pred1))
