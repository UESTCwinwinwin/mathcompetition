import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

#1123
# 从Excel表格中读取数据，假设文件名为 'core_loss_data.xlsx'
data = pd.read_excel('002.xlsx')

# 提取温度、频率、核心损耗和励磁波形类型列
temperature = data.iloc[:, 0]
frequency = data.iloc[:, 1]
core_loss = data.iloc[:, 2]
excitation_waveform = data.iloc[:, 3]

# 提取磁通密度列（第5列到第1029列，即磁通密度 flux_density_1 到 flux_density_1024）
flux_density_columns = data.columns[4:1029]
flux_density_data = data[flux_density_columns]

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
# 准备测试集特征（X_test）
X_test = features
# 将波形的文字与label的数字对应
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(excitation_waveform)
# 输出类别和对应的标签
print("类别与标签的对应关系:")
for idx, label in enumerate(label_encoder.classes_):
    print(f"{label}: {idx}")

# 加载保存的 SVM 模型
loaded_svm_model = joblib.load('svm_model.pkl')
    
# 使用加载的模型进行预测
y_pred_loaded = loaded_svm_model.predict(X_test)

# 打印预测结果进行验证
print('使用加载的模型进行预测的分类报告:')
print(classification_report(labels, y_pred_loaded))
print('使用加载的模型进行预测的混淆矩阵:')
print(confusion_matrix(labels, y_pred_loaded))



