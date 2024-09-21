import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# 读取数据
data = pd.read_excel('001.xlsx', sheet_name=None)

# 合并四个材料的数据
df_list = []
for sheet_name, df in data.items():
    df['material'] = sheet_name  # 添加材料类别
    df_list.append(df)
df = pd.concat(df_list, ignore_index=True)

# 重命名列
df.columns = ['Temperature', 'Frequency', 'Core Loss', 'Waveform'] + [f'B{i}' for i in range(1, 1025)] 

# 独立分析各个因素对磁芯损耗的影响
# 温度对磁芯损耗的影响
temp_loss = df.groupby('Temperature')['Core Loss'].mean()

# 波形对磁芯损耗的影响
waveform_loss = df.groupby('Waveform')['Core Loss'].mean()

# 材料对磁芯损耗的影响
material_loss = df.groupby('Material')['Core Loss'].mean()

# 协同作用分析：温度和波形、温度和材料、波形和材料
temp_waveform_loss = df.groupby(['Temperature', 'Waveform'])['Core Loss'].mean().unstack()
temp_material_loss = df.groupby(['Temperature', 'Material'])['Core Loss'].mean().unstack()
waveform_material_loss = df.groupby(['Waveform', 'Material'])['Core Loss'].mean().unstack()

# 绘制分析结果
fig, ax = plt.subplots(2, 2, figsize=(15, 12))

# 温度对磁芯损耗影响
temp_loss.plot(kind='bar', ax=ax[0, 0], title='Effect of Temperature on Core Loss')
ax[0, 0].set_ylabel('Core Loss (W/m^3)')

# 波形对磁芯损耗影响
waveform_loss.plot(kind='bar', ax=ax[0, 1], title='Effect of Waveform on Core Loss')
ax[0, 1].set_ylabel('Core Loss (W/m^3)')

# 温度和波形协同作用
sns.heatmap(temp_waveform_loss, cmap='coolwarm', annot=True, ax=ax[1, 0])
ax[1, 0].set_title('Interaction Between Temperature and Waveform')

# 温度和材料协同作用
sns.heatmap(temp_material_loss, cmap='coolwarm', annot=True, ax=ax[1, 1])
ax[1, 1].set_title('Interaction Between Temperature and Material')

plt.tight_layout()
plt.show()

# 线性回归分析影响程度
# 对温度、波形、材料进行编码
encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(df[['Waveform', 'Material']])

# 创建模型并进行训练
X = np.concatenate([df[['Temperature', 'Frequency']].values, encoded_features], axis=1)
y = df['Core Loss'].values

model = LinearRegression()
model.fit(X, y)

# 输出各个因素的回归系数，表示其对损耗的影响程度
coefficients = model.coef_
intercept = model.intercept_
print(f'Intercept: {intercept}')
print(f'Coefficients: {coefficients}')
