import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_csv('./HousingData.csv')

print(type(data))
print(data.shape)

print(data.head())
data.info()
print(data.describe())
print(data.loc[:,['CRIM','ZN']])
print(data.iloc[:,[1,3]])

print(data[(data.CRIM > 0.1) & (data.ZN > 0.)])

print(data.dropna().shape)

data_fillna = data.fillna(0.)
print(data_fillna.shape)
print(data.shape)

print(data.mean()) # 평균
print(data.std()) # 표준편차
print(data.median()) # 중앙값

#data.boxplot()
#plt.show()
#data.boxplot(column=['CRIM','ZN','LSTAT'])
#data.loc[:,'CRIM'].plot.bar()
#pd.plotting.scatter_matrix(data, diagonal='kde')
#corr = data.corr() # 특징간의 연관성 피어슨 상관관계(1~-1) 1에 가까울수록 양의 상관관계, -1에 가까운수록 음의 상관관계, 0에 가까울수록 관련이 없다
#양의 상관관계 => 특징이 증가하면 다른 특징도 증가
#음의 상관관계 => 특징이 증가하면 다른 특징은 감소
#sns.heatmap(corr)
#plt.show()

data_nd = data.to_numpy()
print(type(data_nd))
print(data_nd.dtype)

fig  = plt.figure()
plt.scatter(data_nd[:,0],data_nd[:,-1],c='r')
plt.title('scatter')
plt.xlabel('CRIM')
plt.ylabel('MEDV')
fig.show()
plt.show()
