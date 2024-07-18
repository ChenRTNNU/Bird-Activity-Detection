import pandas as pd
from sklearn.model_selection import train_test_split

# 读取CSV文件
csv_file_path1 = 'D:\\DeepLearning\\bird_activity_detection\\labels\\ff1010bird.csv'
df1 = pd.read_csv(csv_file_path1)
csv_file_path2 = 'D:\\DeepLearning\\bird_activity_detection\\labels\\warblrb10k.csv'
df2 = pd.read_csv(csv_file_path2)

# 筛选数据
df1P = df1[df1['hasbird'] == 1]
df1N = df1[df1['hasbird'] == 0].sample(n=2000)
df2P = df2[df2['hasbird'] == 1].sample(n=2000)
df2N = df2[df2['hasbird'] == 0]
df1_2k = pd.merge(df1P, df1N, how='outer')
df2_2k = pd.merge(df2P, df2N, how='outer')

# 划分数据集
train_df1, test_df1 = train_test_split(df1_2k, test_size=0.3, shuffle=True, random_state=42)
training_df1, valid_df1 = train_test_split(train_df1, test_size=0.3, shuffle=True, random_state=42)

train_df2, test_df2 = train_test_split(df2_2k, test_size=0.3, shuffle=True, random_state=42)

# 保存数据集到CSV文件
training_df1.to_csv('dataset/train_f.csv', index=False)
valid_df1.to_csv('dataset/valid_f.csv', index=False)
test_df1.to_csv('dataset/test_f.csv', index=False)
train_df2.to_csv('dataset/train_w.csv', index=False)
test_df2.to_csv('dataset/test_w.csv', index=False)
