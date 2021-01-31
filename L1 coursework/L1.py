# action1
print(sum(list(range(2,102,2))))
print('\n')

# action2
import pandas as pd
df = pd.DataFrame([[68,65,30],[95,76,98],[98,86,88],[90,88,77],[80,90,90]], index=['zhangfei', 'guanyu', 'liubei', 'dianwei', 'xuchu'], columns=['yuwen','shuxue','yingyu'])

# calculate mean, min, max, var, std
df.loc['mean'] = df.mean()
df.loc['min'] = df.min()
df.loc['max'] = df.max()
df.loc['var'] = df.var()
df.loc['std'] = df.std()

# sorted by sum in descending order
df['sum'] = df.iloc[0:5].apply(lambda x:x.sum(), axis=1)
print(df.sort_values(by='sum', ascending=False))
print('\n')

# action3
# load csv file
df1 = pd.read_csv('./car_data_analyze/car_complain.csv')

# clean data
df1['brand'] = df1['brand'].apply(lambda x:x.replace('一汽-大众', '一汽大众'))
df1 = df1.drop('problem', axis=1).join(df1.problem.str.get_dummies(','))
tags = df1.columns[7:]

# sorted by brand in descending order
df2 = df1.groupby(['brand'])['id'].agg(['count'])
df3 = df1.groupby(['brand'])[tags].agg(['sum'])
df3 = df2.merge(df3, left_index=True, right_index=True, how='left')
df3 = df3.sort_values('count', ascending=False)
df3.reset_index(inplace=True)
print('brand with the most complaints\n', df3[['brand', 'count']].iloc[0])
print('\n')

# sorted by car_model in descending order
df4 = df1.groupby(['brand', 'car_model'])['id'].agg(['count'])
df5 = df1.groupby(['brand', 'car_model'])[tags].agg(['sum'])
df5 = df4.merge(df5, left_index=True, right_index=True, how='left')
df5 = df5.sort_values('count', ascending=False)
df5.reset_index(inplace=True)
print('car model with the most complaints\n', df5[['brand', 'car_model', 'count']].iloc[0])
print('\n')

# sorted by average complaints per car model in descending order
df2['car_model_count'] = df1.groupby(['brand'])['car_model'].unique().apply(lambda x: len(x))
df2['average complaints per car model'] = df2['count'] / df2['car_model_count']
df2 = df2.sort_values('average complaints per car model', ascending=False)
df2.reset_index(inplace=True)
print('brand with the most average complaints per car model\n', df2[['brand', 'average complaints per car model']].iloc[0])