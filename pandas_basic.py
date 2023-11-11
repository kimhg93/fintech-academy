import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# 시리즈란 데이터가 순차적으로 나열된 1차원 배열이며, 인덱스(Index)와 값(Value)은 일대일 대응 관계이며, 이는 키(key)와 값(value)이 ‘{key:value}’ 형태로 구성된 딕셔너리와 비슷하다.
dict_data = {'a': 1, 'b': 2, 'c': 3}
series = pd.Series(dict_data)

print(series)

list_data = ['a', 'b', 'c']
series_2 = pd.Series(list_data)

print(series_2)

series_3 = pd.Series(list_data, index=['index1', 'index2', 'index3'])

print(series_3)