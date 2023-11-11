import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

housing = pd.read_csv('./california_housing_train.csv')


housing.head() # 첫 다섯 행 출력
housing.info() # 전체 테이블 구조
housing.describe() # 요약 정보

## 히스토그램
#housing.hist(bins=50, figsize=(20,15))

## 산점도
#housing.plot(kind='scatter',x='longitude',y='latitude')

## 산점도에 투명도 추가
#housing.plot(kind='scatter',x='longitude',y='latitude', alpha=0.1)

## 
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#              s=housing["population"]/100, label="population", figsize=(10,7),
#              c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
#              sharex=False)
#plt.legend()


corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
print(corr_matrix)

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
# plt.axis([0, 16, 0, 550000])

# plt.show()

housing_copy = housing.copy()

housing = housing_copy.drop("median_house_value", axis=1) # 훈련 세트를 위해 레이블 삭제
housing_labels = housing_copy["median_house_value"].copy()

# 열 인덱스
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # *args 또는 **kargs 없음
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # 아무것도 하지 않습니다
    def transform(self, X):
        #print("Step1")
        #print(X)
        #print("Step2")
        #print(X[:, rooms_ix])
        #print("Step3")
        #print(X[:, households_ix])
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        #print("Step4")
        #print(rooms_per_household)
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


#housing
#attr_adder
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

full_pipeline = Pipeline([
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_prepared = full_pipeline.fit_transform(housing.values)
#housing_prepared.shape
#housing_prepared = full_pipeline.fit_transform(housing)

print("Step1")
print(housing)
print("Step2")
print(housing.values)
print("Step3")
print(housing_prepared)


## step 12
lin_reg = LinearRegression()

lin_reg.fit(housing_prepared, housing_labels)
# print(housing_prepared)
# print(housing_labels)

# 훈련 샘플 몇 개를 사용해 전체 파이프라인을 적용해 보겠습니다
#print("step 1")
#print(housing)
#print("step 2")
#print(housing.iloc[:5])
#print("step 3")
#print(housing[:,3])

#some_data_prepared = housing_prepared[:5]
#some_labels = housing_labels[:5]

## step 13
# some_data_prepared = housing_prepared[15:20]
# some_labels = housing_labels[15:20]

# print("Step1")
# print(some_data_prepared)
# print("Step2")
# print(some_labels)


# print("예측:", lin_reg.predict(some_data_prepared))
# print("레이블:", some_labels)



## step 14
# housing_predictions = lin_reg.predict(housing_prepared)
# lin_mse = mean_squared_error(housing_labels, housing_predictions)
# lin_rmse = np.sqrt(lin_mse)
# lin_rmse

# print(lin_rmse)

## step 15
housing_test = pd.read_csv('./california_housing_test.csv')
housing_test_copy = housing_test.copy()

test_data = housing_test_copy.drop("median_house_value", axis=1) # 훈련 세트를 위해 레이블 삭제
test_labels = housing_test_copy["median_house_value"].copy()

print("Step1")
print(test_data)
print("Step2")
print(test_data.values)

test_prepared = full_pipeline.fit_transform(test_data.values)

print("Step3")
print(test_prepared)

predict_result = lin_reg.predict(test_prepared)

print("예측:", predict_result[:5])
print("레이블:", test_labels[:5])

test_predictions = lin_reg.predict(test_prepared)
lin_mse = mean_squared_error(test_labels, test_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

print(lin_mse)


tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

some_data_prepared = housing_prepared[:5]
some_labels = housing_labels.iloc[:5]

#some_data_prepared = full_pipeline.transform(some_data.values)
#some_data_prepared = full_pipeline.fit_transform(some_data.values)
print("step2")
print(some_data_prepared)
#full_pipeline.fit_transform(test_data.values)
some_predictions = tree_reg.predict(some_data_prepared)
print("예측:", some_predictions)
print("레이블:", some_labels)

tree_mse = mean_squared_error(some_labels, some_predictions)
tree_rmse = np.sqrt(tree_mse)
print("tree_rmse: ",tree_rmse)


test_prepared = full_pipeline.fit_transform(test_data.values)

predict_result = tree_reg.predict(test_prepared)

print("예측:", predict_result[:5])
print("레이블:", test_labels[:5])

tree_mse = mean_squared_error(test_labels, predict_result)
tree_rmse = np.sqrt(tree_mse)
print("tree_rmse: ",tree_rmse)