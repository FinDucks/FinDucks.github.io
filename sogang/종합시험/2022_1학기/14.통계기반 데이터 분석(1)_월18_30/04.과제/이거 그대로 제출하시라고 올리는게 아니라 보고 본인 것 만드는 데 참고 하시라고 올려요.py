import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = [10, 6]
#%matplotlib inline
import math
from scipy import stats
import statistics
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
import statsmodels.api as sm
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objs as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from matplotlib.pylab import rcParams
from datetime import datetime
from dateutil.parser import parse
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA

# "diet.csv" 
data=pd.read_csv("diet.csv", index_col = 0)
# data.drop(['Unnamed: 0'], axis = 1, inplace = True)
print(data)
#"diet.csv" 

plt.boxplot(data)
plt.show()

#T 통계량 계산
mean_diff = np.mean(data)
print(mean_diff)
# mean_diff <- mean(diff)
# mean_diff

sd_diff = np.std(data)
sd_diff
# sd_diff <- sd(diff)
# sd_diff

t_stat = mean_diff/(sd_diff/math.sqrt(len(data)))
print(t_stat)
# t_stat <- mean_diff/(sd_diff/sqrt(length(diff)))
# t_stat



t_stat, p_val = stats.ttest_ind(data["Before"], data["After"], equal_var=True, alternative='two-sided')
print("t-statistics: {}, p-value : {}".format(t_stat, p_val))

# t.test(Before, After, alternative=c("two.sided"), paired=TRUE,
#        conf.level=0.95)


t_stat, p_val = stats.ttest_ind(data["Before"], data["After"], equal_var=True, alternative='greater')
print("t-statistics: {}, p-value : {}".format(t_stat, p_val))


# t.test(Before, After, alternative=c("greater"), paired=TRUE,
#        conf.level=0.95)



stockDataset = pd.read_csv("EuStockMarkets.csv")
print(stockDataset)
# head(EuStockMarkets)
# data("EuStockMarkets")
# dim(EuStockMarkets)
# EuStockMarkets

print(stockDataset.describe())
# summary(EuStockMarkets)

print(stockDataset['DAX'])
# EuStockMarkets[,'DAX']


print(statistics.mean(stockDataset['DAX']))
# mean(EuStockMarkets[,'DAX'])

print(statistics.median(stockDataset['DAX']))
# median(EuStockMarkets[,'DAX'])

print(stockDataset.info())
# range(EuStockMarkets[,'DAX'])

print(stockDataset['DAX'].describe())
# summary(EuStockMarkets[,'DAX'])


print(np.var(stockDataset['DAX']))
# var(EuStockMarkets[,'DAX']) 
print(np.std(stockDataset['DAX']))
# sd(EuStockMarkets[,'DAX'])
# 분포의 비대칭성 확인
# install.packages("fBasics")
# library(fBasics)

print(skew(stockDataset['DAX']))
# skewness(EuStockMarkets[,'DAX']) # 왜도

print(kurtosis(stockDataset['DAX']))
# kurtosis(EuStockMarkets[,'DAX']) # 첨도

plt.hist(stockDataset['DAX'])
# hist(EuStockMarkets[,'DAX']) # 히스토그램

plt.boxplot(stockDataset['DAX'])
# boxplot(EuStockMarkets[,'DAX']) # boxplot

plt.scatter(stockDataset['DAX'], stockDataset['SMI'])
# plot(EuStockMarkets[,'DAX'], EuStockMarkets[,'SMI']) #산점도 출력

plt.plot(stockDataset)
# plot(EuStockMarkets) # 시계열 

# 2. 상관분석
# R의 corr 함수를 통해 상관계수를 파악하고 그래프를 통해 독립변수 간의 상관관계 분석을 수행한다.

print(np.corrcoef(stockDataset['DAX'], stockDataset['SMI'])[0,1])
# cor(EuStockMarkets[,'DAX'], EuStockMarkets[,'SMI']) # 상관계수를 계산

print(pd.DataFrame.corr(stockDataset), "\n")

# cor(EuStockMarkets) # 상관계수 

# install.packages("corrplot")
# library(corrplot)

CorrEuStockMarkets = pd.DataFrame.corr(stockDataset)
# CorrEuStockMarkets <- cor(EuStockMarkets)
plt.matshow(CorrEuStockMarkets)
plt.show()
# corrplot(CorrEuStockMarkets, method="ellipse") # 상관계수 행렬


# 주성분 분석
iris = datasets.load_iris()
X = iris.data
print(type(X), "\n", X[:5])

std = preprocessing.StandardScaler()
X_std = std.fit_transform(X)
print(X_std[:5])

# data(iris)
# iris.pca <- prcomp(iris[,1:4]) 



# PCA수행
cov_matrix = np.cov(X_std.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

np.sort(eigenvalues)[::-1]

print("val: ", "\n", eigenvalues)
print("val: ", "\n", eigenvectors)

# iris.pca # 주성분분석 결과
# summary(iris.pca) # PCA 결과
# iris.predict <- predict(iris.pca) # 주성분 점수


explained_variances = []
sums = np.sum(eigenvalues)

for v in eigenvalues:
    explained_variances.append(v/sums)
print(explained_variances)

# iris.predict[, 1:2] # 주성분 1과 주성분 2의 점수 

pc1 = X_std.dot(eigenvectors[0])
pc2 = X_std.dot(eigenvectors[1])

df_pc = pd.DataFrame(data=pc1, columns=['pc1'])
df_pc['pc2'] = pc2
df_pc['class'] = iris.target
col_mapping = {0:'Setosa', 1:'Versicolour', 2:'Virginica'}

df_pc['class'] = df_pc['class'].map(lambda x : col_mapping[x])

df_pc.sample(5)

plt.figure(figsize=(10, 5))
sns.scatterplot(x='pc1', y='pc2', data=df_pc, hue=df_pc['class'], s=100)

#
# biplot(iris.pca) # 주성분 산점도
# data(mtcars) # mtcar datset
# dat <- subset(mtcars, select=c(mpg, am, vs))
# dat

# Logistic Regression 파이썬 버전######
# 컴파일 꼬임 방지를 위해 데이터셋 재호출
iris = datasets.load_iris()
X = iris['data'] # iris.data
y = iris['target'] # iris.target
features = iris['feature_names'] # iris.feature_names
iris_df = pd.DataFrame(X,columns=['sepal_length', 'sepal_width','petal_length', 'petal_width'])
iris_df['species'] = y
print(iris_df.iloc[:5, :])
print(iris_df.describe())
sns.pairplot(iris_df, hue='species', vars=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
plt.show()

# 데이터(X)와 타겟(y)을 학습/검증
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1217)

# Logistic Regression
log_reg = LogisticRegression()

# 모델 적합/학습
log_reg.fit(X_train, y_train)
predictions = log_reg.predict(X_test)
print('y true:', y_test)
print('y pred:', predictions)

#
# log_reg <- glm(vs ~ mpg, data=dat, family=binomial)

# log_reg
# summary(log_reg) 




# AirPassengers 시계열 데이터####
#sm.graphics.tsa.plot_acf(result, lags=50, use_vlines=True)

df = pd.read_csv('AirPassengers.csv')
df.columns = ['Month', 'Number of Passengers']
df.head()
fig = px.line(df, y=df['Number of Passengers'])
fig.show()

#
# data(AirPassengers)
# AirPassengers
# plot(AirPassengers)
x = df['Month'].values
y1 = df['Number of Passengers'].values

fig, ax = plt.subplots(1, 1, figsize=(16, 5), dpi=120)
plt.fill_between(x, y1=y1, y2=-y1, alpha=0.5, linewidth=2, color='seagreen')
plt.ylim(-800, 800)
plt.title("Air Passengers (Two Side View)", fontsize=16)
plt.hlines(y=0, xmin=np.min(df['Month']), xmax=np.max(df['Month']), linewidth=.5)
plt.show()

# plot(stl(AirPassengers, s.window='periodic')) # 계절성 (seasonality), 추세 (trend), 불확실성 (random) 요소로 분해해서 그래프를 확인할 수 있다.


#예측 모델 생성

data = pd.read_csv('AirPassengers.csv')
data['Month']=pd.to_datetime(data['Month'], infer_datetime_format=True)
data=data.set_index(['Month'])
data.head()
plt.xlabel('Month')
plt.ylabel('Number of passengers')
plt.plot(data)
plt.show()

# install.packages("tseries")
# library(tseries)
# difflogAirPassengers <- diff(log(AirPassengers))
# plot(difflogAirPassengers)
# adf.test(difflogAirPassengers, alternative="stationary", k=0)
#
# install.packages("forecast")
# library(forecast)
# auto.arima(difflogAirPassengers)
# fitted <- arima(log(AirPassengers), c(1, 0, 1), seasonal =
#                     list(order = c(0, 1, 1), period = 12))
# fitted
#
## predicted <- predict(fitted, n.ahead = 120)
# ts.plot(AirPassengers, exp(predicted$pred), lty = c(1,2)) # predicted$pred 항목에 log(AirPassengers)의 예측치 값이 저장



# 6. 군집화: K 평균 군집화

iris = datasets.load_iris()

X = iris.data[:, 2:]       
y = iris.target

plt.scatter(X[:,0], X[:,1], c = y, cmap ='gist_rainbow')   # 산점도
plt.xlabel('Petal Width', fontsize = 18)                
plt.ylabel('Petal Length', fontsize = 18)                     

inertia_arr = []
k_range = range(1, 10)

for k in k_range:
    kmeans = KMeans(n_clusters= k, random_state= 21)
    kmeans.fit(X)
    inertia = kmeans.inertia_
    inertia_arr.append(inertia)

# Elobw Method 그래프 그리기
plt.plot(k_range, inertia_arr, marker= 'o')
plt.xlabel('Number of clusters', fontsize=13)
plt.ylabel('Inertia', fontsize=13)
plt.show()


#
# install.packages('cluster')
# library(cluster)
# data(iris)
# nc2=pam(iris, 2) #  군집화
# si2 = silhouette(nc2)
# summary(si2)
# plot(si2) # 실루엣결과
# nc3=pam(iris, 3); si3=silhouette(nc3);
# summary(si3)
#
#

# # K 평균 군집화 모델
# # K=2인 경우 군집화 
# iris.kc = kmeans(iris[, 1:4], 2) # 군집수 2
# iris.kc # K=2일때의 결과
kmeans = KMeans(n_clusters = 2, random_state=21)
kmeans.fit(X)

y_pred = kmeans.labels_

# 비교 그래프
fig, axes = plt.subplots(1, 2, figsize = (16,8))

# Iris 꽃잎 그래프
axes[0].scatter(X[:, 0], X[:, 1], c = y, cmap = 'gist_rainbow', edgecolor = 'k', s = 150)
axes[0].set_xlabel('Petal length', fontsize = 18)
axes[0].set_ylabel('Petal width', fontsize = 18)
axes[0].tick_params(direction = 'in', length = 10, width = 5, colors = 'k', labelsize = 20)
axes[0].set_title('Actual', fontsize = 18)

# K-Means Clustering예측
axes[1].scatter(X[:, 0], X[:, 1], c = y_pred, cmap = 'jet', edgecolor = 'k', s = 150)
axes[1].set_xlabel('Petal length', fontsize = 18)
axes[1].set_ylabel('Petal width', fontsize = 18)
axes[1].tick_params(direction = 'in', length = 10, width = 5, colors = 'k', labelsize = 20)
axes[1].set_title('Predicted', fontsize = 18)
plt.show()

# # K=3 군집화
# iris.kc = kmeans(iris[, 1:4], 3) 
# iris.kc # K=3일때의 결과

kmeans = KMeans(n_clusters = 3, random_state=21)
kmeans.fit(X)

y_pred = kmeans.labels_

# 비교 그래프 그리기
fig, axes = plt.subplots(1, 2, figsize = (16,8))

# Iris 꽃잎 그래프
axes[0].scatter(X[:, 0], X[:, 1], c = y, cmap = 'gist_rainbow', edgecolor = 'k', s = 150)
axes[0].set_xlabel('Petal length', fontsize = 18)
axes[0].set_ylabel('Petal width', fontsize = 18)
axes[0].tick_params(direction = 'in', length = 10, width = 5, colors = 'k', labelsize = 20)
axes[0].set_title('Actual', fontsize = 18)

# K-Means Clustering을 통해 예측한 결과 그래프
axes[1].scatter(X[:, 0], X[:, 1], c = y_pred, cmap = 'jet', edgecolor = 'k', s = 150)
axes[1].set_xlabel('Petal length', fontsize = 18)
axes[1].set_ylabel('Petal width', fontsize = 18)
axes[1].tick_params(direction = 'in', length = 10, width = 5, colors = 'k', labelsize = 20)
axes[1].set_title('Predicted', fontsize = 18)
plt.show()

# # K=3인 경우 군집화
# iris.kc = kmeans(iris[, 1:4], 3) # 군집수 3으로 시행
# iris.kc # K=3일때의 결과 출력

# 파생변수를 활용하여 분석모델을 확장한다.
id = ["c01", "c02", "c03", "c04", "c05", "c06", "c07"]
age = [25, 45, 31, 30, 49, 53, 27]

customers = pd.DataFrame({"id":id, "age":age})
print(customers)



# id <- c("c01", "c02", "c03", "c04", "c05", "c06", "c07")
# age <- c(25, 45, 31, 30, 49, 53, 27)
# customers <- data.frame(id, age, stringsAsFactors = F)
# customers
# sapply(customers, class)

def func1(x) :
    if x >= 20 and x < 30 :
        return 1
    else:
        return 0
def func2(x) :
    if x >= 30 and x < 40 :
        return 1
    else:
        return 0
def func3(x) :
    if x >= 40 and x < 50 :
        return 1
    else:
        return 0
def func4(x) :
    if x >= 50 and x < 60 :
        return 1
    else:
        return 0

customers["age20s"] = customers["age"].apply(lambda x : func1(x))
customers["age30s"] = customers["age"].apply(lambda x : func2(x))
customers["age40s"] = customers["age"].apply(lambda x : func3(x))
customers["age50s"] = customers["age"].apply(lambda x : func4(x))
print(customers)

# customers <- transform(customers,
#                        age20s = ifelse(age >= 20 & age < 30, 1, 0),
#                        age30s = ifelse(age >= 30 & age < 40, 1, 0),
#                        age40s = ifelse(age >= 40 & age < 50, 1, 0),
#                        age50s = ifelse(age > 50 & age < 60, 1, 0))
# customers


# 배깅 (Bagging)

iris = load_iris()
X, y = iris.data[:, [0, 2]], iris.target

model1 = DecisionTreeClassifier(max_depth=10, random_state=0).fit(X, y)
model2 = BaggingClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=100, random_state=0).fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
plt.subplot(121)
Z1 = model1.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, Z1, alpha=0.6, cmap=plt.cm.jet)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=1, s=50, cmap=plt.cm.jet, edgecolors="k")
plt.title("개별 모형")
plt.subplot(122)
Z2 = model2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, Z2, alpha=0.6, cmap=plt.cm.jet)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=1, s=50, cmap=plt.cm.jet, edgecolors="k")
plt.title("배깅 모형")
plt.suptitle("붓꽃 데이터의 분류 결과")
plt.tight_layout()
plt.show()



# data(iris)
# train_index <- c(sample(1:50, 25), sample(51:100, 25), sample(101:150, 25))
#
# # 총 150개 iris 데이터 중 임의로 2개의 데이터 세트로 나눈다.
# install.packages("robustbase")
# library(adabag) # 배깅을 위한 R 패키지 로딩
# bagging_iris <- bagging(Species ~ ., data = iris[train_index, ], mfinal = 10, control = rpart.control(maxdepth = 1)) # 10개의 트리로 구성된 배깅 모델을 구축
# bagging_iris # 결과 출력
#
# predict_bagging_iris <- predict.bagging(bagging_iris, newdata =
#                                           iris[-train_index, ]) # 앞서 생성한 배깅모델을 활용하여 새로운 데이터에 대한 예측 수행

# 부스팅(Boosting)
# 부스팅은 잘못 분류된 개체들에 관심을 가지고 이들을 더 잘 분류하기 위해서 잘못 분류된 개체들에 집중하여 새로운 분류 규칙을 만드는 단계를 반복하는 방법이다.
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
# Train Adaboost Classifer
model = abc.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = model.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# data(iris)
# train_index <- c(sample(1:50, 25), sample(51:100, 25), sample(101:150, 25))
#
# library(adabag)
#
# boosting_iris <- boosting(Species ~ ., data = iris[train_index, ], mfinal = 10, control = rpart.control(maxdepth = 1)) # 부스팅을 위한 10개의 트리가 생성되고 데이터를 기반으로 순차적 학습이 시작
#
# boosting_iris # 결과 출력



# 랜덤 포레스트(Random forest)
# 랜덤 포레스트는 여러 개의 의사결정트리를 임의로 학습하는 앙상블 방법이다
#loading the iris dataset
iris = load_iris()
#training data 설정
x_train = iris.data[:-30]
y_train = iris.target[:-30]
#test data 설정
x_test = iris.data[-30:] # test feature data
y_test = iris.target[-30:] # test feature data
rfc = RandomForestClassifier(n_estimators=10)

# rfc.fin()에 훈련 데이터를 입력해 Random Forest 모듈을 학습
rfc.fit(x_train, y_train)
#Test data를 입력해 target data를 예측 (매번 달라짐)
prediction = rfc.predict(x_test)
#예측 결과 precision과 실제 test data의 target을 비교
print(prediction==y_test)
rfc.score(x_test, y_test)

#
# data(iris)
# index <- sample(2,nrow(iris),replace=TRUE,prob=c(0.7,0.3)) # 훈련 데이터 세트(1)과 검증 데이터 세트(2) 구분을 위해 7:3의 비율로 무작위 샘플링 수행
# training_data <- iris[index==1,]
# testing_data <- iris[index==2,]
# library(randomForest) # 랜덤 포레스트를 위한 R 패키지 로딩
# rf_iris <- randomForest(Species~., data=training_data, ntree=100,
#                           proximity=TRUE) # 100개의 의사결정트리를 통해서 랜덤 포레스트를 학습
# rf_iris # 결과 확인
#
# predicted_iris <- predict(rf_iris, newdata=testing_data) # 랜덤 포레스트 모델을 이용하여 testing_data를 예측



time = list(range(1,319))
value = [random.random()+32 for r in range(1,319)]
print(time)
print(value)


data1 = pd.DataFrame({"time":time, "value":value})
print(data1)
x = data1['time']
y = data1['value']

data1.plot.scatter(x = 'time', y = 'value')

# Unnamed: 0 컬럼을 drop하여 제거
# data1.drop(['Unnamed: 0'], axis = 1, inplace = True)

# Call forecast
# install.packages("forecast")
# library(forecast)
#
# # csv 읽기
# # "example.csv" 파일에 있는 내용을 가져온다.
# ex_data <- read.csv(file="example.csv", header = TRUE, check.names = F)
# ex_data
# plot(ex_data)

# df["Numbers"] = [float(str(i).replace(",", "")) for i in df["Numbers"]]

# training_data = ex_data[1:293, 2] # 총 318 데이터 중 처음 293개의 데이터를 훈련 데이터 세트로 설정
# testing_data = ex_data[294:318, 2] # 나머지 25개의 데이터를 검증 데이터 세트로 설정

data1["time"] = data1["time"].astype("datetime64[ns]")
data1.head()

training_data = data1[1:293] # 총 318 데이터 중 처음 293개의 데이터를 훈련 데이터 세트로 설정
testing_data = data1[294:318] # 나머지 25개의 데이터를 검증 데이터 세트로 설정

training_data['value']

training_data['value'].plot()
# 각각의 데이터 세트를 시계열 형식으로 변환
# training_timeseries = ts(training_data, frequency = 1)
# testing_timeseries = ts(testing_data, frequency = 1)
# plot.ts(training_timeseries)



model_arima= auto_arima(training_data['value'],trace=True, error_action='ignore', start_p=1,start_q=1,max_p=5,max_q=5,suppress_warnings=True,stepwise=False,seasonal=False)

# AIC 값을 최소화 하는 p와 q 값을 선택
model_arima.fit(training_data['value'])


# max_p = 5 # 최대 ARIMA p 값
# max_q = 5 # 최대 ARIMA q 값
# # 각 p, q 값별로 AIC 값을 계산
#
# AIC_set = matrix(0, nrow = (max_p+1), ncol = (max_q+1))
# for (p in 0:max_p){
#      for (q in 0:max_q)
#        {
#          model = arima(training_timeseries, order = c(p,1,q),
#                         method = "ML")
#          AIC_set[(p+1), (q+1)] = model$aic
#        }
#    }


# AIC 값을 최소화 하는 p와 q 값을 선택
# which(AIC_set == min(AIC_set), arr.ind = TRUE)

start_index = data1["time"][294]
end_index = data1["time"][317]

## 404 가 아닌 pq 값  1,0,1  일때가 더 높았기에 해당 order 로 모델 호출
forecast = model_arima.predict(start=start_index, end=end_index, typ='levels')
print(forecast)

# model = arima(training_timeseries, order = c(4,1,4))
# forecasted = forecast(model, h=25) # 25개 데이터에 대한 예측

MAE_result = statistics.mean(abs([float(n) for n in testing_data['value']] - statistics.mean(forecast))) #MAE
MSE_result = statistics.mean(([float(n) for n in testing_data['value']] - statistics.mean(forecast))**2) #MAE
MAPE_result = statistics.mean((abs([float(n) for n in testing_data['value']] - statistics.mean(forecast)))/testing_data['value']*100) #MAE
print(MAE_result)
print(MSE_result)
print(MAPE_result)

# MAE_result = mean(abs(testing_data - forecasted$mean)) #MAE
# MSE_result = mean((testing_data - forecasted$mean)^2) # MSE
# MAPE_result = mean(abs(testing_data - forecasted$mean) / testing_data)*100 # MAPE ### testing 으로 되있던것 오타 수정
# MAE_result
# MSE_result
# MAPE_result




# ROC 곡선 기법을 통한 분류모델 성능 평가


x_data = [
    [2, 1],
    [3, 2],
    [3, 4],
    [5, 5],
    [7, 5],
    [2, 5],
    [8, 9],
    [9, 10],
    [6, 12],
    [9, 2],
    [6, 10],
    [2, 4]
]
y_data = [0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0]

labels = ['fail', 'pass']


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=777, stratify=y_data)


model = LogisticRegression()

model.fit(x_train, y_train)


y_predict = model.predict(x_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, model.decision_function(x_test))
#false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:, 1])

#roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
roc_auc = metrics.roc_auc_score(y_test, model.decision_function(x_test))
#roc_auc = metrics.roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])

plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive Rate(1 - Specificity)')
plt.ylabel('True Positive Rate(Sensitivity)')

plt.plot(false_positive_rate, true_positive_rate, 'b', label='Model (AUC = %0.2f)'% roc_auc)
plt.plot([0,1],[1,1],'y--')
plt.plot([0,1],[0,1],'r--')

plt.legend(loc='lower right')
plt.show()


# install.packages('ROCR')
# library(ROCR)
# data(ROCR.simple)
# predict_simple <- prediction( ROCR.simple$predictions,
#                                 ROCR.simple$labels)
# perf_simple <- performance(predict_simple,"tpr","fpr")
# plot(perf_simple)
# abline(a=0, b=1) # 북동쪽 코너에서 남서쪽 코너를 잇는 직선을 표시
#
# performance(predict_simple,"auc")




# 군집모델의 경우 내부평가(던 지수, CH 지수 등) 혹은 외부평가(랜드 지수, 자카드 지수 등) 등을 이용하여 모델 성능을 평가한다
# iris 군집 모델 생성 및 검증, 결과과 출력

# rpart 와 clvalid가 파이썬에 없기때문에
# ARI 지표로 군집모델 평가 하는 방법을 대신 작성

# matplotlib 설정

plt.rcParams['axes.unicode_minus'] = False # 축 -표시

# dataset
x, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# pre-processing
scaler = StandardScaler().fit(x)
x_scaled = scaler.transform(x)

# model list
algorithms = [KMeans(n_clusters=2),
              AgglomerativeClustering(n_clusters=2),
                      DBSCAN()]
kmean_pred = algorithms[0].fit_predict(x_scaled)

# random cluster
random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(x))

### visualization

fig, axes = plt.subplots(2, 2,
                         subplot_kw={'xticks':(), 'yticks':()})
axes[0,0].scatter(x_scaled[:, 0], x_scaled[:, 1], c=random_clusters,
                cmap=mglearn.cm3, s=60, edgecolors='k')
axes[0,0].set_title('random assign - ARI: {:.3f}'.format(
    adjusted_rand_score(y, random_clusters))) # 실제, 모델로 구한 클러스터
axes[0,1].scatter(x_scaled[:, 0], x_scaled[:, 1], c=kmean_pred,
                  cmap=mglearn.cm3, s=60, edgecolors='k')
axes[0,1].set_title('{} - ARI: {:.3f}'.format(algorithms[0].__class__.__name__, #__class__.__name__ ==> 클래스에서 이름 속성
                                              adjusted_rand_score(y, kmean_pred)))

for ax, algorithm in zip(axes.ravel()[2:], algorithms[1:]):
    clusters = algorithm.fit_predict(x_scaled)
    ax.scatter(x_scaled[:, 0], x_scaled[:, 1], c=clusters,
               cmap=mglearn.cm3, s=60, edgecolors='k')
    ax.set_title('{} - ARI: {:.3f}'.format(algorithm.__class__.__name__,
                                           adjusted_rand_score(y, clusters)))
plt.show()



# library(caret)
# set.seed(102)
# index4training <- createDataPartition(y=iris$Species, p=0.7, list=FALSE)
# training_data <- iris[index4training,]
# testing_data <- iris[-index4training,]
# training.data <- scale(training_data[-5])
# summary(training.data)
#
# # 군집모델 생성
# kmeans_iris <- kmeans(training.data[,-5], centers=3, iter.max=10000)
#
# # 검증 데이터 세트를 통한 모델 정확성 확인
# library("rpart")
#
# training_data$cluster <- as.factor(kmeans_iris$cluster)
# data.matrix(training_data)
# fitted <- train(x=training.data[,-5], y=training_data$cluster, method="rpart")
# # fitted <- train(data.matrix(0), y=data.matrix(1), method="rpart")
# testing.data <- as.data.frame(scale(testing_data[-5]))
#
# testClusters <- predict(fitted, testing.data)
# testClusters <- as.numeric(as.character(testClusters))
#
# training_data

# 내부평가 방법을 이용한 모델성능 평가하기
# # R 에서 사용  하는 평가방법
# install.packages('clValid')
# library("clValid")
# dunn(,testClusters, testing_data)
