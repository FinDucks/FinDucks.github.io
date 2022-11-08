data <-read.csv(file="C:/Users/tjtmd/FinDucks.github.io/sogang/4학기/시계열 자료 분석과 예측/portfolio.csv", header=TRUE, sep=",")
                      
# 1)
cor(data)
# 2)유의수준(a) 1%에서 p-value 를 이용하여 검정하시오. 
# 즉, 모집단의 상관계수를 ρ 로 표기할 때, H0: ρ =0 vs. H1: ρ≠0 을 검정
# 상관계수가 0이다 귀무가설 상관계수가 0이 아니다 대립가설
# p-value = 0.02818
head(data)
cor.test(data$Last.Year, data$This.Year, sig.level=0.01)
cor.test()

a = ' 
data:  data$Last.Year and data$This.Year
t = 2.429, df = 15, p-value = 0.02818
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
 0.06805998 0.80610662
sample estimates:
      cor 
0.5313237 
'
#유의수준이 다름
data_bodyfat <-read.csv(file="C:/Users/tjtmd/FinDucks.github.io/sogang/4학기/시계열 자료 분석과 예측/BodyFat_hw.csv", header=TRUE, sep=",")
# 1)산포도 그리기 
data_bodyfat
plot(data_bodyfat$Girth,data_bodyfat$Fat,main = 'X: Girth, Y: Fat의 산포도')
# 2) 상관계수(r)와 결정계수 R^2 구하기
cor(data_bodyfat)
a= "
      Girth       Fat
Girth 1.0000000 0.8188484
Fat   0.8188484 1.0000000
"

model <- lm(data_bodyfat$Fat ~ data_bodyfat$Girth, data=data_bodyfat)
model
summary(model)
# Multiple R-squared:  0.6705, Adjusted R-squared: 0.6636

# 3) 상관계수와 결정계수의 관계를 설명하시오.
# 상관계수의 제곱이 결정계수로 음과 양이 없어지므로, 상관 관계 분석이 아닌 정량화를 통한 회귀 분석에서 사용하는 수치임.
0.8188484**2
# 4) 절편과 기울기의 OLS 추정값을 적으시오.
model
# (Intercept)  data_bodyfat$Girth  
# -36.2397              0.5905  

# Ordinary Least Square(최소제곱)
model$coefficients
plot(data_bodyfat$Girth, data_bodyfat$Fat, asp=10)
abline(model,col="red")
abline(model$coefficients)
# 5) 추정한 기울기 값의 의미를 적으시오. (예: A 가 한 단위 증가함에 따른 B 가 …만큼 …)
# Girth가 1 증가할 때마다 Fat이 0.5905증가
# 6) 기울기의 유의성을 검정하려고 한다. 이를 위한 귀무가설과 대립가설을 적으시오.
귀무가설 
# 7) 기울기의 유의성을 유의수준(a) 5%에서 p-value 를 이용하여 t 검정을 하고 결과를 보고하시오.
confint(model)
# 8) F 검정에 대한 귀무가설과 대립가설을 적으시오.
# 9) F 검정을 유의수준(a) 5%에서 p-value 를 이용하여 검정하고 결과를 보고하시오.
var.test(data_bodyfat$Girth, data_bodyfat$Fat)
anova(model)
vcov(model)

# 10) 잔차(residual) 그림을 그리시오.
library(lmtest)
bptest(data_bodyfat$Fat~data_bodyfat$Girth, data=data_bodyfat) # Breusch-Pagan test (heteroscedasticity test)
bptest(model)
dwtest(data_bodyfat$Fat~data_bodyfat$Girth, data=data_bodyfat) # Durbin-Watson test (correlation test)
dwtest(model)
plot(fitted(model), residuals(model),xlab="fitted",ylab="residuals") # constant variance?



# 11) Leverage point 를 이용하여 이상치가 있는지 확인하고, 있다면 몇 번째 값인지 보고하시오.
ginf=influence(model) 
ginf$hat 
plot(ginf$hat)
gs=summary(model)
gs$sig
ginf$hat[ginf$hat>0.141]
# 12) studentized deleted residuals 을 이용하여 이상치가 있는지 확인하고, 있다면 몇 번째 값인지 보고하시오.
r1=rstudent(model) # studentized deleted residuals
plot(r1)
r1[r1>2]
r1[r1<(-2)]
# 2 37, 50 번째 이상치
# 13) 영향치(influemtial point)가 있는지 확인하고, 있다면 몇 번째 값인지 보고하시오.
cg=cooks.distance(model) # cooks distance
plot(cg)
cg[cg>1] # 없음

# 14) 잔차의 Q-Q plot 을 그리시오.
qqnorm(residuals(model),ylab="residuals") # normality?
qqline(residuals(model))
plot(model,2)
# 15) 잔차의 정규분포(normal distribution) 여부를 판단하기 위한 가설은 H0: F=F0 vs. H1: F≠F0 이다.
# 여기서, F0 는 정규분포를 의미함. 이러한 가설을 k-s test 를 이용하여 유의수준 10%에서 검정하시오.
summary (model)
p1<-rstandard(model)
ks.test(p1, pnorm)

x=residuals(model)
ks.test(x, "pnorm", m=mean(x), sd=sd(x))

# 16) 잔차의 이분산성 여부를 판단하기 위한 가설은 H0: 오차는 동분산 vs. H1: 오차는 이분산 이다.
# 이러한 가설을 B-P test 를 이용하여 유의수준 10%에서 검정하시오.
# Breusch-Pagan test (heteroscedasticity test)
library(lmtest)
bptest(Fat~Girth, data=data_body)
bptest(model)

# 17) 잔차의 상관성(correlation) 여부를 판단하기 위해서 Durbin-Watson test 를 이용하여 유의수준 10%에서 검정하시오.
# Durbin-Watson test (correlation test)
dwtest(Fat~Girth, data=data_body)
dwtest(model)
plot(fitted(model), residuals(model), xlab="fitted", ylab="residuals") #constant variance;
qqnorm(residuals(model), ylab="residuals") #normality
qqline(residuals(model))
