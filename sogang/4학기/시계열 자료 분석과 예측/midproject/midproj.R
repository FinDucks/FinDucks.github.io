data <-read.csv(file="C:/Users/tjtmd/FinDucks.github.io/sogang/4학기/시계열 자료 분석과 예측/midproject/growth.csv", header=TRUE, sep=",")
data
full=lm(rate_match~내부온도관측치+ 내부습도관측치+ CO2관측치+  EC관측치+ 총추정광량, data=data)
summary(full)
anova(full)
library(car)

vif(full)
max(vif(full))

library(lmtest)
# sharpiro-Wilks test
shapiro.test(residuals(full))
# Breusch-Pagan test (heteroscedasticity test)
bptest(full)
# Durbin-Watson test (correlation test)
dwtest(full)
plot(fitted(full), residuals(full),xlab="fitted",ylab="residuals")

################

reduced=lm(rate_match~CO2관측치+ 총추정광량, data=data)
summary(reduced)
anova(reduced)
library(car)

vif(reduced)
max(vif(reduced))

library(lmtest)
# sharpiro-Wilks test
shapiro.test(residuals(reduced))

# Breusch-Pagan test (heteroscedasticity test)
bptest(reduced)
# Durbin-Watson test (correlation test)
dwtest(reduced)
plot(fitted(reduced), residuals(reduced),xlab="fitted",ylab="residuals")

###################
# Leverage point 를 이용하여 이상치가 있는지 확인하고, 있다면 몇 번째 값인지 확인
ginf=influence(reduced) 
ginf$hat 
plot(ginf$hat)
gs=summary(reduced)
gs$sig
mean(gs$sig)*3
ginf$hat[ginf$hat>mean(gs$sig)*3]
# 12) studentized deleted residuals 을 이용하여 이상치가 있는지 확인하고, 있다면 몇 번째 값인지 확인
r1=rstudent(reduced) # studentized deleted residuals
plot(r1)
r1[r1>2]
r1[r1<(-2)]
# 2 37, 50 번째 이상치
# 13) 영향치(influemtial point)가 있는지 확인하고, 있다면 몇 번째 값인지 확인
cg=cooks.distance(reduced) # cooks distance
plot(cg)
cg[cg>1] # 없음

# 14) 잔차의 Q-Q plot 을 그리시오.
qqnorm(residuals(reduced),ylab="residuals") # normality?
qqline(residuals(reduced))
