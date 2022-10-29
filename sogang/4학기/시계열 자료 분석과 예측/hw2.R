home <-read.table(file="C:/Users/tjtmd/Documents/GitHub/FinDucks.github.io/sogang/4학기/시계열 자료 분석과 예측/home.txt", header=TRUE, sep="\t")

names(data)
full=lm(Price~SqFt+LotSize+Baths, data=home)
reduced=lm(Price~SqFt+LotSize, data=data)
summary(full)

anova(reduced, full)
anova(full)
anova(reduced)

full
lm(Price~SqFt+LotSize+Baths, data=home)
summary(full)
library(car)
vif(full)
nowhome=data.frame(SqFt=2955, LotSize=20.2, Baths=3)
predict(full, new=nowhome)
