ele <-read.table(file="C:/Users/tjtmd/Documents/GitHub/FinDucks.github.io/sogang/4학기/시계열 자료 분석과 예측/election1.txt", header=TRUE, sep="\t")
ele
names(ele)
full=lm(choice~Age65+Urban+ColGrad+Union+Neast+West+Seast, data=ele)
summary(full)
reduced=lm(choice~Neast+West+Seast, data=ele)
summary(full)
summary(reduced)
anova(reduced, full)
anova(full)
anova(reduced)


out=lm(Price~SqFt+LotSize+Baths)
null=lm(Price~1, data=ele)
step(full, data=ele, direction="backward")
nowele=data.frame(Age65=17, Urban=79, ColGrad=68, Union= 21, area ="MidWest")
reduce1=lm
predict(full, new=nowele)


library(readxl)
data_election=read_excel("election1.xls")
attach(data_election)
out_election=lm(choice~Age65+Urban+ColGrad+Union+Neast+West+Seast)
out_election=lm(choice~Age65+Urban+ColGrad+Union+factor(Area), data=ele)
summary(out_election)

step(full, data=ele, direction="backward") #AIC (Backward)

str(full)
full= lm(formula = choice ~ Age65 + Urban + ColGrad + Union + Neast + 
     Seast, data = ele)
vif(full)
newone_election=data.frame(Age65=17, Urban=79, ColGrad=68, Union=21, Area="MidWest")
predict(out_election, new=newone_election)
