# mpg 데이터 불러와 class, cty 변수만 남긴 뒤  class 변수가 "compact"인 자동차와 "suv"인 자동차를 추출한다.
mpg <- as.data.frame(ggplot2::mpg)

# dplyr패키지는 데이터 전처리 작업에 가장 많이 사용되는 패키지
# filter(), select(), arrange() 등등의 함수가 있다.
# filter() 행 추출
# select() 열(변수) 추출
# %>% 기호는 함수를 나열하는 방식(파이프 연산자), 단축키 Ctrl + Shift + M
library(dplyr)
mpg_diff <- mpg %>%
  select(class, cty) %>%
  filter(class %in% c("compact", "suv"))

# 앞에서부터 6행까지 출력
head(mpg_diff)

# 빈도표로 compact, suv 수 알아보기
table(mpg_diff$class)


# t.test()는 R에 내장된 함수이다.
# 추출한 mpg_diff 데이터 지정, ~기호를 이용하여 비교할 값인 cty(도시 연비)변수와 비교할 집단인 
# class(자동차 종류) 변수를 지정한다.
# t 검정은 비교하는 집단의 분산(값이 퍼져 있는 정도)이 같은지 여부에 따라 적용하는 공식이 다르다.
# 여기서는 집단 간 분산이 같다고 가정하고 var.equal 에 T를 지정하였다.
t.test(data = mpg_diff, cty ~ class, var.equal = T)



# 일반 휘발유(Regular)를 사용하는 자동차와 고급 휘발유(Premium)를 사용하는
# 자동차 간 도시 연비 차이가 통계적으로 유의한지 알아보자.


mpg_diff2 <- mpg %>% 
  select(fl, cty) %>% 
  filter(fl %in% c("r","p"))

head(mpg_diff2)

table(mpg_diff2$fl)

t.test(data = mpg_diff2, cty ~ fl, var.equal = T)

