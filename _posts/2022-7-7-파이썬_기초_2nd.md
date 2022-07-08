```python
# 파이썬 프로그래밍 언어
# 입력 : input
# 저장 : 변수
# 처리 : 산술연산, 분기, 판별문
# 출력 : print
```


```python
# 프로그래밍 언어!
# 말하기 : 문제정의를 잘하자, 하고싶은 얘기를 잘 정리하자 
# 쓰기 : 문법(파이썬 문법)
# 읽기 : 다른 사람의 코드, 문서
# 듣기 : 오류 메시지
```


```python
# 회원 가입 필수 항목 중 이름을 입력받는다. 단, 입력되지 않은 경우 예외처리한다.
name = input("이름을 입력하세요 : ")
```


```python

```


```python
# 반복문
for ~하는 동안 ~하겠다.
while 조건이 참인 동안 ~하겠다.
```


```python
# 바구니 : 자료구조
# 리스트, 딕셔너리, 튜플, 셋
# range : 범위생성자
# range(y) : 0~y미만
# range(x,y) : x~y미만
# range(x,y,z) : x~y미만까지 z씩 증감

```


```python
for 이름 in 바구니:
    실행구문
```


```python
for x in range(10):
    print("hello")
```


```python
for _ in range(10):
    print("hello")
```


```python
for k in range(10,1, -1):
    print(k)
```


```python
# 1~100까지의 정수 중 홀수만 출력하는 for문을 작성해보세요.
for x in range(1,101,2):
    print(x)
```


```python
for x in range(1,101):
    if x % 2 == 1:
        print(x)
```


```python
int x;
x = 0
```


```python
#
x = 0 # x라는 변수를 만들고 거기에 0이라는 값을 저장한다.
x = 1 # x라는 변수의 값을 1로 변경한다.
```


```python
# 1~100까지의 정수 중 짝수들의 합을 출력하시오.
# 단계별로 확인이 필요
total = 0
for num in range(2,101,2):
    total = total + num
print(total)
```

    100
    


```python
if condition:
    실행구문
    실행구문
    실행구문
if condition:
    print("test")
    for x in range(10):
        print('ab')
```


```python
# 들여쓰기
# 탭 VS Space
# 2칸 : 
# 4칸 : 현재 보통의 경우
# 8칸 : 윈도우 95, 98 때
```


```python
if (condition) {
    if (condition) {
    if (condition) {
    if (condition) {
    if (condition) {
        실행구문
}}}}}

```


```python
# 1~100까지의 정수 중 3의 배수들의 합을 출력하시오.
total = 0
for num in range(3,101,3):
    total = total + num
print(total)
```

    1683
    


```python
# 1~100까지 짝수만 출력하도록 for문을 작성해보시오.
for num in range(2,101,2):
    print(num)
```


```python
# 1~100까지 짝수만 출력하도록 for문을 작성해보시오.
for num in range(1,101):
    print(num+1)
```


```python
# 1~100까지 짝수만 출력하도록 for문을 작성해보시오.
for num in range(1,51):
    print(num*1)
```


```python
while 조건식:
    실행구문
```


```python
a = 3
while a < 7:
    print(a)
    a = a + 1
```


```python
# while 문을 사용해서
# 1~100까지의 정수들을 순서대로 출력하시오.
number = 1
while number <= 100:
    print(number)
    number = number + 1
```


```python
number = number + 1
# 복합할당 연산자
number += 1
number -= 1
number *= 2
number /= 2
```


```python
# while문을 이용해서
# 입력받은 정수부터 + 100까지의 정수를 순서대로 출력하시오.
# 1. 정수를 입력 받는다.
number = 10
# 2. 형변환 한다.
# 3. while문을 작성한다. 조건은 입력받은 정수 + 100보다 작거나 같을 때까지
end = number + 100
while number <= end:
    print(number)
    number += 1
```


```python
# 순차
# 분기
# 반복
```


```python
number = input("정수를 입력하세요 : ")
# 2. 형변환 한다.
number = int(number)
# 3. while문을 작성한다. 조건은 입력받은 정수 + 100보다 작거나 같을 때까지
end = number + 100
while number <= end:
    print(number)
    number += 1
```


```python
number = "10"
# 2. 형변환 한다.
number = 10
# 3. while문을 작성한다. 조건은 입력받은 정수 + 100보다 작거나 같을 때까지
end = 10 + 100
while number <= end:
    print(number)
    number += 1
```


```python

```


```python
# 반복문 제어 연산자
# continue
# break
number = 1
end = 10
while number <= end:
    number += 1
    if number < 5:
        continue
    print(number)
```


```python
number = 1
end = 10
while number <= end:
    number += 1
    if number < 5:
        pass
    else:
        print(number)
```


```python
# 주석 설명
number = 1
end = 10
while number <= end:
    number += 1
    if number >= 5:
        print(number)
```


```python
while True:
    print("hi")
    break
```

    hi
    


```python
# 노트북
# 가성비
# 이쁘다
# 특수 기능
# 확장 가능
# 어워드 2020 1st

```


```python
# 어제 작성한 원달러 변환 프로그램을
# 무제한 실행되도록 만들어봅시다. 

```


```python
while True:
    print("1. 원 -> 달러")
    print("2. 달러 -> 원")
    print("3. 프로그램 종료")
    menu = input("메뉴를 선택하세요 : ")
    if menu == "1":
        won = input("원을 입력하세요 : ")
        try:
            won = float(won)
        except:
            won = 0
            print("숫자로 변환이 불가능 합니다.")
        usd = won / 1306.5
        print(usd)
    elif menu == "2":
        # 달러 -> 원
        usd = input("달러를 입력하세요 : ")
        try:
            usd = float(usd)
        except:
            usd = 0
            print("숫자로 변환이 불가능 합니다.")
        won = usd * 1306.5
        print(won)
    elif menu == "3":
        break
        # exit() 프로그램 종료 명령어
    else:
        print("없는 메뉴입니다.")
```


```python
#
while True:
    try:
        weight = input("체중을 입력하세요 : ")
        weight = float(weight)
        break
    except:
        print("변환할 수 없는 값입니다.")
        weight = 0
```


```python
while True:
    try:
        weight = input("체중을 입력하세요 : ")
        weight = float(weight)
    except:
        print("변환할 수 없는 값입니다.")
        weight = 0    
    else:
        break
```

    체중을 입력하세요 : asdf
    변환할 수 없는 값입니다.
    체중을 입력하세요 : sadf
    변환할 수 없는 값입니다.
    체중을 입력하세요 : 234
    


```python

```


```python
for x in "hello":

    print(x)
```

    h
    e
    l
    l
    o
    


```python
# 데이터 타입
# 원시형 : 문자, 숫자(정수,실수), 불(Boolean)

# 바구니
# 시퀀스-(문자열, 리스트), 딕트, 튜플, 셋

# 사용자 정의 타입
# 클래스
```


```python
# 모듈과 라이브러리
# 프레임워크
```


```python
import random
# 외부 모듈을 불러온다. 사용하려고
# 1. 파이썬 내장 모듈 (built-in)
# 2. 추가 설치한 모듈(소스코드 혹은 pip 명령어 등을 이용해서)
# 3. 내가 작성한 모듈(혹은 우리 부서, 회사에서)
```


```python
import random

import pandas as pd
import numpas as np
from bs4 import BeautifulSoup

from .model import *
```


```python
# random이라는 모듈을 통째로 rnd라는 별명을 지정해서 불러오세요.
import random as rnd
```


```python

```


```python
# 6개 중복없이 뽑기
num1 = rnd.randint(1,45)

num2 = rnd.randint(1,45)
while num1 == num2:
    num2 = rnd.randint(1,45)
num3 = rnd.randint(1,45)
while num1 == num3 or num2 == num3:
    num3 = rnd.randint(1,45)
    
num4 = rnd.randint(1,45)
while num1 == num4 or num2 == num4 or num3 == num4:
    num4 = rnd.randint(1,45)
    
num5 = rnd.randint(1,45)
while num1 == num5 or num2 == num5 or num3 == num5 or num4 == num5:
    num5 = rnd.randint(1,45)

num6 = rnd.randint(1,45)
while num1 == num6 or num2 == num6 or num3 == num6 or num4 == num6 or num5 == num6:
    num6 = rnd.randint(1,45)
    
print(num1, num2, num3, num4, num5, num6)
```

    31 5 27 16 44 32
    


```python
msg = "1234567890-=+_`~qwertyuiop[]\asdfghjkl;'zxcvbnm,./'"
for x in msg:
    print(x)
```


```python
# 자료구조 쉽게 익히기
# 만들기
# 값 읽기
# 값 변경
# 값 추가하기
# 값 삭제하기
```


```python
# 1. 만들기 (list 리스트)
# 비어있는 바구니 만들기
test = []
test = list()
# 요소가 있는 바구니 만들기
test = [1,2,3,4,5]
test = list(range(10))
# list comprehension
```


```python
# fruit이라는 이름을 가진 빈 리스트를 만드시오
fruit = []
```


```python
type(fruit)
```




    list




```python
# fruit이라는 이름을 가진 리스트에 각자 좋아하는 과일 3개의 이름을 넣어서 만드시오.
fruit = ["메론","수박","참외"]
```


```python
for item in fruit:
    print(item)
```

    메론
    수박
    참외
    


```python
# 시퀀스 객체는 순번(index)를 갖는다.
fruit[1] # <----- 이것만 단독으로 변수 취급을 할 수 있다.
```




    '수박'




```python
fruit[1] = "딸기"
```


```python
fruit
```




    ['메론', '딸기', '참외']




```python
fruit[-1]
```




    '참외'




```python
fruit.insert(100,"바나나")
```


```python
fruit.insert(-1,"그라비올라")
```


```python
fruit
```




    ['메론', '딸기', '참외', '그라비올라', '바나나']




```python
fruit.insert(0,"귤")
```


```python
fruit
```




    ['귤', '메론', '딸기', '참외', '그라비올라', '바나나']




```python
len(fruit)
```




    6




```python
fruit.insert(len(fruit),"망고")
```


```python
fruit
```




    ['귤', '메론', '딸기', '참외', '그라비올라', '바나나', '망고']




```python
fruit.append("구아바")
```


```python
fruit
```




    ['귤', '메론', '딸기', '참외', '그라비올라', '바나나', '망고', '구아바']




```python
fruit.extend(["1","2","3"])
```


```python
fruit
```




    ['귤', '메론', '딸기', '참외', '그라비올라', '바나나', '망고', '구아바', '1', '2', '3']




```python
fruit + ["test","test2"]
# 값이 튀어나오면 원본은 변경되지 않았다.
```


```python
# 삭제
fruit.remove("딸기")
```


```python
fruit.pop(2)
```




    '참외'




```python
fruit
```




    ['귤', '메론', '그라비올라', '바나나', '망고', '구아바', '1', '2']




```python
int()
str()
float()
```


```python
# 로또 번호 뽑기
# 1. 1~45까지 원본 공이 있는 통(바구니) 준비하기
# 2. 바구니에서 하나씩 랜덤으로 뽑아내기
# 3. 총 6개를 뽑기
balls = list(range(1,46))
```


```python
for _ in range(6):
    index = rnd.randint(0,len(balls)-1) # 0~44 사이의 랜덤한 정수
    ball = balls.pop(index)
    print(ball)
```

    16
    29
    35
    6
    17
    12
    


```python

```
