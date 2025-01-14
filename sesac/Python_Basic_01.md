# 파이썬 기초 -1

# 주피터 노트북 단축키


```python
# 실행 단축키 : Ctrl + Enter, Alt+Enter, Shift + Enter
print("python Basic")
```

    python Basic
    

- 셀 추가 삭제 단축키 : 

- a,b ==> 위아래 셀 추가, 

- 삭제 : dd, x

- 복사 : c

- 붙여넣기 : v

- 라인 번호 확인하기 : l

- 출력 접어두기 : o



문법정리
입력, 출력, 저장, 처리
출력1
화면에 출력하기
파일에 출력하기(저장하기)


# 출력


```python
print() # 프린트 명령어
```

    
    


```python
print(1234) # 프린트 명령어
```

    1234
    


```python
print("Hello") # 프린트 명령어
```

    Hello
    


```python
print(1234, "kkkk", sep="")
```

    1234kkkk
    


```python
print("Hi", 1234, end="\na123b")
```

    Hi 1234
    a123b


```python
#어떤 값을 직접 적어서 사용하지 않는다. <--- 하드 코딩
import math
print(3.14)
print(math.pi)
```

    3.14
    3.141592653589793
    


```python
# 저장 : 변수에 저장한다.
# 갑을 저장해두는 공간, 그릇, 서랍
```


```python
print(pi)
```

    3.141592653589793
    

# 변수에 값 넣기


```python
# 저장할 수 있는 값의 종류!
# 타입 - Type -Data Type
```


```python
num=1
num2 =3.14
```


```python
str1 = "hi"
str2 = 'Hi'
str3="""hi"""
```

변수명(변수이름, 그릇이름) = 저장할 값

- 문법적 규칙(안지키면 오류)

1. 대소문자 구분 : A , a
2. 숫자로 시작불가 : 7a(x)
3. 특수문자, 띄어쓰기 불가 : _(언더바만 가능)
4. 예약어 사용 불가: 문법 용어는 변수명으로 사용금지


- 관례적 규칙(사람끼리 약속한것)
1. 명사, 동사 사용(형용사 사용 지양)
2. 지정한 값의 의미를 명확히 나타내기
3. 표기방법(관례적)


ex) my name is

카멜 : myNameIs(함수명)

파스칼 : MyNameIs(클래스명)

스네이크 : my_name_is(변수명)

헝가리안 : stMynameis

           intMynameis
           



```python
저장(변수)
출력(print)
입력(input)
```


```python
input() # 어떤 값을 입력 받아서 다음에 재활용하고 싶어서
```

    kkkkk
    




    'kkkkk'




```python
# * 아래쪽에 출력 결과가 튀어 나온다면 값을 복사해둔 것이다 
# *  아래쪽에 뭔가 값이 튀어나온다면 실행 후 반환값이 있다는 것이다.
```


```python
print()
```

    
    


```python
name = input()
```

    jake
    


```python
name
```




    'jake'



위와 같이 작성하면 뭘 입력하라는지 알 수 없음 
==>


```python
name = input("이름을 입력하세요 : ")
```

    이름을 입력하세요 : 제이크
    


```python
name
```




    '제이크'




```python
# 예제1 
# 사용자로부터 취미를 입력받아 화면에 출력하는 프로그램을 작성하시오
```


```python
h=input("취미 : ")
print("너의 취미는", h)
```

    취미 : 요트
    너의 취미는 요트
    


```python
# 위 예제 2줄이 바로 머리에 떠오르지 않는다면 추천할 만한 훈련법
# 분할정복
# 긴 문제를 만났을 때 내가 해야할 일들을 우선 한글로 정리한다.
# 1. input을 사용해서 취미를 입력받는다.
# 2. 입력받은 내용을 저장한 변수를 print를 사용해서 화면에 출력한다.


```


```python
# 연산
# 산술연산
# - 사칙연산
# -특수연산
# 문자열 연산
```


```python
a=7
b=4
c=5
```


```python
3 + 4
a-2
7 *b
c / b
```




    1.25




```python
c /b
```




    1.25




```python
7 **2  # 거듭제곱
7 // 4 # 몫
7 % 3 # 나머지
```




    1




```python
# 연산예제
```


```python
# 입력 +연산
# 계산기
# 두개의 정수를 입력받아
# 두 숫자의 사칙 연산을 출력하는 프로그램을 작성하시오.
a, b= input("사칙연산을 수행할 숫자 두개 입력하세요 :").split()
print('더하기 연산 :', a+b)
print('빼기기 연산 :', a-b)
print('곻하기 연산 :', a*b)
print('나누기 연산 :', a/b)
```

    사칙연산을 수행할 숫자 두개 입력하세요 :10 2
    더하기 연산 : 102
    


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp/ipykernel_18308/1503808021.py in <module>
          5 a, b= input("사칙연산을 수행할 숫자 두개 입력하세요 :").split()
          6 print('더하기 연산 :', a+b)
    ----> 7 print('빼기기 연산 :', a-b)
          8 print('곻하기 연산 :', a*b)
          9 print('나누기 연산 :', a/b)
    

    TypeError: unsupported operand type(s) for -: 'str' and 'str'



```python
# 파이썬의 input은 무조건 문자로만 입력받아진다.
# 내가 숫자로 바꾸자고하면 type casting 형변환을 해서 사용해야한다.
# int() 함수를 통해 변수를 str(글자)형식에서 int(정수) 타입으로 변경해주어야 계산이 진행됨
a=int(a)
b=int(b)
```


```python
print('더하기 연산 :', a+b)
print('빼기기 연산 :', a-b)
print('곻하기 연산 :', a*b)
print('나누기 연산 :', a/b)
```

    더하기 연산 : 12
    빼기기 연산 : 8
    곻하기 연산 : 20
    나누기 연산 : 5.0
    

변수

- 정수 : int

- 실수 : float

- 문자 : str

- 불대수 : bool


```python
a= '3.14'
float(a)
```




    3.14



# 양식 문자열

- % 문법
- format 명령어
- f-string


```python
# %문법
"내 이름은 %s이고, 나이는 %d입니다. 몸무게는 %fkg입니다." %("hello", 20, 74.92)
```




    '내 이름은 hello이고, 나이는 20입니다. 몸무게는 74.920000kg입니다.'



- s = strint ==> 글자
- d = 정수
- f = 실수


```python
# format 명령어
"내 이름은  {}입니다. {}, {}".format("hello", 20, 74.92)
```




    '내 이름은  hello입니다. 20, 74.92'




```python
# {}안에 순서를 넣음으로써 위치 지정 가능
"내 이름은  {0}입니다. {2}, {1}".format("hello", 20, 74.92)
```




    '내 이름은  hello입니다. 74.92, 20'




```python
# 표시 소수점 자리수 설정 가능
"내 이름은  {0}입니다. {2:8.3f}, {1}".format("hello", 20, 74.92)
```




    '내 이름은  hello입니다.   74.920, 20'




```python
# f-strint
name = "짱구"
f"내 이름은 {name}입니다."
```




    '내 이름은 짱구입니다.'




```python
n1= 10
n2 = 30
sum = n1 + n2
f'{n1} + {n2} = {sum}입니다'
```




    '10 + 30 = 40입니다'




```python
# 예제 BMI 계산기 만들기
# BMI를 이용한 비만도 계산은 자신의 몸무게를 키의 제곱으로 나누는 것 
# kg, m 단위를 사용
# 사용자의 체중과 신장을 입력받고 BMI를 계싼하여 출력하시오
# 단 출력 문구는 
# "체중 xx.xx, 신장 xxx,xx cm 일 때 BMI는 xxx.xx. 입니다."
# 라고 출력하시오.
hight= float(input("자신의 키를 입력해주세요(cm) :"))
weight = float(input("자신의 몸무게를 입력해주세요(kg) :"))
BMI = weight / ((hight/100) **2)
print(BMI)
print(f"체중 {weight: 2.2f}, 신장 {hight: 3.2f}cm 일 때 BMI는 {BMI : 3.2f} 입니다.")
```

    자신의 키를 입력해주세요(cm) :174
    자신의 몸무게를 입력해주세요(kg) :88
    29.065926806711587
    체중  88.00, 신장  174.00cm 일 때 BMI는  29.07 입니다.
    

연산자 우선 순위

- **

- *, /, //, %

- +, -

# if문

### 비교 구문 if -elif - else

if (조건문 혹은 조건식) :


    실행문
    
    
elif (조건문 혹은 조건식) :


    실행문
    
else :

    실행문

# 비교연산자


```python
# 조건문 혹은 조건식
# 명제 라고 할 수 있다.
# True, False 라고 결과 값이 나오는 식
# 비교 구문

a > b
a < b
a >= b
a <= b
a == b
a != b
```


```python
# BMI가 18.5 이하면 저체중 ／ 18.5 ~ 22.9 사이면 정상 ／ 23.0 ~ 24.9 사이면 과체중 ／ 25.0 이상부터는 비만으로 판정.
if BMI <= 18.5:
    print("저체중입니다.")
elif BMI <=22.9:
    print("정상입니다.")
elif BMI <= 24.9:
    print("과체중입니다.")
else:
    print("비만입니다.")
```

    비만입니다.
    


```python
# BMI가 18.5 이하면 저체중 ／ 18.5 ~ 22.9 사이면 정상 ／ 23.0 ~ 24.9 사이면 과체중 ／ 25.0 이상부터는 비만으로 판정.
if BMI <= 18.5:
    print("저체중입니다.")
elif 18.5 < BMI <23 :
    print("정상입니다.")
elif 23 <= BMI <= 24.9:
    print("과체중입니다.")
else:
    print("비만입니다.")
```

    비만입니다.
    

# and, or 연산자


```python
# and or
# && ||
# and : 그리고
# or : 또는
# 
```


```python
# BMI가 18.5 이하면 저체중 ／ 18.5 ~ 22.9 사이면 정상 ／ 23.0 ~ 24.9 사이면 과체중 ／ 25.0 이상부터는 비만으로 판정.
if BMI <= 18.5:
    print("저체중입니다.")
elif (18.5 < BMI) & (BMI <23) :
    print("정상입니다.")
elif (23 <= BMI) & (BMI < 25) :
    print("과체중입니다.")
else:
    print("비만입니다.")
```

    비만입니다.
    

# err 처리


```python
# 오류의 종류
# 문법 : 오타
# 실행 : 이미지 파일이 해당 위치에 없다
# 논리 논리적으로 문제가 있는 경우 
```


```python
# 오류 처리 구문
try :
    오류가 발생할 수도 있는 실행문
except:
    오류가 발생했을 때 후속 처리할 실행문
```


```python
int('ab')
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    ~\AppData\Local\Temp/ipykernel_18308/2387052414.py in <module>
    ----> 1 int('ab')
    

    ValueError: invalid literal for int() with base 10: 'ab'



```python
try :
    int('ab')
except:
    print('오류 발생')
```

    오류 발생
    


```python
try :
    int('ab')
except ValueError:
    print("값 변환 오류 발생")
except:
    print('오류 발생')
```

    값 변환 오류 발생
    


```python
# 연습문제
# 현재 환율 : 1 USD =1,306.5 won
# 1. 메뉴를 구성한다
# 1번을 선택하면 원을 입력받아서 몇달러인지 계산
# 2번을 선택하면 달러를 입력받아서 몇 원인지 계산해주는 프로그램을 작성
menu_num = input("원하시는 계산 방법 숫자를 입력해주세요. \n 1 : 원화 -> 달러 계산 \n 2 : 달러 -> 원화 계산")
menu_num = int(menu_num)
if menu_num ==1:
    money = input("바꾸실 원화 금액을 입력해주세요.")
    doller = float(money)/1306.5
    print(f"{doller}USD 입니다.")
else:
    money = input("바꾸실 달러 금액을 입력해주세요.")
    won = float(money)*1306.5
    print(f"{won}USD 입니다.")
```

    원하시는 계산 방법 숫자를 입력해주세요. 
     1 : 원화 -> 달러 계산 
     2 : 달러 -> 원화 계산1
    바꾸실 원화 금액을 입력해주세요.1306.5
    1.0USD 입니다.
    


```python
jupyter nbconvert --to markdown Python_Basic_01.ipynb
```


      File "C:\Users\tjtmd\AppData\Local\Temp/ipykernel_23960/4033320348.py", line 1
        jupyter nbconvert --to markdown Python_Basic_01.ipynb
                ^
    SyntaxError: invalid syntax
    



```python

```
