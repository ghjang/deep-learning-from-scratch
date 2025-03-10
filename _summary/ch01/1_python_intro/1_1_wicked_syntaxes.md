# 1.1 파이썬 특유의 주요 문법

> [목차로 돌아가기](../../README.md) | [다음: 타입 힌트와 타입 시스템](1_2_type_system.md)

다른 주요 언어와 구분되는 파이썬의 특징적인 문법과 기능을 간단히 정리한다.

## 1.1.1 들여쓰기는 파이썬의 블록 구분 문법이다

파이썬은 중괄호(`{}`)가 아닌 들여쓰기를 사용해 코드 블록을 구분한다:
  
  ```python
  # 올바른 들여쓰기
  if x > 0:
      print("양수입니다")
      y = x * 2
  else:
      print("0 또는 음수입니다")
      y = -x * 2
  
  # 들여쓰기 오류 예시
  if x > 0:
      print("양수입니다")
    y = x * 2  # IndentationError 발생: 들여쓰기 수준이 일치하지 않음
  ```

중첩된 블록의 지정은 다음과 같이 작성할 수 있다:

```python
def outer_function():
    x = 10
    
    class InnerClass:
        def display(self):
            print(x)  # 외부 변수 접근
    
    def inner_function():
        print(x)  # 외부 변수 접근
        inner_instance = InnerClass()
        inner_instance.display()  # 이너 클래스 메서드 호출
        
    inner_function()  # 내부 함수 호출

outer_function()  # 외부 함수 호출
```

들여쓰기 주의사항:
  
* __일관성__: 같은 블록의 들여쓰기 수준은 동일해야 함
* __공백과 탭 혼용 금지__: 같은 파일에서 공백과 탭을 혼용하지 않음
* __권장 방식__: PEP 8에 따라 4개의 공백 사용 권장
* __빈 블록 처리__: 빈 블록에는 `pass`나 `...` 사용

## 1.1.2 `**` 연산자는 '거듭제곱'을 나타낸다

`**` 연산자는 거듭제곱 계산을 수행한다:

```python
# 기본 사용법
print(2 ** 3)      # 8 (2의 3제곱)
print(10 ** 2)     # 100 (10의 2제곱)

# 음수와 실수 지수도 가능
print(4 ** 0.5)    # 2.0 (4의 제곱근)
print(2 ** -1)     # 0.5 (2의 -1제곱)

# 변수와 함께 사용
base = 2
exponent = 10
print(base ** exponent)  # 1024 (2의 10제곱)
```

## 1.1.3 `//` 연산자는 '나눗셈의 몫'을 나타낸다

`//` 연산자는 나눗셈 후 소수점 이하를 버린 정수 결과(몫)를 반환한다:

```python
# 기본 사용법
print(7 // 3)     # 2 (7을 3으로 나눈 몫)
print(10 // 3)    # 3
print(-7 // 3)    # -3 (음수는 더 작은 정수로 내림)

# 실수형과 함께 사용 시에도 내림 나눗셈 수행
print(7.0 // 3)   # 2.0
print(7 // 3.0)   # 2.0

# 일반 나눗셈('/')과 비교
print(7 / 3)      # 2.3333333333333335 (정확한 나눗셈 결과)
print(7 // 3)     # 2 (몫만 반환)
```

## 1.1.4 `:=`(왈러스 연산자)는 할당과 표현식을 동시에 수행한다

파이썬 3.8부터 도입된 왈러스 연산자는 변수에 값을 할당하면서 동시에 그 표현식을 평가할 수 있게 해준다:

  ```python
  # 일반적인 방식
  data = get_data()
  if data:
      process(data)
      
  # 왈러스 연산자 사용
  if data := get_data():
      process(data)
      
  # 반복문에서의 활용
  while chunk := file.read(8192):
      process(chunk)
      
  # 리스트 컴프리헨션에서 활용
  results = [transformed for x in data if (transformed := transform(x)) is not None]
  ```

## 1.1.5 `if` 키워드는 제어문, 표현식, 가드 조건 등 다양한 맥락에서 사용된다

파이썬에서 `if` 키워드는 다른 언어와 달리 여러 문맥에서 유연하게 활용된다:

### a. 기본 제어 흐름 제어문

가장 기본적인 형태로, 코드 실행 경로를 분기한다:

```python
# 기본적인 if-elif-else 제어 구조
if x > 0:
    print("양수입니다")
elif x == 0:
    print("0입니다")
else:
    print("음수입니다")
```

### b. 조건부 표현식

파이썬에서는 `if-else` 문을 표현식으로 사용해 한 줄로 조건에 따른 값 할당이 가능하다:

```python
# 조건부 표현식 (conditional expression)
result = "양수" if x > 0 else "음수 또는 0"
```

조건부 표현식의 구문은 다른 언어들의 삼항 연산자와 비슷하지만 더 영어 문장에 가깝다:

```python
# 파이썬 조건부 표현식
value = true_expr if condition else false_expr

# 다른 언어의 삼항 연산자 (예: JavaScript, Java, C++)
# value = condition ? true_expr : false_expr
```

#### 활용 예시

```python
# 함수 반환값에 사용
def get_absolute(x):
    return x if x >= 0 else -x

# 리스트 컴프리헨션과 함께 사용
numbers = [1, -2, 3, -4, 5]
abs_numbers = [n if n >= 0 else -n for n in numbers]
print(abs_numbers)  # [1, 2, 3, 4, 5]

# 딕셔너리와 함께 사용
user = {"name": "Kim", "admin": True}
greeting = f"환영합니다 {'관리자' if user['admin'] else '사용자'} {user['name']}님"
print(greeting)  # "환영합니다 관리자 Kim님"

# 함수 인자로 사용
print("합격" if score >= 60 else "불합격")
```

#### 중첩된 조건부 표현식

조건부 표현식은 중첩될 수 있지만, 가독성이 떨어질 수 있으므로 주의해야 한다:

```python
# 중첩 조건부 표현식
result = "양수" if x > 0 else "0" if x == 0 else "음수"

# 더 읽기 쉬운 동일 코드
if x > 0:
    result = "양수"
elif x == 0:
    result = "0"
else:
    result = "음수"
```

### c. 패턴 매칭의 가드 조건

Python 3.10부터 도입된 `match-case`에서 `if`는 패턴 매칭 후 추가 조건을 검사하는 가드 역할을 한다:

```python
def check_point(point):
    match point:
        case (x, y) if x == y:  # 패턴 매칭 후 가드 조건 검사
            return f"대각선 위의 점 ({x}, {y})"
        case (x, y) if x > 0 and y > 0:
            return f"제1사분면의 점 ({x}, {y})"
        case _:
            return "기타 위치의 점"
```

### d. 리스트 컴프리헨션의 필터링

리스트 컴프리헨션에서 `if`는 요소를 필터링하는 역할을 한다:

```python
# if를 사용한 필터링
even_numbers = [x for x in range(10) if x % 2 == 0]
print(even_numbers)  # [0, 2, 4, 6, 8]

# 여러 조건 조합
matrix = [[1, 2], [3, 4], [5, 6]]
flattened = [x for row in matrix if len(row) > 0 for x in row if x % 2 == 0]
print(flattened)  # [2, 4, 6]
```

파이썬의 `if` 키워드는 이처럼 다양한 맥락에서 유연하게 사용될 수 있어, 상황에 적합한 간결한 코드 작성이 가능하다.

## 1.1.6 언패킹 연산자 `*`, `**`는 시퀀스와 매핑을 풀어준다

파이썬에서는 두 가지 유형의 언패킹 연산자가 있다. 이러한 언패킹 연산자는 보통의 문장에서도 사용할 수도 있고, 함수 정의에서 가변 인자를 처리할 때 유용하게 사용된다.

### `*` 연산자 - 시퀀스 언패킹(리스트, 튜플, 문자열 등)
  
```python
# 시퀀스 분해하여 변수에 할당
first, *middle, last = [1, 2, 3, 4, 5]
print(first)   # 1
print(middle)  # [2, 3, 4]
print(last)    # 5

# 함수 호출 시 시퀀스를 개별 인자로 확장
def add(a, b, c):
    return a + b + c
    
values = [1, 2, 3]
print(add(*values))  # 6  (add(1, 2, 3)와 동일)

# 리스트 병합
list1 = [1, 2, 3]
list2 = [4, 5]
combined = [*list1, *list2]
print(combined)  # [1, 2, 3, 4, 5]
```
  
### `**` 연산자 - 딕셔너리 언패킹(키-값 쌍)
  
```python
# 딕셔너리를 키워드 인자로 확장
def create_profile(name, age, job):
    return f"{name}, {age}, {job}"

data = {"name": "Kim", "age": 30, "job": "Developer"}
profile = create_profile(**data)  # create_profile(name="Kim", age=30, job="Developer")와 동일

# 딕셔너리 병합 (Python 3.5+)
defaults = {"color": "red", "size": "medium"}
options = {"size": "large", "material": "cotton"}
settings = {**defaults, **options}
print(settings)  # {'color': 'red', 'size': 'large', 'material': 'cotton'}
# 중복 키는 나중에 오는 딕셔너리의 값으로 덮어씌워짐
```

### 함수 정의에서의 언패킹 - 가변 인자 처리
  
```python
# *args: 가변 길이 위치 인자
def sum_all(*args):
    """임의 개수의 인자를 받아 모두 더하는 함수"""
    total = 0
    for num in args:
        total += num
    return total

print(sum_all(1, 2, 3))        # 6
print(sum_all(10, 20, 30, 40)) # 100

# **kwargs: 가변 길이 키워드 인자
def print_info(**kwargs):
    """임의 개수의 키워드 인자를 받아 출력하는 함수"""
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Kim", age=30, job="Developer")
# 출력:
# name: Kim
# age: 30
# job: Developer

# *args와 **kwargs 함께 사용
def flexible_function(*args, **kwargs):
    """위치 인자와 키워드 인자를 모두 받는 유연한 함수"""
    print(f"위치 인자: {args}")
    print(f"키워드 인자: {kwargs}")

flexible_function(1, 2, 3, name="Test", value=True)
# 출력:
# 위치 인자: (1, 2, 3)
# 키워드 인자: {'name': 'Test', 'value': True}
```

__일반 파라미터와 언패킹 파라미터 혼합:__

파이썬에서 함수 정의 시 일반 파라미터와 언패킹 파라미터를 함께 사용할 때는 엄격한 순서 규칙을 따라야 한다.

```python
# 파라미터 순서 규칙
def complex_function(
    pos1, pos2,              # 1. 일반 위치 파라미터
    /,                       # 위치 전용 파라미터 구분자 (3.8+)
    pos_or_kwd1, pos_or_kwd2,# 2. 위치 또는 키워드로 전달 가능한 파라미터
    *args,                   # 3. 추가 위치 인자 (튜플로 수집)
    kwd1, kwd2,              # 4. 키워드 전용 파라미터 (키워드로만 전달 가능)
    **kwargs                 # 5. 추가 키워드 인자 (딕셔너리로 수집)
):
    print(f"위치 전용: {pos1}, {pos2}")
    print(f"위치 또는 키워드: {pos_or_kwd1}, {pos_or_kwd2}")
    print(f"추가 위치 인자: {args}")
    print(f"키워드 전용: {kwd1, kwd2}")
    print(f"추가 키워드 인자: {kwargs}")

# 호출 예시
complex_function(
    1, 2,                    # pos1, pos2 (위치 전용)
    "a", "b",                # pos_or_kwd1, pos_or_kwd2
    "extra1", "extra2",      # *args로 수집됨
    kwd1="k1", kwd2="k2",    # kwd1, kwd2 (키워드 전용)
    extra_kwd="extra_val"    # **kwargs로 수집됨
)
```

__주의사항:__

* __순서 규칙 위반은 SyntaxError 발생__:

    ```python
    # 잘못된 예: *args가 키워드 전용 인자보다 뒤에 위치
    def wrong_order(a, b, kwd_only=None, *args):  # 구문 오류 발생!
        pass
    ```

* __기본값은 위치 전용이나 위치/키워드 인자에만 지정 가능__:

    ```python
    # 올바른 예
    def func(a, b=10, *, c, d=20):
        pass
        
    func(1, c=3)  # a=1, b=10, c=3, d=20
    ```

* __파이썬 3.8부터 위치 전용 파라미터 도입__:

    `/` 구분자 이전의 파라미터는 반드시 위치 인자로만 전달해야 함

    ```python
    def pos_only_func(x, y, /, z):
        print(x, y, z)
        
    pos_only_func(1, 2, z=3)    # 가능
    pos_only_func(1, 2, 3)      # 가능
    pos_only_func(x=1, y=2, z=3)# 오류! x, y는 위치 전용
    ```

이 언패킹 연산자들은 코드를 더 간결하게 만들고, 가변 길이의 인자를 다룰 때 유용하다.

## 1.1.6 `is`는 객체 식별자 비교를 수행하는 키워드이다

`is`는 두 변수가 메모리상에서 동일한 객체를 참조하는지 비교하는 키워드이다.
  
* `==` 연산자: 논리적 동등성(logical equality) 비교 - 두 객체의 값이 같은지 비교
* `is` 키워드: 물리적 동일성(physical identity) 비교 - 두 변수가 실제로 메모리 상 같은 객체를 가리키는지 비교
  
  ```python
  a = [1, 2, 3]
  b = [1, 2, 3]
  c = a
  
  print(a == b)  # True (값이 같음)
  print(a is b)  # False (다른 객체)
  print(a is c)  # True (같은 객체)
  ```

## 1.1.7 `as`는 컨텍스트에 따라 다른 역할을 하는 키워드이다

`as`는 파이썬에서 컨텍스트에 따라 다른 의미로 사용되는 키워드이다. 다음과 같은 방식으로 `as` 키워드는 다양한 문맥에서 "이름 바인딩"이라는 일관된 개념을 유지하면서도 상황에 맞는 특화된 역할을 수행한다.

### `import` 문에서 - 모듈이나 객체에 별칭(alias)을 부여

```python
import numpy as np
from datetime import datetime as dt
```
  
### `with` 문에서 - 컨텍스트 관리자의 반환값을 변수에 바인딩

```python
with open('file.txt') as file:
    content = file.read()
```
  
### `except` 절에서 - 발생한 예외 객체를 변수에 바인딩

```python
try:
    result = 1 / 0
except ZeroDivisionError as error:
    print(f"에러 발생: {error}")
```
  
### 패턴 매칭(Python 3.10+)에서 - 매칭된 값에 이름 부여

```python
command = ["save", "example.txt"]   # 예시 커맨드

match command:
    case ["quit" | "exit" as cmd]:
        print(f"{cmd} 명령으로 종료합니다")
    case ["save" as action, filename]:  # 예시 커맨드는 이 경우에 매칭됨.
        print(f"{action}: {filename}에 저장합니다")
```
  
## 1.1.8 `yield`는 제너레이터 함수를 만드는 키워드이다

`yield`는 함수가 값을 반환하면서도 실행 상태를 유지하게 해주는 키워드로, 제너레이터 함수를 정의하는 데 사용된다:

  ```python
  # 기본적인 제너레이터 함수
  def count_up_to(max):
      count = 1
      while count <= max:
          yield count  # 값을 반환하고 함수 상태 저장
          count += 1   # 다음 호출 시 여기서부터 재개
          
  # 제너레이터 사용
  counter = count_up_to(5)  # 제너레이터 객체 생성 (아직 실행되지 않음)
  
  print(next(counter))  # 1 (첫 번째 yield까지 실행)
  print(next(counter))  # 2 (두 번째 yield까지 실행)
  
  # for 루프로 나머지 값 소비
  for number in counter:
      print(number)     # 3, 4, 5 출력
  ```

제너레이터는 다음과 같은 특징을 갖는다:

* __메모리 효율성__: 모든 결과를 한 번에 메모리에 저장하지 않고 필요할 때만 계산
* __지연 평가__: 요청할 때만 다음 값을 생성 (lazy evaluation)
* __상태 유지__: 함수의 실행 상태가 유지되어 다음 호출 시 이어서 실행
  
```python
# 대용량 데이터 처리 예시
def read_large_file(file_path, chunk_size=1024):
    with open(file_path, 'r') as file:
        while True:
            data = file.read(chunk_size)
            if not data:  # 파일의 끝에 도달
                break
            yield data
                
# 수 기가바이트 파일도 적은 메모리로 처리 가능
for chunk in read_large_file('huge_log.txt'):
    process_data(chunk)
```
  
`yield from` 구문으로 다른 이터러블의 값을 위임해서 yield할 수 있다:
  
```python
def chain(*iterables):
    for it in iterables:
        yield from it  # it의 각 항목을 하나씩 yield
        
# 사용 예시
result = list(chain([1, 2], [3, 4, 5], [6]))
print(result)  # [1, 2, 3, 4, 5, 6]
```

## 1.1.9 컨텍스트 매니저(with 문)는 자원 관리를 자동화한다

컨텍스트 매니저는 파이썬의 "스코프 가드" 역할을 하는 구문으로, 리소스 획득과 해제를 자동화한다. 내부적으로는 `__enter__`와 `__exit__` 메서드 쌍으로 동작한다:

```python
# 파일 자동 닫기
with open('file.txt', 'r') as file:  # __enter__ 호출: 파일 열기, 반환값을 file에 바인딩
    data = file.read()
# 블록을 벗어날 때 자동으로 __exit__ 호출: 파일 닫기
```

컨텍스트 매니저의 내부 동작 흐름:

1. `with` 문 시작: 표현식의 `__enter__` 메서드 호출
2. `__enter__` 메서드의 반환 값을 `as` 뒤의 변수에 할당
3. 블록 내 코드 실행
4. 블록 종료 시 __어떤 상황에서든__ `__exit__` 메서드 호출
    * 정상 종료: `__exit__(None, None, None)` 호출
    * 예외 발생: `__exit__(exc_type, exc_value, traceback)` 호출

```python
# 예외 처리 자동화 예시
class Transaction:
    def __enter__(self):
        print("트랜잭션 시작")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            print("트랜잭션 커밋")
            return True  # 정상 종료
        else:
            print(f"오류 발생: {exc_type}, 롤백 수행")
            return False  # 예외 전파 (True 반환 시 예외 억제)

# 정상 케이스
with Transaction() as tx:
    print("작업 수행")
# 출력:
# 트랜잭션 시작
# 작업 수행
# 트랜잭션 커밋

# 오류 케이스
try:
    with Transaction() as tx:
        print("작업 시작")
        raise ValueError("오류 발생!")  # 의도적 예외
except ValueError:
    print("예외 처리됨")
# 출력:
# 트랜잭션 시작
# 작업 시작
# 오류 발생: <class 'ValueError'>, 롤백 수행
# 예외 처리됨
```
  
contextmanager 데코레이터를 사용하여 컨텍스트 매니저를 더 간단하게 구현해 사용할 수 있는 경우도 있다:
  
```python
from contextlib import contextmanager

@contextmanager
def managed_resource():
    print("리소스 획득")    # __enter__ 부분
    try:
        yield "리소스"      # as 뒤 변수에 할당되는 값
    finally:                # 항상 실행됨 (__exit__ 부분)
        print("리소스 해제") # 예외가 발생해도 반드시 실행
        
with managed_resource() as resource:
    print(f"리소스 사용 중: {resource}")
```
  
컨텍스트 매니저는 파일 핸들링, 데이터베이스 연결, 락(lock) 관리, 트랜잭션 제어 등 "획득-사용-해제" 패턴이 필요한 상황에서 코드 안전성을 높여준다.

## 1.1.10 `match-case`는 파이썬의 패턴 매칭 구문이다

파이썬 3.10부터 도입된 `match-case`는 다른 언어의 `switch-case`와 유사하지만 더 강력한 구조적 패턴 매칭 기능을 제공한다.

```python
# 기본 사용법
def process_command(command):
    match command:
        case "quit":
            return "종료합니다"
        case "help":
            return "도움말을 표시합니다"
        case _:  # 와일드카드 패턴 (default 역할)
            return f"알 수 없는 명령: {command}"
```

### 패턴 매칭의 주요 기능

#### a. __구조 분해 기능__: 복합 데이터 타입의 내부 값을 변수에 바인딩

```python
# 튜플 패턴
def process_point(point):
    match point:
        case (0, 0):
            return "원점"
        case (0, y):  # y는 두 번째 요소 값을 담는 변수
            return f"y축 위의 점 (0, {y})"
        case (x, 0):  # x는 첫 번째 요소 값을 담는 변수
            return f"x축 위의 점 ({x}, 0)"
        case (x, y):  # x, y에 각각 요소 값이 바인딩됨
            return f"좌표 ({x}, {y})"
        case _:
            return "유효한 좌표가 아님"

print(process_point((0, 5)))  # "y축 위의 점 (0, 5)"
```

#### b. __시퀀스 패턴__: 리스트 등 시퀀스 타입의 요소 매칭

```python
def process_command_list(commands):
    match commands:
        case []:
            return "명령이 없습니다"
        case ["quit"]:
            return "종료 명령"
        case ["load", filename]:  # 두 번째 요소를 filename에 바인딩
            return f"{filename} 파일 로드"
        case ["save", filename, "backup"]:
            return f"{filename} 백업 저장"
        case ["save", *filenames]:  # 여러 파일 이름을 filenames 리스트에 바인딩
            return f"여러 파일 저장: {filenames}"

print(process_command_list(["save", "doc1.txt", "doc2.txt"]))  # "여러 파일 저장: ['doc1.txt', 'doc2.txt']"
```

#### c. __클래스 패턴__: 객체의 속성 기반 매칭

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def locate_point(point):
    match point:
        case Point(x=0, y=0):
            return "원점"
        case Point(x=0, y=y):  # x가 0이고, y 속성값을 y 변수에 바인딩
            return f"y축 위의 점, y={y}"
        case Point(x=x, y=0):  # y가 0이고, x 속성값을 x 변수에 바인딩
            return f"x축 위의 점, x={x}"
        case Point():
            return "일반 좌표"
        case _:
            return "점이 아님"

print(locate_point(Point(0, 5)))  # "y축 위의 점, y=5"
```

### 타입 패턴과 가드 조건

`match-case`에서는 값의 타입을 검사하고 값을 변수에 바인딩할 수 있으며, 추가 조건(`if` 가드)도 지정할 수 있다:

```python
def check_value(value):
    match value:
        case int(n) if n < 0:  # value가 int 타입이고 음수일 때
            # n에는 value의 값이 바인딩됨
            return f"음수 정수: {n}"
        case int(n) if n > 0:  # value가 int 타입이고 양수일 때
            return f"양수 정수: {n}"
        case int(0):           # value가 정확히 정수 0일 때
            return "0"
        case float(f):         # value가 float 타입일 때, f에 그 값이 바인딩됨
            return f"실수: {f}"
        case str(s):           # value가 str 타입일 때, s에 그 값이 바인딩됨
            return f"문자열: {s}"
        case _:                # 그 외 모든 경우
            return f"기타 타입: {type(value).__name__}"

# 실행 예시
print(check_value(-5))      # "음수 정수: -5"
print(check_value(3.14))    # "실수: 3.14"
print(check_value("hello")) # "문자열: hello"
print(check_value([1,2,3])) # "기타 타입: list"
```

여기서 중요한 점:

* `int(n)`, `float(f)` 등은 __생성자 호출이 아님__
* 이것은 "value가 int 타입이면 그 값을 n에 바인딩하라"는 패턴 매칭 구문
* 변수 바인딩이 성공하면 해당 케이스 내에서 변수를 사용할 수 있음
* 가드 조건(`if n < 0` 등)은 패턴이 매칭된 후 추가 필터링을 위해 사용

### 복합 패턴 예시

여러 패턴을 조합하여 복잡한 데이터 구조를 효과적으로 처리할 수 있다:

```python
def process_data(data):
    match data:
        case {"type": "user", "name": name, "admin": True}:
            return f"관리자: {name}"
        case {"type": "user", "name": name}:
            return f"일반 사용자: {name}"
        case {"type": "product", "items": [{"name": item_name, "price": price}, *_]}:
            return f"첫 상품: {item_name}, 가격: {price}"
        case [{"id": id_val, **rest}, *_] if id_val > 100:
            return f"ID {id_val}의 기록, 추가 정보: {rest}"
        case _:
            return "알 수 없는 데이터 형식"
```

`match-case`는 데이터 분석, 파싱, 이벤트 처리 등 복잡한 구조를 다루는 코드를 간결하고 가독성 있게 작성하는 데 큰 도움이 된다.

## 1.1.11 `...`(Ellipsis)는 `pass`와 유사하게 사용된다

`pass`는 '아무것도 하지 않는 문장'이다. 예를 들어서 `if` 문에서 '아무것도 하지 않을 때' 사용할 수 있다:

```python
if x < 0:
    print('negative')
else:
    ... # 올바른 파이썬 문법이다.
```

pass와 마찬가지로 Ellipsis(...)는 문법적으로 유효한 표현식이지만 아무 작업도 수행하지 않는다.
  
```python
def todo_function():
    ...  # 아직 구현하지 않은 함수

class TodoClass:
    ...  # 아직 구현하지 않은 클래스
```

> [목차로 돌아가기](../../README.md) | [다음: 타입 힌트와 타입 시스템](1_2_type_system.md)
