# 1.7 함수형 프로그래밍 요소

> [목차로 돌아가기](../../README.md) | [이전: 내장 전역 함수](./1_6_builtin_functions.md)

파이썬은 객체지향 언어이면서 동시에 함수형 프로그래밍의 많은 개념을 지원한다. 함수형 프로그래밍은 함수를 일급 객체(first-class citizens)로 다루고 데이터 변형보다 함수 적용에 집중하는 프로그래밍 패러다임이다.

## 1.7.1 람다 함수(Lambda Functions)

### a. 람다 함수의 기본 개념

람다 함수는 익명 함수(anonymous function)로, 간단한 함수를 한 줄로 정의할 수 있다. 주로 함수를 인자로 전달하거나 반환할 때 사용된다.

```python
# 기본 문법: lambda 매개변수: 표현식
add = lambda x, y: x + y
print(add(5, 3))  # 8

# 일반 함수와 비교
def add_regular(x, y):
    return x + y

print(add_regular(5, 3))  # 8
```

### b. 람다 함수의 특징 및 제한사항

람다 함수는 간결하지만, 표현력에 제한이 있다:

```python
# 람다 함수는 단일 표현식만 포함 가능
# 올바른 사용
square = lambda x: x**2

# 불가능한 방식 (여러 문장, 조건문 등)
# lambda x: 
#    if x > 0:
#        return x**2
#    else:
#        return 0

# 대신 조건부 표현식은 가능
conditional_square = lambda x: x**2 if x > 0 else 0
```

### c. 람다 함수의 실용적 활용

람다 함수는 고차 함수(higher-order functions)와 함께 사용할 때 특히 유용하다:

```python
# 정렬 시 key 함수로 활용
people = [('Alice', 25), ('Bob', 20), ('Charlie', 30)]
sorted_by_age = sorted(people, key=lambda person: person[1])
print(sorted_by_age)  # [('Bob', 20), ('Alice', 25), ('Charlie', 30)]

# 필터링
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4, 6, 8, 10]

# 데이터 변환
squares = list(map(lambda x: x**2, numbers))
print(squares)  # [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
```

## 1.7.2 일급 함수(First-Class Functions)

### a. 함수를 변수에 할당

파이썬에서 함수는 일급 객체이므로 변수에 할당할 수 있다:

```python
def greet(name):
    return f"Hello, {name}!"

# 함수를 변수에 할당
greeting_function = greet
print(greeting_function("Alice"))  # "Hello, Alice!"
```

### b. 함수를 인자로 전달

함수를 다른 함수의 인자로 전달할 수 있다:

```python
def apply_function(func, value):
    return func(value)

def double(x):
    return x * 2

def square(x):
    return x ** 2

print(apply_function(double, 5))  # 10
print(apply_function(square, 5))  # 25
print(apply_function(lambda x: x + 1, 5))  # 6
```

### c. 함수를 반환값으로 사용

함수가 다른 함수를 반환할 수 있다:

```python
def create_multiplier(factor):
    def multiply(x):
        return x * factor
    return multiply

double = create_multiplier(2)
triple = create_multiplier(3)

print(double(5))  # 10
print(triple(5))  # 15
```

### d. 클로저(Closure)

함수가 자신이 생성된 환경의 변수를 기억하는 것을 클로저라고 한다:

```python
def counter():
    count = 0
    def increment():
        nonlocal count  # 외부 함수의 변수 사용
        count += 1
        return count
    return increment

c = counter()
print(c())  # 1
print(c())  # 2
print(c())  # 3
```

## 1.7.3 함수형 프로그래밍 도구

### a. 내장 함수형 프로그래밍 도구

파이썬은 함수형 프로그래밍을 위한 여러 내장 함수를 제공한다:

```python
# map(): 모든 요소에 함수 적용
numbers = [1, 2, 3, 4, 5]
squared = map(lambda x: x**2, numbers)
print(list(squared))  # [1, 4, 9, 16, 25]

# filter(): 조건을 만족하는 요소만 선택
evens = filter(lambda x: x % 2 == 0, numbers)
print(list(evens))  # [2, 4]

# functools.reduce(): 누적 연산 수행
from functools import reduce
sum_all = reduce(lambda x, y: x + y, numbers)
print(sum_all)  # 15 (1 + 2 + 3 + 4 + 5)
```

### b. 함수형 프로그래밍을 위한 functools 모듈

`functools` 모듈은 함수형 프로그래밍을 위한 고급 도구를 제공한다:

```python
from functools import partial, lru_cache

# partial(): 함수의 일부 인자를 고정
def power(base, exponent):
    return base ** exponent
    
square = partial(power, exponent=2)
cube = partial(power, exponent=3)

print(square(4))  # 16
print(cube(4))    # 64

# lru_cache: 함수 호출 결과를 캐싱
@lru_cache(maxsize=None)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 캐싱 덕분에 중복 계산 없이 효율적으로 계산
print(fibonacci(30))  # 빠른 계산 가능
```

## 1.7.4 파이썬에서 효과적인 함수형 프로그래밍

### a. 리스트 컴프리헨션과 제너레이터 표현식

파이썬은 함수형 스타일의 데이터 변환을 위한 간결한 구문을 제공한다:

```python
# 리스트 컴프리헨션 (map과 filter 대체)
numbers = [1, 2, 3, 4, 5]

# map 대체
squares = [x**2 for x in numbers]  # [1, 4, 9, 16, 25]

# filter 대체
evens = [x for x in numbers if x % 2 == 0]  # [2, 4]

# map + filter 조합 대체
even_squares = [x**2 for x in numbers if x % 2 == 0]  # [4, 16]

# 제너레이터 표현식 (지연 평가)
sum_squares = sum(x**2 for x in range(1000000))  # 메모리 효율적
```

### b. 함수형 프로그래밍의 장점과 한계

함수형 프로그래밍의 장점:

- 코드의 가독성과 재사용성 향상
- 부수 효과(side effects) 최소화로 디버깅 용이
- 병렬 처리에 적합

파이썬에서의 한계:

- 불변성(immutability)을 강제하지 않음
- 꼬리 재귀 최적화(tail recursion optimization) 부재
- 재귀 호출 깊이 제한 (기본 1000)

```python
# 함수형 스타일의 순수 함수
def add_pure(x, y):
    return x + y  # 부수 효과 없음

# 객체지향과 함수형 스타일 혼합
class Counter:
    def __init__(self, initial=0):
        self.value = initial
        
    def increment(self, step=1):
        # 불변성 유지를 위해 새 객체 반환
        return Counter(self.value + step)
        
    def __str__(self):
        return str(self.value)

c1 = Counter(5)
c2 = c1.increment(3)
print(f"c1: {c1}, c2: {c2}")  # c1: 5, c2: 8
```

함수형 프로그래밍 원칙을 따르더라도 파이썬의 멀티패러다임 특성을 활용하는 것이 최선의 접근법이다.

> [목차로 돌아가기](../../README.md) | [이전: 내장 전역 함수](./1_6_builtin_functions.md)
