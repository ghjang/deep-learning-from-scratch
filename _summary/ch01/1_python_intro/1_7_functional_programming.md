# 1.7 함수형 프로그래밍 요소

> [목차로 돌아가기](../../README.md) | [이전: 내장 전역 함수](./1_6_builtin_functions.md)

파이썬은 객체지향 언어이면서 동시에 함수형 프로그래밍의 많은 개념을 지원한다. 함수형 프로그래밍은 함수를 일급 객체(first-class citizens)로 다루고 데이터 변형보다 함수 적용에 집중하는 프로그래밍 패러다임이다.

## 1.7.1 람다 함수(Lambda Functions)

### a. 람다 함수의 기본 개념

람다 함수는 익명 함수(anonymous function)로, 간단한 함수를 한 줄로 정의할 수 있다. 주로 함수를 인자로 전달하거나 반환할 때 사용된다.

```python
# 기본 문법: lambda 매개변수: 표현식
add = lambda x, y: x + y
print(add(5, 3))  # 8 - 두 숫자의 합 반환

# 일반 함수와 비교
def add_regular(x, y):
    return x + y

print(add_regular(5, 3))  # 8 - 동일한 결과
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

**중요한 점:** 파이썬의 람다는 다른 언어와 달리 단 하나의 표현식으로만 제한된다. 이는 파이썬의 창시자인 귀도 반 로썸(Guido van Rossum)의 설계 철학에서 비롯되었다. 그는 복잡한 동작은 명시적인 이름 있는 함수로 구현하고, 람다는 정말 간단한 일회용 함수에만 사용하는 것이 코드 가독성을 높인다고 생각했다. 람다 함수는 간단하고 즉시 사용되는 함수에 가장 적합하다.

```python
# JavaScript에서 가능하지만 파이썬에서는 불가능한 람다 함수
# JavaScript: (x) => { 
#    if (x > 0) { 
#        return x * x; 
#    } else { 
#        return 0; 
#    }
# }

# 파이썬에서는 반드시 일반 함수로 정의해야 함
def complex_square(x):
    if x > 0:
        return x * x
    else:
        return 0
```

### c. 람다 함수의 실용적 활용

람다 함수는 고차 함수(higher-order functions)와 함께 사용할 때 특히 유용하다:

```python
# 정렬 시 key 함수로 활용
people = [('Alice', 25), ('Bob', 20), ('Charlie', 30)]
sorted_by_age = sorted(people, key=lambda person: person[1])
print(sorted_by_age)  # [('Bob', 20), ('Alice', 25), ('Charlie', 30)] - 나이순 정렬됨

# 필터링
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4, 6, 8, 10] - 짝수만 필터링됨

# 데이터 변환
squares = list(map(lambda x: x**2, numbers))
print(squares)  # [1, 4, 9, 16, 25, 36, 49, 64, 81, 100] - 각 숫자의 제곱값
```

## 1.7.2 일급 함수(First-Class Functions)

### a. 함수를 변수에 할당

파이썬에서 함수는 일급 객체이므로 변수에 할당할 수 있다:

```python
def greet(name):
    return f"Hello, {name}!"

# 함수를 변수에 할당
greeting_function = greet
print(greeting_function("Alice"))  # "Hello, Alice!" - 원래 함수처럼 호출 가능
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

print(apply_function(double, 5))  # 10 - 5 * 2
print(apply_function(square, 5))  # 25 - 5² = 25
print(apply_function(lambda x: x + 1, 5))  # 6 - 5 + 1
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

print(double(5))  # 10 - 5 * 2
print(triple(5))  # 15 - 5 * 3
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
print(c())  # 1 - 첫 호출
print(c())  # 2 - 두 번째 호출
print(c())  # 3 - 세 번째 호출
```

**중요한 점:** 파이썬은 객체 기반 언어이므로 모든 변수는 객체에 대한 참조를 저장한다. 클로저는 이러한 참조를 캡처하는 방식(참조 방식의 변수 캡처)으로 작동한다. 클로저는 함수가 생성될 때의 환경을 '기억'하기 때문에, 함수형 프로그래밍에서 상태를 유지하면서도 부작용을 최소화하는 강력한 도구다.

**가변 객체 vs 불변 객체 캡처:** 클로저에서 가변 객체(list, dict 등)와 불변 객체(int, str 등)를 참조할 때 동작이 다르다:

```python
def outer():
    # 불변 객체 (재할당에는 nonlocal 필요)
    count = 0
    
    # 가변 객체 (내용 수정에는 nonlocal이 필요 없음)
    items = []
    
    def inner_counter():
        nonlocal count  # 불변 객체를 변경할 때 필요
        count += 1      # 재할당이므로 nonlocal 없이는 불가능
        return count
    
    def inner_collector(item):
        items.append(item)  # 가변 객체 수정은 nonlocal 없이도 가능
        return items
    
    return inner_counter, inner_collector

counter, collector = outer()

print(counter())       # 1
print(counter())       # 2
print(collector("a"))  # ['a']
print(collector("b"))  # ['a', 'b']
```

위 예제에서 볼 수 있듯이:

1. 불변 객체(`count`)는 값 자체를 변경할 수 없기 때문에 값을 변경하려면 재할당이 필요하다. 재할당을 위해서는 `nonlocal` 키워드가 필요하다.
2. 가변 객체(`items`)는 객체 참조를 통해 직접 내용을 수정할 수 있다. 객체 자체를 변경하는 것이 아니라 내용을 수정하는 것이기 때문에 `nonlocal` 키워드 없이도 가능하다.

이러한 차이는 파이썬에서 객체가 어떻게 다뤄지는지에 대한 기본 원칙과 관련 있다. 불변 객체는 값을 수정할 수 없으므로 새로운 객체를 할당해야 하지만, 가변 객체는 참조를 통해 내용을 직접 수정할 수 있다.

```python
# 참조 방식 캡처로 인한 특징적 동작
def outer():
    numbers = []
    for i in range(3):
        def inner():
            return i  # 변수 i의 참조를 캡처
        numbers.append(inner)
    return numbers

functions = outer()
results = [f() for f in functions]
print(results)  # [2, 2, 2] - 모든 함수가 마지막 i 값(2)을 참조
```

이 예제에서 모든 inner 함수가 동일한 변수 `i`를 참조하므로, 루프가 끝난 후에는 모든 함수가 `i`의 최종값인 2를 반환한다. 이를 해결하려면 함수 호출 시 값을 바인딩하는 추가 클로저가 필요하다:

```python
def outer_fixed():
    numbers = []
    for i in range(3):
        def make_func(i=i):  # 기본 인자로 현재 i 값을 고정
            def inner():
                return i
            return inner
        numbers.append(make_func())
    return numbers

functions = outer_fixed()
results = [f() for f in functions]
print(results)  # [0, 1, 2] - 각 함수가 생성 시점의 i 값을 참조
```

**`i=i` 표현식 설명:** 여기서 `i=i`는 매개변수 기본값 지정 문법으로, 기본값이 무엇인지 이해하는 것이 중요하다:

1. 오른쪽 `i`는 루프의 현재 반복에서의 `i` 값(0, 1, 2 중 하나)을 참조한다
2. 이 값은 **함수 정의 시점에** 평가되어 기본 매개변수로 저장된다
3. 각 함수 정의마다 다른 시점의 `i` 값이 저장된다
4. 정수는 불변(immutable) 객체이므로 그 참조는 변경될 수 없다

이는 마치 값이 복사된 것처럼 보이지만, 실제로는 객체 참조가 저장된 것이다. 불변 객체(int, str 등)의 경우 참조된 객체가 변경될 수 없어서 "값 복사"처럼 동작한다. 클로저에서 루프 변수를 "캡처"할 때 이러한 패턴이 자주 사용된다.

**가변 객체를 기본 인자로 사용할 때 주의사항:** 불변 객체와 달리 가변(mutable) 객체를 함수 파라미터의 기본값으로 사용할 때는 예상치 못한 동작이 발생할 수 있다:

```python
def outer_with_mutable_default():
    functions = []
    for i in range(3):
        def add_to_list(item, result=[]):  # 가변 객체(리스트)를 기본값으로 사용
            result.append(item)
            return result
        functions.append(add_to_list)
    
    return functions

# 가변 기본값의 결과 확인
f1, f2, f3 = outer_with_mutable_default()
print(f1(0))  # [0]
print(f2(1))  # [0, 1] - 같은 리스트 객체가 공유됨!
print(f3(2))  # [0, 1, 2] - 세 함수가 모두 동일한 기본값 공유

# 올바른 패턴: None을 기본값으로 사용
def safer_function(item, result=None):
    if result is None:
        result = []  # 함수 호출마다 새 리스트 생성
    result.append(item)
    return result

print(safer_function(1))  # [1]
print(safer_function(2))  # [2]
```

파이썬에서 함수 기본 인자는 함수가 정의될 때 **한 번만** 평가되어 함수 객체의 속성으로 저장된다. 따라서 가변 객체를 기본값으로 사용하면 모든 함수 호출이 동일한 객체 인스턴스를 공유하게 되어 의도치 않은 부작용이 발생할 수 있다.

**nonlocal 키워드 심층 설명:** 파이썬에서 `nonlocal` 키워드는 중첩 함수에서 외부 함수의 변수를 수정하기 위해 사용된다. 람다 함수에서는 사용할 수 없으며(람다는 단일 표현식만 허용), 일반 중첩 함수에서만 사용 가능하다. 기본적으로 파이썬의 변수 스코프 규칙은 다음과 같다:

1. 내부 함수는 외부 함수의 변수를 **읽을 수** 있다
2. 하지만 내부 함수에서 외부 함수의 변수에 값을 **할당하면**, 파이썬은 이를 내부 함수의 새로운 지역 변수 생성으로 간주한다
3. `nonlocal` 키워드는 "이 변수는 지역 변수가 아니라 외부 함수의 변수를 가리킨다"고 명시한다

```python
def outer():
    x = 1
    
    def inner1():
        print(x)  # 외부 변수 읽기: 가능 (출력: 1)
    
    def inner2():
        x = 2  # 새 지역 변수를 생성 (외부 x와 다름)
        print(x)  # 출력: 2
    
    def inner3():
        nonlocal x  # 이 x는 외부 함수의 x와 동일하다고 선언
        x = 3       # 외부 함수의 x를 수정
        print(x)    # 출력: 3
    
    inner1()
    print(x)  # 출력: 1 (아직 변경 안됨)
    inner2()
    print(x)  # 출력: 1 (inner2는 외부 x에 영향 없음)
    inner3()
    print(x)  # 출력: 3 (inner3가 외부 x를 변경)
```

**다중 중첩 함수에서의 nonlocal:** 함수가 여러 단계로 중첩된 경우, `nonlocal`은 가장 가까운 바깥쪽 스코프부터 변수를 찾는다:

```python
def outer():
    x = "outer"
    
    def middle():
        x = "middle"
        
        def inner():
            nonlocal x  # middle 함수의 x를 참조
            x = "changed middle"
            
            def deepest():
                nonlocal x  # 이미 변경된 middle의 x를 참조
                print(f"deepest: {x}")  # "deepest: changed middle"
            
            deepest()
            
        inner()
        print(f"middle after: {x}")  # "middle after: changed middle"
    
    middle()
    print(f"outer: {x}")  # "outer: outer" (변경되지 않음)

outer()
```

내부 함수에서 더 바깥쪽 스코프의 변수를 직접 참조하려면 여러 단계의 `nonlocal` 선언이 필요할 수 있다:

```python
def outer():
    count = 0
    
    def middle():
        # 여기서 outer의 count를 수정하려면
        nonlocal count
        
        def inner():
            nonlocal count  # 이제 outer의 count를 참조
            count += 1
            print(f"inner: {count}")
        
        inner()
    
    middle()
    print(f"final count: {count}")  # "final count: 1"

outer()
```

**파라미터와 값 복사/참조:** 파이썬에서 함수 파라미터는 항상 "객체 참조에 의한 전달"(call by object reference) 방식으로 동작한다:

```python
def modify(a, b):
    a = 100       # 새 정수 객체를 가리키도록 지역 참조를 변경 (원본 영향 없음)
    b.append(4)   # b가 참조하는 리스트 객체를 직접 수정 (원본 변경됨)
    print(f"함수 내부: a={a}, b={b}")

x = 1         # 불변 객체
y = [1, 2, 3] # 가변 객체
print(f"호출 전: x={x}, y={y}")  # x=1, y=[1, 2, 3]
modify(x, y)                    # 함수 내부: a=100, b=[1, 2, 3, 4]
print(f"호출 후: x={x}, y={y}")  # x=1, y=[1, 2, 3, 4]
```

## 1.7.3 함수형 프로그래밍 도구

### a. 내장 함수형 프로그래밍 도구

파이썬은 함수형 프로그래밍을 위한 여러 내장 함수를 제공한다:

```python
# map(): 모든 요소에 함수 적용
numbers = [1, 2, 3, 4, 5]
squared = map(lambda x: x**2, numbers)
print(list(squared))  # [1, 4, 9, 16, 25] - 각 요소의 제곱값

# filter(): 조건을 만족하는 요소만 선택
evens = filter(lambda x: x % 2 == 0, numbers)
print(list(evens))  # [2, 4] - 짝수만 선택됨

# functools.reduce(): 누적 연산 수행
from functools import reduce
sum_all = reduce(lambda x, y: x + y, numbers)
print(sum_all)  # 15 (1 + 2 + 3 + 4 + 5) - 모든 요소의 합계
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

print(square(4))  # 16 - 4² = 16
print(cube(4))    # 64 - 4³ = 64

# lru_cache: 함수 호출 결과를 캐싱
@lru_cache(maxsize=None)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 캐싱 덕분에 중복 계산 없이 효율적으로 계산
print(fibonacci(30))  # 빠른 계산 가능 (832040)
```

**중요한 점:** `lru_cache` 데코레이터는 특히 재귀 함수나 계산 비용이 높은 함수에서 성능을 크게 향상시킬 수 있다. 이는 동일한 입력에 대한 함수 호출 결과를 메모리에 저장하는 메모이제이션 기법을 구현한 것이다.

## 1.7.4 파이썬에서 효과적인 함수형 프로그래밍

### a. 리스트 컴프리헨션과 제너레이터 표현식

파이썬은 함수형 스타일의 데이터 변환을 위한 간결한 구문을 제공한다:

```python
# 리스트 컴프리헨션 (map과 filter 대체)
numbers = [1, 2, 3, 4, 5]

# map 대체
squares = [x**2 for x in numbers]  # [1, 4, 9, 16, 25] - map과 동일한 결과

# filter 대체
evens = [x for x in numbers if x % 2 == 0]  # [2, 4] - filter와 동일한 결과

# map + filter 조합 대체
even_squares = [x**2 for x in numbers if x % 2 == 0]  # [4, 16] - 짝수만 추출하여 제곱

# 제너레이터 표현식 (지연 평가)
sum_squares = sum(x**2 for x in range(1000000))  # 메모리 효율적 - 값을 미리 계산하지 않음
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
print(f"c1: {c1}, c2: {c2}")  # c1: 5, c2: 8 - c1은 변경되지 않고 새 객체 c2가 생성됨
```

**중요한 점:** 함수형 프로그래밍과 객체지향 프로그래밍은 상호 배타적이지 않다. 파이썬에서는 두 패러다임의 장점을 결합하여 더 표현력 있고 유지보수하기 쉬운 코드를 작성할 수 있다.

함수형 프로그래밍 원칙을 따르더라도 파이썬의 멀티패러다임 특성을 활용하는 것이 최선의 접근법이다.

> [목차로 돌아가기](../../README.md) | [이전: 내장 전역 함수](./1_6_builtin_functions.md)
