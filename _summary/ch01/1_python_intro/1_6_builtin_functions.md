# 1.6 내장 전역 함수

> [목차로 돌아가기](../../README.md) | [이전: 타입 힌트 심화: 값과 타입의 경계](./1_5_type_hint_deep_dive.md) | [다음: 함수형 프로그래밍 요소](./1_7_functional_programming.md)

파이썬에는 언제 어디서든 사용할 수 있는 다양한 내장 전역 함수들이 있다. 이 함수들은 별도의 임포트 없이 바로 사용 가능하다.

## 왜 파이썬은 전역 함수를 사용하는가?

많은 객체지향 언어에서는 `length = array.length()`나 `size = list.size()`처럼 객체의 메서드로 기능을 제공하는 반면, 파이썬은 `len(list)`와 같이 전역 함수 방식을 많이 채택했다. 이는 단순한 선택이 아닌 파이썬의 철학과 설계 원칙에 기반한다:

### a. 일관성과 범용성

내장 타입뿐만 아니라 사용자 정의 타입에도 동일한 방식으로 적용될 수 있다. `len()`은 리스트, 문자열, 사전 등 모든 시퀀스와 컨테이너에 일관되게 작동하며, 사용자 정의 클래스도 `__len__()` 메서드만 구현하면 똑같이 작동한다.

```python
# 다양한 타입에 대해 동일하게 len() 적용
print(len("hello"))      # 문자열: 5
print(len([1, 2, 3]))    # 리스트: 3
print(len({"a": 1, "b": 2}))  # 딕셔너리: 2

# 사용자 정의 클래스
class MyCollection:
    def __init__(self, items):
        self._items = items
    def __len__(self):
        return len(self._items)

my_coll = MyCollection([1, 2, 3, 4])
print(len(my_coll))     # 4 - 내장 타입과 동일하게 작동
```

### b. 가독성과 표현력

파이썬의 창시자 귀도 반 로썸(Guido van Rossum)은 `len(x)`가 `x.len()`보다 더 명확하고 읽기 쉽다고 여겼다. 특히 코드를 읽을 때 주요 연산(길이 계산)이 먼저 오고 대상(객체)이 나중에 오는 것이 가독성에 도움이 된다.

### c. 파이썬의 "덕 타이핑" 철학

파이썬은 "오리처럼 꽥꽥거리고 오리처럼 걷는다면, 그것은 오리일 것이다"라는 덕 타이핑 접근법을 취한다. 전역 함수는 객체의 특정 인터페이스(`__len__`, `__str__` 등)를 호출함으로써 이런 철학을 구현한다.

### d. 객체지향과 함수형 패러다임의 균형

파이썬은 객체지향 언어이면서도 함수형 프로그래밍 요소를 통합했다. 전역 함수는 이러한 다중 패러다임 지원에 기여한다.

```python
# 함수형 프로그래밍 스타일
numbers = [1, 2, 3, 4, 5]
doubled = list(map(lambda x: x * 2, numbers))
filtered = list(filter(lambda x: x > 5, doubled))
```

귀도 반 로썸은 이렇게 말했다: "파이썬은 대부분의 것들이 객체인 객체지향 언어지만, 이는 모든 연산이 메서드여야 한다는 의미는 아닙니다. 일부 연산은 함수로서 더 자연스럽게 표현됩니다."

## 1.6.1 타입 및 객체 식별 함수

이 섹션에서는 객체의 타입과 식별자를 확인하는 함수들을 다룬다.

### a. `type()` - 객체의 타입 확인

`type()` 함수는 런타임에 객체의 실제 타입을 반환한다. 이는 정적 타입 검사와는 다른 개념이다.

정적 타입 힌트와 런타임 타입 검사의 차이에 대한 자세한 내용은 [타입 힌트와 실제 런타임 동작의 차이](./1_5_type_hint_deep_dive.md#155-타입-힌트와-실제-런타임-동작의-차이)를 참조하라.

```python
# 기본 사용
print(type(42))        # <class 'int'>
print(type("hello"))   # <class 'str'>
print(type([1, 2, 3])) # <class 'list'>

# 런타임 타입 비교
x = 42
print(type(x) is int)  # True (런타임에 x의 실제 타입이 int임을 확인)
print(type(x) == int)  # True (is와 동일한 결과)
```

### b. `isinstance()` - 객체의 타입 검사

`isinstance()` 함수는 객체가 특정 타입이나 그 타입의 서브클래스인지 확인한다. `type()` 함수보다 더 유연하고 객체 지향적인 타입 검사가 가능하다.

```python
# 기본 타입 검사
print(isinstance(42, int))         # True
print(isinstance("hello", str))    # True
print(isinstance("hello", int))    # False

# 여러 타입 중 하나인지 검사
print(isinstance(42, (int, float)))  # True
print(isinstance(3.14, (int, float)))  # True

# 상속 관계 검사
class Animal:
    pass

class Dog(Animal):
    pass

dog = Dog()
print(isinstance(dog, Dog))     # True
print(isinstance(dog, Animal))  # True - 상속 관계 인식
print(isinstance(Animal(), Dog))# False
```

`isinstance()`는 `type()` 함수와 달리 상속 관계를 고려하므로, 객체 지향 프로그래밍에서 더 적합한 타입 검사 방법이다.

### c. `issubclass()` - 클래스 상속 관계 확인

`issubclass()` 함수는 한 클래스가 다른 클래스의 서브클래스인지 확인한다. 두 클래스 간의 상속 관계를 검사하는 데 사용된다.

```python
# 기본 사용법
class Animal:
    pass

class Dog(Animal):
    pass

class Retriever(Dog):
    pass

print(issubclass(Dog, Animal))       # True (Dog는 Animal의 서브클래스)
print(issubclass(Retriever, Animal)) # True (간접 상속도 검사)
print(issubclass(Animal, Dog))       # False (Animal은 Dog의 서브클래스가 아님)
print(issubclass(Dog, Dog))          # True (모든 클래스는 자기 자신의 서브클래스로 간주)

# 여러 클래스 중 하나의 서브클래스인지 검사
print(issubclass(Dog, (list, dict, Animal)))  # True (두 번째 인자로 튜플 전달 가능)

# 사용자 정의 클래스와 내장 타입 검사
print(issubclass(bool, int))  # True (bool은 int의 서브클래스)
print(issubclass(list, object))  # True (모든 클래스는 object의 서브클래스)
```

`issubclass()`와 `isinstance()`의 차이점:

- `issubclass(A, B)`는 클래스 A가 클래스 B의 서브클래스인지 검사
- `isinstance(x, A)`는 객체 x가 클래스 A의 인스턴스인지 검사

### d. `id()` - 객체의 메모리 주소 확인

`id()` 함수는 객체의 고유 식별자(메모리 주소)를 반환한다. 이는 객체의 동일성을 확인할 때 유용하다.

```python
# 변수가 같은 객체를 참조하는지 확인
x = [1, 2, 3]
y = x          # y는 x와 동일한 객체를 참조
z = [1, 2, 3]  # z는 새로운 객체

print(id(x))  # 예: 140712834927872
print(id(y))  # x와 동일한 ID
print(id(z))  # 다른 ID

# is 연산자와의 관계
print(x is y)  # True (id(x) == id(y))
print(x is z)  # False (id(x) != id(z))

# 정수 캐싱 예시
a = 256
b = 256
print(a is b)  # True (작은 정수는 캐싱됨)

c = 257
d = 257
print(c is d)  # False (큰 정수는 새로운 객체 생성)
```

## 1.6.2 객체 속성 및 관리 함수

이 섹션에서는 객체의 구조와 속성을 검사하고 접근하는 함수들을 다룬다.

### a. `dir()` - 객체의 속성과 메서드 확인

`dir()` 함수를 이용해서 객체의 속성과 메서드를 알 수 있다.

```python
print(dir(42))        # int의 모든 메서드와 속성
print(dir("hello"))   # str의 모든 메서드와 속성
  
# 사용자 정의 클래스에도 적용 가능
class MyClass:
    def __init__(self):
        self.x = 10
        self.y = 20
      
    def my_method(self):
        pass
  
obj = MyClass()
print(dir(obj))  # 모든 속성과 메서드 목록 출력
```

### b. `len()` - 시퀀스 길이 확인

`len()` 함수를 이용해서 리스트나 문자열의 길이를 알 수 있다.

```python
print(len([1, 2, 3]))     # 3
print(len("hello"))       # 5
print(len({"a": 1, "b": 2}))  # 2
  
# 사용자 정의 객체도 __len__ 메서드를 구현하면 사용 가능
class MyClass:
    def __len__(self):
        return 42
  
obj = MyClass()
print(len(obj))  # 42
```

### c. `hasattr()` - 객체가 특정 속성을 가지고 있는지 확인

```python
class Person:
    name = "Kim"

p = Person()
print(hasattr(p, "name"))    # True
print(hasattr(p, "age"))     # False
```

### d. `getattr()` - 객체의 속성 값 가져오기

```python
class Person:
    name = "Kim"

p = Person()
print(getattr(p, "name"))             # "Kim"
print(getattr(p, "age", "Unknown"))   # "Unknown" (기본값 설정)
```

### e. `setattr()` - 객체의 속성 값 설정하기

```python
class Person:
    pass

p = Person()
setattr(p, "name", "Lee")
print(p.name)  # "Lee"
```

### f. `delattr()` - 객체의 속성 삭제하기

```python
class Person:
    name = "Kim"

p = Person()
delattr(p, "name")
print(hasattr(p, "name"))  # False
```

## 1.6.3 출력 및 문자열 표현 함수

이 섹션에서는 객체를 문자열로 표현하고 출력하는 함수들을 다룬다.

### a. `print()` - 표준 출력에 값 출력하기

`print()` 함수는 값을 표준 출력(보통 콘솔 화면)에 출력한다.

```python
# 기본 사용법
print("Hello, World!")  # Hello, World!

# 여러 인자 출력
print("이름:", "홍길동", "나이:", 30)  # 이름: 홍길동 나이: 30

# sep 인자로 구분자 변경
print("apple", "banana", "orange", sep=", ")  # apple, banana, orange

# end 인자로 끝 문자 변경
print("안녕하세요", end="! ")
print("반갑습니다")  # 안녕하세요! 반갑습니다

# file 인자로 출력 대상 변경
with open("output.txt", "w") as f:
    print("파일에 기록됨", file=f)
```

### b. `str()` - 사용자 친화적 문자열 변환

`str()` 함수는 객체를 사용자가 읽기 쉬운 문자열 형태로 변환한다. 이는 객체의 `__str__` 메서드를 호출하는 것과 같다.

```python
# 기본 사용
print(str(42))        # "42"
print(str(3.14159))   # "3.14159"
print(str([1, 2, 3])) # "[1, 2, 3]"

# 사용자 정의 클래스에서의 활용
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __str__(self):
        return f"{self.name}, {self.age}세"

person = Person("홍길동", 30)
print(str(person))  # "홍길동, 30세"
print(f"{person}")  # f-string에서 객체를 사용하면 str()이 자동 호출됨
```

### c. `repr()` - 개발자용 문자열 표현

`repr()` 함수는 객체를 명확하고 모호하지 않게 표현하는 문자열을 반환한다. 이는 객체의 `__repr__` 메서드를 호출하는 것과 같다. 이상적으로는 `eval(repr(obj)) == obj`가 성립해야 한다.

```python
# 기본 타입의 repr
print(repr("Hello"))     # "'Hello'" (따옴표 포함)
print(repr([1, 2, 3]))   # "[1, 2, 3]"
print(repr({"a": 1}))    # "{'a': 1}"

# repr과 str의 차이
text = "Hello\nWorld"
print(str(text))    # Hello
                    # World (줄바꿈 적용)
print(repr(text))   # 'Hello\nWorld' (이스케이프 시퀀스 표시)

# 사용자 정의 클래스에서 활용
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"
    
    def __str__(self):
        return f"({self.x}, {self.y})"

p = Point(3, 4)
print(str(p))   # "(3, 4)" - 사용자 친화적
print(repr(p))  # "Point(3, 4)" - 개발자용, 재생성 가능한 형태
```

### d. `format()` - 문자열 포맷팅

`format()` 메서드는 문자열 내의 중괄호(`{}`)로 표시된 필드를 지정된 값으로 대체한다.

```python
# 기본 사용법
print("이름: {}, 나이: {}".format("홍길동", 30))  # 이름: 홍길동, 나이: 30

# 인덱스 지정
print("{1}는 {0}이다".format("사과", "빨간색"))  # 빨간색는 사과이다

# 이름 지정
print("{name}은 {age}살이다".format(name="철수", age=20))  # 철수은 20살이다

# 포맷 지정
print("{:.2f}".format(3.14159))  # 3.14 (소수점 2자리까지)
print("{:,}".format(1000000))    # 1,000,000 (천 단위 구분 기호)
print("{:>10}".format("좌측"))    # "       좌측" (우측 정렬, 10자리)
print("{:<10}".format("우측"))    # "우측       " (좌측 정렬, 10자리)
print("{:^10}".format("중앙"))    # "    중앙    " (중앙 정렬, 10자리)

# f-string (Python 3.6+): format()의 더 간결한 대안
name = "홍길동"
age = 30
print(f"{name}은 {age}살이다")  # 홍길동은 30살이다
print(f"{3.14159:.2f}")        # 3.14
```

## 1.6.4 타입 변환 함수

파이썬은 기본 데이터 타입 간 변환을 위한 내장 함수를 제공한다.

### a. `int()` - 정수로 변환

```python
# 기본 사용법
print(int(3.14))     # 3 (소수점 이하 절삭)
print(int(-2.7))     # -2 (소수점 이하 절삭)
print(int("42"))     # 42 (문자열을 정수로 변환)

# 진법 지정 변환
print(int("101", 2))  # 5 (2진수 "101"을 10진수로 변환)
print(int("FF", 16))  # 255 (16진수 "FF"를 10진수로 변환)
print(int("0o77", 0)) # 63 (접두어로 진법 추론: "0o"는 8진수)
```

### b. `float()` - 부동소수점으로 변환

```python
# 기본 사용법
print(float(42))       # 42.0
print(float("-3.14"))  # -3.14
print(float("1.5e3"))  # 1500.0 (지수 표기법 지원)

# 특수 값
print(float("inf"))    # inf (무한대)
print(float("-inf"))   # -inf (음의 무한대)
print(float("nan"))    # nan (Not a Number)
```

### c. `complex()` - 복소수 생성 및 변환

```python
# 기본 사용법
print(complex(1, 2))     # (1+2j)
print(complex("3+4j"))   # (3+4j)
print(complex(5))        # (5+0j)
```

### d. `bool()` - 불린으로 변환

```python
# 기본 사용법
print(bool(0))      # False
print(bool(42))     # True (0이 아닌 숫자는 True)
print(bool(""))     # False (빈 문자열)
print(bool("text")) # True (비어있지 않은 문자열)
print(bool([]))     # False (빈 리스트)
print(bool([1, 2])) # True (비어있지 않은 리스트)

# 사용자 정의 객체의 참/거짓 판정
class CustomBool:
    def __init__(self, value):
        self.value = value
    
    def __bool__(self):
        return self.value > 0

print(bool(CustomBool(5)))  # True
print(bool(CustomBool(-1))) # False
```

### e. 컬렉션 변환 함수

컬렉션 타입 간의 변환을 수행하는 함수들:

```python
# list() 변환
print(list("hello"))         # ['h', 'e', 'l', 'l', 'o']
print(list({1, 2, 3}))       # [1, 2, 3]
print(list({"a": 1, "b": 2}))  # ['a', 'b']

# tuple() 변환
print(tuple([1, 2, 3]))      # (1, 2, 3)
print(tuple("abc"))          # ('a', 'b', 'c')

# set() 변환 - 중복 제거
print(set([1, 2, 2, 3]))     # {1, 2, 3}
print(set("hello"))          # {'h', 'e', 'l', 'o'}

# dict() 변환
pairs = [("a", 1), ("b", 2)]
print(dict(pairs))          # {'a': 1, 'b': 2}
print(dict(x=1, y=2))       # {'x': 1, 'y': 2}
```

## 1.6.5 이터레이션 및 시퀀스 함수

이 섹션에서는 이터레이션과 시퀀스 처리를 위한 함수들을 다룬다.

### a. `range()` - 수치 범위 생성

`range()` 함수는 수치 시퀀스를 생성하는 함수다:

```python
# 기본 사용법
for i in range(5):        # 0부터 4까지
    print(i, end=' ')     # 0 1 2 3 4

# 시작값과 끝값 지정
for i in range(2, 5):     # 2부터 4까지
    print(i, end=' ')     # 2 3 4

# 스텝 값 지정
for i in range(0, 10, 2): # 0부터 9까지 2씩 증가
    print(i, end=' ')     # 0 2 4 6 8
```

### b. `enumerate()` - 인덱스와 값의 쌍 생성

`enumerate()` 함수는 순회 가능한 객체의 각 요소에 인덱스를 부여하여 (인덱스, 요소) 형태의 튜플을 반환한다:

```python
# 기본 사용법
fruits = ["apple", "banana", "orange"]
for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")
# 출력:
# 0: apple
# 1: banana
# 2: orange

# 시작 인덱스 지정
for i, fruit in enumerate(fruits, start=1):
    print(f"{i}: {fruit}")
# 출력:
# 1: apple
# 2: banana
# 3: orange

# 리스트로 변환 
indexed_fruits = list(enumerate(fruits))
print(indexed_fruits)  # [(0, 'apple'), (1, 'banana'), (2, 'orange')]

# 딕셔너리 생성에 활용
fruit_dict = {i: name for i, name in enumerate(fruits)}
print(fruit_dict)  # {0: 'apple', 1: 'banana', 2: 'orange'}
```

### c. `zip()` - 여러 이터러블 병렬 순회

`zip()` 함수는 여러 이터러블의 요소들을 동시에 순회한다:

```python
names = ['Alice', 'Bob', 'Charlie']
ages = [24, 50, 18]

# 기본 사용법
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")

# dict() 생성에 활용
users = dict(zip(names, ages))
print(users)  # {'Alice': 24, 'Bob': 50, 'Charlie': 18}

# 길이가 다른 경우 (짧은 쪽 기준)
numbers = [1, 2]
letters = ['a', 'b', 'c']
print(list(zip(numbers, letters)))  # [(1, 'a'), (2, 'b')]
```

### d. `reversed()` - 시퀀스를 역순으로 반환

`reversed()` 함수는 주어진 시퀀스의 요소를 역순으로 반환하는 이터레이터를 생성한다:

```python
# 기본 사용법
numbers = [1, 2, 3, 4, 5]
rev_numbers = reversed(numbers)
print(list(rev_numbers))  # [5, 4, 3, 2, 1]

# 문자열 역순
text = "Python"
rev_text = reversed(text)
print(''.join(rev_text))  # "nohtyP"

# 범위 역순
rev_range = reversed(range(5))
print(list(rev_range))  # [4, 3, 2, 1, 0]

# 고급 활용: 회문(palindrome) 검사
def is_palindrome(s):
    # 문자열 정제 (소문자 변환 및 공백/특수문자 제거)
    s = ''.join(c.lower() for c in s if c.isalnum())
    return s == ''.join(reversed(s))

print(is_palindrome("A man, a plan, a canal: Panama"))  # True
print(is_palindrome("race a car"))  # False
```

## 1.6.6 함수형 프로그래밍 도구

이 섹션에서는 함수형 프로그래밍을 지원하는 내장 함수들을 다룬다. 이 함수들은 주로 람다 함수와 함께 사용된다. 람다 함수의 고급 활용에 대한 자세한 내용은 [1.7 함수형 프로그래밍 요소](./1_7_functional_programming.md#171-람다-함수lambda-functions)를 참조하라.

### a. `map()` - 요소별 함수 적용

`map()` 함수는 이터러블의 각 요소에 함수를 적용한 결과를 반환한다:

```python
numbers = [1, 2, 3, 4, 5]

# 제곱 계산
squares = map(lambda x: x**2, numbers)
print(list(squares))  # [1, 4, 9, 16, 25]

# 여러 인자 사용
a = [1, 2, 3]
b = [10, 20, 30]
sums = map(lambda x, y: x + y, a, b)
print(list(sums))  # [11, 22, 33]
```

### b. `filter()` - 조건에 맞는 요소 선택

`filter()` 함수는 이터러블에서 조건을 만족하는 요소만 선택한다:

```python
numbers = range(-5, 5)

# 양수만 선택
positives = filter(lambda x: x > 0, numbers)
print(list(positives))  # [1, 2, 3, 4]

# None을 필터 함수로 사용하면 False로 평가되는 요소 제거
mixed = [0, 1, False, True, '', 'hello', [], [1, 2]]
valid = filter(None, mixed)
print(list(valid))  # [1, True, 'hello', [1, 2]]
```

### c. `sorted()` - 정렬된 새 리스트 반환

`sorted()` 함수는 이터러블의 요소들을 정렬한 새 리스트를 반환한다:

```python
# 기본 정렬
print(sorted([3, 1, 4, 1, 5, 9, 2]))  # [1, 1, 2, 3, 4, 5, 9]

# key 함수로 정렬 기준 지정
words = ['banana', 'pie', 'Washington', 'book']
print(sorted(words, key=len))  # ['pie', 'book', 'banana', 'Washington']
print(sorted(words, key=str.lower))  # ['banana', 'book', 'pie', 'Washington']

# 역순 정렬
print(sorted([1, 2, 3], reverse=True))  # [3, 2, 1]

# 고급 활용: 객체 정렬
students = [
    {'name': 'Alice', 'grade': 85},
    {'name': 'Bob', 'grade': 92},
    {'name': 'Charlie', 'grade': 78}
]
top_students = sorted(students, key=lambda s: s['grade'], reverse=True)
for student in top_students:
    print(f"{student['name']}: {student['grade']}")
# 출력:
# Bob: 92
# Alice: 85
# Charlie: 78
```

### d. `all()` - 모든 요소가 참인지 확인

`all()` 함수는 이터러블의 모든 요소가 True로 평가될 때만 True를 반환한다:

```python
# 모든 요소가 참인 경우에만 True 반환
print(all([True, True, True]))  # True
print(all([True, False, True])) # False

# 비어있는 이터러블은 항상 True
print(all([]))  # True

# 숫자 리스트 예시
print(all([1, 2, 3]))     # True (모든 요소가 0이 아님)
print(all([1, 0, 3]))     # False (0은 거짓으로 평가됨)

# 조건식과 함께 활용
numbers = [2, 4, 6, 8, 10]
print(all(num % 2 == 0 for num in numbers))  # True (모두 짝수)

numbers = [2, 4, 5, 8, 10]
print(all(num % 2 == 0 for num in numbers))  # False (5는 짝수가 아님)
```

### e. `any()` - 하나라도 참인 요소가 있는지 확인

`any()` 함수는 이터러블에서 하나 이상의 요소가 True로 평가될 때 True를 반환한다:

```python
# 하나 이상의 요소가 참인 경우 True 반환
print(any([False, False, True]))  # True
print(any([False, False, False])) # False

# 비어있는 이터러블은 항상 False
print(any([]))  # False

# 숫자 리스트 예시
print(any([0, 0, 3]))     # True (3은 참으로 평가됨)
print(any([0, 0, 0]))     # False (모든 요소가 거짓)

# 조건식과 함께 활용
numbers = [1, 3, 5, 7, 8]
print(any(num % 2 == 0 for num in numbers))  # True (8은 짝수)

numbers = [1, 3, 5, 7, 9]
print(any(num % 2 == 0 for num in numbers))  # False (모두 홀수)
```

## 1.6.7 수학 및 집계 함수

수학 연산과 데이터 집계를 위한 내장 함수들이다.

### a. `sum()` - 시퀀스의 합계 계산

```python
# 기본 사용
numbers = [1, 2, 3, 4, 5]
print(sum(numbers))  # 15

# 시작값 지정
print(sum(numbers, 10))  # 25 (10 + 1 + 2 + 3 + 4 + 5)

# 제너레이터 표현식과 함께 사용
print(sum(x*x for x in range(5)))  # 0² + 1² + 2² + 3² + 4² = 30

# 주의: 문자열 연결에는 부적합
strings = ['a', 'b', 'c']
# print(sum(strings))  # TypeError: unsupported operand type(s)
print(''.join(strings))  # 'abc' (문자열 연결에는 join 사용)
```

### b. `min()`, `max()` - 최솟값/최댓값 찾기

```python
# 기본 사용
numbers = [5, 2, 9, -1, 7]
print(min(numbers))  # -1
print(max(numbers))  # 9

# 직접 인자로 전달
print(min(5, 2, 9, -1, 7))  # -1
print(max(5, 2, 9, -1, 7))  # 9

# key 함수를 사용한 사용자 정의 비교
words = ["apple", "banana", "pear", "watermelon"]
print(min(words, key=len))  # "pear" (가장 짧은 단어)
print(max(words, key=len))  # "watermelon" (가장 긴 단어)

# 빈 시퀀스와 기본값
empty = []
print(min(empty, default=0))  # 0 (빈 시퀀스일 때 기본값 반환)
print(max(empty, default=0))  # 0
```

### c. `abs()` - 절대값 계산

```python
# 숫자의 절대값
print(abs(-5))     # 5
print(abs(3.14))   # 3.14
print(abs(-3.14))  # 3.14

# 복소수의 절대값 (크기)
print(abs(3+4j))   # 5.0 (√(3² + 4²) = 5)
```

### d. `round()` - 반올림

```python
# 소수점 반올림 (기본은 정수로)
print(round(3.7))    # 4
print(round(3.2))    # 3
print(round(-3.7))   # -4

# 특정 소수점 자리까지 반올림
print(round(3.14159, 2))   # 3.14
print(round(3.14159, 3))   # 3.142

# 주의: 반올림 동작 (1.5 -> 2, 2.5 -> 2)
print(round(0.5))   # 0 (기대와 다를 수 있음)
print(round(1.5))   # 2
print(round(2.5))   # 2 (짝수로 반올림하는 Banker's rounding 적용)
print(round(3.5))   # 4
```

### e. `pow()` - 거듭제곱 계산

```python
# 기본 사용
print(pow(2, 3))    # 8 (2³ = 8)
print(pow(5, 2))    # 25 (5² = 25)

# 음수/분수 지수
print(pow(4, 0.5))  # 2.0 (4의 제곱근)
print(pow(2, -2))   # 0.25 (1/2² = 1/4 = 0.25)

# 세 번째 인자: 나머지 계산
print(pow(3, 4, 5))  # 1 (3⁴ % 5 = 81 % 5 = 1)
```

## 1.6.8 문자열 및 문자 변환 함수

문자와 문자열 처리를 위한 유틸리티 함수들이다.

### a. `chr()` 및 `ord()` - 문자와 코드포인트 변환

```python
# 정수 -> 문자 (유니코드 코드포인트에 해당하는 문자 반환)
print(chr(65))    # 'A'
print(chr(97))    # 'a'
print(chr(8364))  # '€'
print(chr(44032)) # '가'

# 문자 -> 정수 (문자의 유니코드 코드포인트 값 반환)
print(ord('A'))   # 65
print(ord('a'))   # 97
print(ord('€'))   # 8364
print(ord('가'))  # 44032
```

### b. `ascii()` - ASCII 표현 문자열 반환

```python
# 특수 문자나 비-ASCII 문자를 이스케이프 시퀀스로 변환
print(ascii("hello"))      # 'hello'
print(ascii("안녕하세요"))   # '\uc548\ub155\ud558\uc138\uc694'
print(ascii(['가', '나']))  # ["'\uc548'", "'\ub2c8'"]

# repr()과 유사하지만 비-ASCII 문자를 항상 이스케이프 처리
value = "€20"
print(repr(value))  # '€20'
print(ascii(value)) # '\u20ac20'
```

## 1.6.9 입출력 함수

사용자 상호작용과 기본 입출력 처리를 위한 함수들이다.

### a. `input()` - 사용자로부터 입력 받기

```python
# 기본 사용
name = input("이름을 입력하세요: ")
print(f"안녕하세요, {name}님!")

# 입력값 타입 변환
age = int(input("나이를 입력하세요: "))
print(f"내년에는 {age + 1}세가 되시겠네요.")

# 주의사항: 항상 문자열을 반환하므로 필요시 타입 변환 필요
height = input("키를 입력하세요(cm): ")
# print(height + 10)  # TypeError: can't add int to str
print(float(height) + 10)  # 올바른 사용법
```

### b. `open()` - 파일 열기

`open()` 함수는 파일을 열고 파일 객체를 반환한다. 파일을 열고 닫는 과정은 컨텍스트 매니저(`with` 구문)를 사용하는 것이 좋다.

```python
# 기본 사용법 (읽기 모드)
with open("example.txt", "r") as file:
    content = file.read()
    print(content)

# 쓰기 모드
with open("output.txt", "w") as file:
    file.write("Hello, World!\n")
    file.write("This is a test file.")

# 추가 모드
with open("log.txt", "a") as file:
    file.write("새 로그 항목\n")

# 이진 모드 (바이너리 파일)
with open("image.jpg", "rb") as file:
    binary_data = file.read()
```

### c. `print()` - 표준 출력에 값 출력하기

`print()` 함수의 고급 사용법:

```python
# 출력 대상 변경
import sys
print("표준 오류에 출력", file=sys.stderr)

# 임시 파일에 출력
import tempfile
with tempfile.TemporaryFile('w+') as f:
    print("임시 파일에 기록", file=f)
    f.seek(0)  # 파일 포인터 처음으로 이동
    print(f.read())  # 임시 파일에서 읽기

# flush 인자로 출력 버퍼 즉시 비우기
import time
for i in range(5):
    print(i, end=' ', flush=True)
    time.sleep(0.5)  # 출력 사이에 0.5초 지연
```

## 1.6.10 고급 함수 조합 패턴

파이썬 내장 함수들을 함께 조합하여 강력한 데이터 처리 패턴을 만들 수 있다.

### a. 내장 함수 조합 예시

```python
# map과 filter 조합
numbers = [1, 2, 3, 4, 5, 6]
result = list(map(lambda x: x**2, filter(lambda x: x % 2 == 0, numbers)))
print(result)  # [4, 16, 36] (짝수만 필터링하고 제곱)

# 리스트 컴프리헨션으로 동일한 작업
result = [x**2 for x in numbers if x % 2 == 0]
print(result)  # [4, 16, 36]

# max와 key 함수 활용
words = ["apple", "banana", "cherry", "date", "elderberry"]
longest = max(words, key=len)
print(longest)  # "elderberry"

# sorted와 lambda 함수 조합
students = [
    {"name": "Alice", "score": 85},
    {"name": "Bob", "score": 92},
    {"name": "Charlie", "score": 78}
]
top_students = sorted(students, key=lambda s: s["score"], reverse=True)
for student in top_students:
    print(f"{student['name']}: {student['score']}")
# 출력:
# Bob: 92
# Alice: 85
# Charlie: 78
```

### b. 함수형 스타일 데이터 처리

```python
# 명령형 스타일 vs 함수형 스타일
data = [1, 2, 3, 4, 5]

# 명령형 스타일
result = []
for x in data:
    if x % 2 == 0:
        result.append(x * 10)
print(result)  # [20, 40]

# 함수형 스타일
from functools import reduce

# 짝수만 필터링하고 각 요소에 10을 곱한 후 합계 계산
result = reduce(lambda a, b: a + b, 
               map(lambda x: x * 10, 
                  filter(lambda x: x % 2 == 0, data)))
print(result)  # 60 (20 + 40)

# 파이프라인 스타일로 작성하면 가독성 향상
def pipeline(data, *funcs):
    result = data
    for func in funcs:
        result = func(result)
    return result

result = pipeline(
    data,
    lambda d: filter(lambda x: x % 2 == 0, d),
    lambda d: map(lambda x: x * 10, d),
    lambda d: reduce(lambda a, b: a + b, d)
)
print(result)  # 60
```

### c. 제너레이터와 함수 조합

내장 함수들을 제너레이터와 함께 사용하면 메모리 효율적인 데이터 처리 파이프라인을 구축할 수 있다:

```python
# 큰 데이터셋을 효율적으로 처리하는 예시
def process_large_dataset(data_source, batch_size=1000):
    # 데이터를 배치 단위로 처리
    def batched(iterable, n):
        batch = []
        for item in iterable:
            batch.append(item)
            if len(batch) == n:
                yield batch
                batch = []
        if batch:  # 남은 항목 처리
            yield batch
    
    # 여러 변환 함수들
    def parse(batch):
        return [int(x) for x in batch if x.strip().isdigit()]
        
    def filter_valid(batch):
        return filter(lambda x: 0 < x < 100, batch)
        
    def transform(batch):
        return map(lambda x: x * 2, batch)
    
    # 파이프라인 구성
    for raw_batch in batched(data_source, batch_size):
        parsed = parse(raw_batch)
        filtered = filter_valid(parsed)
        transformed = transform(filtered)
        yield from transformed

# 사용 예시 (큰 파일 데이터 처리)
def read_lines(filename):
    with open(filename, 'r') as f:
        for line in f:
            yield line

# 실제 사용 시에는 아래와 같이 활용 가능
# for result in process_large_dataset(read_lines('large_data.txt')):
#     process_result(result)
```

### d. 복합 함수 생성 패턴

내장 함수들을 활용해 새로운 고차 함수를 생성하는 패턴:

```python
def compose(*functions):
    """
    여러 함수를 합성하는 고차 함수
    오른쪽에서 왼쪽으로 함수들이 적용됨
    """
    if not functions:
        return lambda x: x  # 항등 함수
    
    def composed_function(x):
        result = x
        for f in reversed(functions):
            result = f(result)
        return result
    
    return composed_function

# 사용 예시
def add_one(x): return x + 1
def double(x): return x * 2
def square(x): return x ** 2

# f(x) = (x + 1)² * 2
pipeline = compose(double, square, add_one)
print(pipeline(3))  # ((3 + 1)² * 2) = 32
```

## 1.6.11 기타 유용한 내장 함수

이 섹션에서는 특정 카테고리에 명확히 속하지 않지만 유용한 내장 함수들을 다룬다.

### a. `eval()` 및 `exec()` - 문자열로 된 코드 실행

```python
# eval() - 표현식 평가 후 결과 반환
x = 10
result = eval('x * 5 + 3')
print(result)  # 53

# 딕셔너리로 네임스페이스 제공
scope = {'x': 5, 'y': 20}
print(eval('x + y', scope))  # 25

# exec() - 문장 실행 (반환값 없음)
exec('a = 5; b = 7; print(a + b)')  # 12

local_vars = {}
exec('result = [i**2 for i in range(5)]', {}, local_vars)
print(local_vars['result'])  # [0, 1, 4, 9, 16]

# 주의: 보안 위험이 있으므로 신뢰할 수 없는 입력에는 사용하지 말 것
```

### b. `globals()` 및 `locals()` - 심볼 테이블 접근

```python
# globals() - 전역 심볼 테이블 반환
global_var = "전역 변수"

def test_globals():
    print(globals()['global_var'])  # "전역 변수"
    globals()['new_global'] = "새 전역 변수"

test_globals()
print(new_global)  # "새 전역 변수"

# locals() - 현재 로컬 심볼 테이블 반환
def test_locals(arg):
    x = 10
    y = 'hello'
    print(locals())  # {'arg': 값, 'x': 10, 'y': 'hello'}

test_locals(5)
```

### c. `hash()` - 객체의 해시값 계산

```python
# 불변 객체의 해시값
print(hash("hello"))     # 문자열의 해시값
print(hash((1, 2, 3)))   # 튜플의 해시값

# 동일한 값은 동일한 해시 반환
s1 = "python"
s2 = "python"
print(hash(s1) == hash(s2))  # True

# 가변 객체는 해시 불가
try:
    hash([1, 2, 3])  # TypeError: unhashable type: 'list'
except TypeError as e:
    print(e)
```

### d. `callable()` - 호출 가능 여부 확인

```python
# 함수와 메서드는 호출 가능
def my_func():
    pass

print(callable(my_func))  # True
print(callable(len))      # True

# 클래스는 호출 가능 (__call__ 메서드로 인스턴스화)
print(callable(dict))     # True

# 일반 객체는 호출 불가능
print(callable("string")) # False

# __call__ 메서드 구현 시 호출 가능
class Callable:
    def __call__(self):
        return "Called!"

obj = Callable()
print(callable(obj))      # True
print(obj())              # "Called!"
```

### e. `vars()` - 객체의 __dict__ 속성 반환

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

p = Person("Alice", 30)
print(vars(p))  # {'name': 'Alice', 'age': 30}

# 인자 없으면 locals()와 동일
def test():
    x = 10
    y = 20
    print(vars())  # locals()와 동일
```

### f. `iter()` 및 `next()` - 이터레이터 다루기

```python
# 이터러블에서 이터레이터 얻기
my_list = [1, 2, 3]
iterator = iter(my_list)

# 이터레이터에서 값 하나씩 가져오기
print(next(iterator))  # 1
print(next(iterator))  # 2
print(next(iterator))  # 3
# print(next(iterator))  # StopIteration 예외 발생

# 기본값과 함께 사용
iterator = iter([])
print(next(iterator, "기본값"))  # "기본값"

# 이터레이터 프로토콜 구현한 커스텀 클래스
class CountDown:
    def __init__(self, start):
        self.count = start
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.count <= 0:
            raise StopIteration
        self.count -= 1
        return self.count + 1

# 사용 예시
for i in CountDown(5):
    print(i, end=' ')  # 5 4 3 2 1
```

이처럼 파이썬의 내장 함수들은 단독으로 사용할 때도 유용하지만, 함께 조합하여 더 강력하고 표현력 있는 코드를 작성할 수 있다. 더 복잡한 함수형 프로그래밍 패턴에 대해서는 [1.7 함수형 프로그래밍 요소](./1_7_functional_programming.md)를 참조하라.

> [목차로 돌아가기](../../README.md) | [이전: 타입 힌트 심화: 값과 타입의 경계](./1_5_type_hint_deep_dive.md) | [다음: 함수형 프로그래밍 요소](./1_7_functional_programming.md)
