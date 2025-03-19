# 1.3 객체 시스템과 객체 지향 프로그래밍 지원

> [목차로 돌아가기](../../README.md) | [이전: 타입 시스템과 타입 힌트](./1_2_type_system.md) | [다음: 기본 데이터 타입](./1_4_basic_data_types.md)

## 1.3.1 파이썬의 객체 시스템

### a. 파이썬에서 모든 표현 대상은 '객체'이다

'객체'는 '메모리에 저장된 데이터'와 '데이터를 처리하는 함수'를 가지고 있다. '객체'는 '변수'에 할당할 수 있다. '변수'는 '객체'를 가리키는 '레퍼런스'를 가지고 있다.

`class`로 명시적으로 표현되는 타입뿐만 아니라 `int`, `float`, `list`, `dict` 등의 내장 타입도 '객체'이다. 다음과 같은 코드는 모두 '객체'를 생성한다:

```python
a = 10
b = 2.5
c = [1, 2, 3]
d = {"apple": 100, "banana": 100}
```

단순 숫자값도 객체이기 때문에 다음과 같이 메소드를 호출할 수 있다:

```python
print((10).bit_length())    # 출력: 4
```

### b. 파이썬에서 모든 표현 대상은 `object` 클래스를 상속받은 '객체'이다

파이썬에서는 모든 타입이 `object` 클래스를 암묵적으로 상속받는다. 이는 모든 객체가 공통으로 가지는 기본 메서드와 속성이 있다는 것을 의미한다:

```python
# 모든 타입은 object의 자식이다
print(isinstance(42, object))          # True
print(isinstance("hello", object))     # True
print(isinstance([1, 2, 3], object))   # True
print(isinstance(len, object))         # True (함수도 객체다)
print(isinstance(type, object))        # True (타입도 객체다)

# object 클래스에서 상속받은 공통 메서드들
num = 42
print(dir(num))  # 모든 속성과 메서드 목록 출력
print(num.__class__)  # <class 'int'>
print(num.__str__())  # "42"
print(num.__repr__()) # "42"

# 사용자 정의 클래스도 자동으로 object를 상속
class MyClass:  # 암묵적으로 object 상속
    pass
    
class MyExplicitClass(object):  # 명시적으로 object 상속 (동일한 의미)
    pass

obj = MyClass()
print(isinstance(obj, object))  # True
print(obj.__class__.__bases__) # (<class 'object'>,)
```

모든 객체가 공통적으로 갖는 주요 메서드들:

* `__str__()`: 문자열 표현 (`str()` 함수가 호출)
* `__repr__()`: 개발자를 위한 상세 문자열 표현
* `__class__`: 객체의 타입 정보
* `__doc__`: 문서화 문자열
* `__dict__`: 객체의 속성 목록 (네임스페이스)
* `__hash__()`: 해시 값 계산 (딕셔너리 키로 사용 가능한지 결정)
* `__eq__()`: 동등성 비교 (`==` 연산자)

### c. 클래스와 인스턴스의 기본 개념

클래스는 객체의 설계도이고, 인스턴스는 그 설계도로 만들어진 실체다:

```python
# 클래스 정의
class Person:
    # 클래스 변수 - 모든 인스턴스가 공유
    species = "Homo sapiens"
    
    # 초기화 메서드 (생성자)
    def __init__(self, name, age):
        # 인스턴스 변수 - 각 인스턴스마다 고유
        self.name = name
        self.age = age
        
    # 인스턴스 메서드
    def say_hello(self):
        return f"안녕하세요, 제 이름은 {self.name}입니다. 저는 {self.age}세입니다."
    
    # 클래스 메서드
    @classmethod
    def from_birth_year(cls, name, year):
        # 현재 연도 기준으로 나이 계산 (간략화)
        return cls(name, 2023 - year)
        
    # 정적 메서드
    @staticmethod
    def is_adult(age):
        return age >= 19  # 한국 기준

# 인스턴스 생성
person1 = Person("김철수", 30)
person2 = Person("이영희", 25)

# 인스턴스 메서드 호출
print(person1.say_hello())  # 안녕하세요, 제 이름은 김철수입니다. 저는 30세입니다.

# 클래스 메서드 호출 - 출생연도로 인스턴스 생성
person3 = Person.from_birth_year("박지민", 1995)
print(person3.age)  # 28 (2023-1995)

# 정적 메서드 호출
print(Person.is_adult(17))  # False
print(Person.is_adult(20))  # True
```

## 1.3.2 변수와 객체 참조

### a. 파이썬의 변수는 객체의 참조이다

C/C++과 달리 파이썬의 변수는 메모리 주소가 아닌 객체에 대한 이름(레이블)이다:

```python
# C/C++에서:
# int a = 5;  // 변수 a는 정수 값 5를 저장하는 메모리 위치

# 파이썬에서:
a = 5  # 변수 a는 정수 객체 5를 참조
```

주요 차이점:

1. __변수 선언 없음__: 파이썬은 변수 선언 없이 바로 할당
2. __타입 명시 없음__: 참조하는 객체에 따라 타입이 결정됨
3. __메모리 접근 제한__: 저수준 메모리 직접 조작 불가 (C처럼 포인터 연산 없음)
4. __자동 메모리 관리__: 참조되지 않는 객체는 자동 정리

```python
# 객체 참조 변경 예시
a = [1, 2, 3]  # a는 리스트 객체를 참조
b = a          # b도 동일한 리스트를 참조

b.append(4)    # b를 통해 리스트 수정
print(a)       # [1, 2, 3, 4] - a도 변경됨 (같은 객체)

b = [5, 6]     # b는 새 리스트를 참조
print(a)       # [1, 2, 3, 4] - a는 변경되지 않음
```

### b. 객체에는 가변 객체와 불변 객체가 있다

파이썬의 객체는 생성 후 수정 가능 여부에 따라 두 그룹으로 나뉜다:

1. __불변 객체 (Immutable)__: 생성 후 내용을 변경할 수 없음
   * 숫자 (`int`, `float`, `complex`)
   * 문자열 (`str`)
   * 튜플 (`tuple`)
   * 불변 집합 (`frozenset`)
   * 바이트 (`bytes`)

2. __가변 객체 (Mutable)__: 생성 후 내용을 변경할 수 있음
   * 리스트 (`list`)
   * 딕셔너리 (`dict`)
   * 집합 (`set`)
   * 바이트 배열 (`bytearray`)
   * 사용자 정의 클래스 객체 (대부분)

```python
# 불변 객체 예시 - 문자열
s1 = "hello"
s2 = s1

# 새 문자열 생성 (기존 문자열 수정 불가능)
s1 = s1 + " world"

print(s1)  # "hello world"
print(s2)  # "hello" (원본 유지)

# 가변 객체 예시 - 리스트
l1 = [1, 2, 3]
l2 = l1

# 같은 객체 수정
l1.append(4)

print(l1)  # [1, 2, 3, 4]
print(l2)  # [1, 2, 3, 4] (l1과 같은 객체라 함께 변경)
```

가변성은 파이썬의 코딩 스타일에 큰 영향을 미친다:

* 불변 객체는 안전하게 공유할 수 있음 (예: 함수 기본값으로)
* 가변 객체는 참조를 공유할 때 주의해야 함
* 불변 객체는 해시 가능하므로 딕셔너리 키나 세트 요소로 사용 가능

### c. 함수 인자는 객체 참조로 전달된다 (Call by Object Reference)

파이썬의 함수 인자 전달 방식은 종종 "pass by assignment" 또는 "call by object reference"라고 불린다:

1. __가변 객체와 불변 객체의 차이__:

    ```python
    # 불변 객체 전달 (정수, 문자열, 튜플 등)
    def modify_value(x):
        x = x + 1  # 새 객체를 생성해 x에 할당 (원본 변경 안됨)
        print(f"함수 내부: {x}")

    num = 10
    modify_value(num)  # "함수 내부: 11" 출력
    print(num)         # 10 (변경되지 않음)

    # 가변 객체 전달 (리스트, 딕셔너리, 셋 등)
    def modify_list(lst):
        lst.append(4)   # 참조된 객체 자체를 수정 (원본 변경됨)
        print(f"함수 내부: {lst}")

    my_list = [1, 2, 3]
    modify_list(my_list)  # "함수 내부: [1, 2, 3, 4]" 출력
    print(my_list)        # [1, 2, 3, 4] (변경됨)
    ```

2. __객체 재할당 VS 객체 수정__:

```python
def reassign_vs_modify(lst, num):
    lst.append(100)   # 원래 객체 수정 (호출자에게 영향)
    lst = [200, 300]  # 새 객체 할당 (함수 내부만 영향)
    num += 50         # 새 정수 객체 생성 (호출자에게 영향 없음)

data = [1, 2, 3]
value = 10
reassign_vs_modify(data, value)

print(data)   # [1, 2, 3, 100] (append의 영향만 받음)
print(value)  # 10 (변경 안됨)
```

## 1.3.3 객체의 메모리 관리

### a. 가비지 컬렉션과 참조 카운팅으로 메모리를 관리한다

파이썬의 메모리 관리는 두 가지 메커니즘으로 이루어진다:

1. __참조 카운팅__ - 기본 메모리 관리 방식:

    ```python
    import sys

    # 객체의 참조 수 확인
    x = [1, 2, 3]
    print(sys.getrefcount(x) - 1)  # 1 (함수 호출 시 임시 참조 제외)

    y = x  # x가 참조하는 객체를 y도 참조
    print(sys.getrefcount(x) - 1)  # 2

    del y  # y 삭제로 참조 카운트 감소
    print(sys.getrefcount(x) - 1)  # 1
    ```

    `del` 키워드는 변수를 삭제하여 참조 카운트를 감소시킨다. 참조 카운트가 0이 되면 객체는 즉시 메모리에서 해제된다.

    ```python
    # 변수 명시적 삭제
    z = [1, 2, 3]
    del z  # z 변수 자체를 삭제, 참조 카운트가 0이 되어 객체도 해제
    # print(z)  # NameError: name 'z' is not defined

    # 객체의 일부 삭제
    d = {"a": 1, "b": 2}
    del d["a"]  # 딕셔너리에서 키 'a' 제거
    print(d)  # {'b': 2}
    ```

2. __가비지 컬렉터__ - 순환 참조 처리:

    ```python
    # 순환 참조 예시
    def create_cycle():
        lst = []
        lst.append(lst)  # 자기 자신을 참조 (순환 참조)
        return lst

    cycle = create_cycle()
    # 참조 카운트는 1 이상이지만 외부에서 접근 불가능해지면
    # 주기적인 가비지 컬렉션에 의해 정리됨
    del cycle

    # 명시적 가비지 컬렉션 호출 (보통은 불필요)
    import gc
    gc.collect()  # 순환 참조 객체들을 검출하고 해제
    ```

파이썬 메모리 관리의 특징:

* 대부분의 객체는 즉시 참조 카운팅으로 해제됨
* 순환 참조는 가비지 컬렉터가 주기적으로 검출하여 해제
* `del` 키워드는 변수를 삭제하며, 이로 인해 참조 카운트가 0이 되면 객체도 해제됨
* 메모리 관리는 자동화되어 있어 개발자가 직접 메모리 할당/해제를 고려할 필요가 적음

### b. 약한 참조와 리소스 관리 기법

메모리 누수를 방지하고 시스템 리소스를 효율적으로 관리하기 위한 고급 기법:

```python
import weakref

class Resource:
    def __init__(self, name):
        self.name = name
        print(f"{name} 리소스 생성됨")
        
    def __del__(self):
        print(f"{self.name} 리소스 해제됨")
        
obj = Resource("첫 번째")

# 약한 참조 생성
weak_ref = weakref.ref(obj)
print(weak_ref() is obj)  # True - 같은 객체 참조

# 강한 참조 제거 시 객체 자동 해제
obj = None  # 유일한 강한 참조 제거
print(weak_ref())  # None - 약한 참조는 남아있지만 객체는 해제됨
```

## 1.3.4 객체 지향 프로그래밍 핵심 원칙

### a. 상속과 다형성

상속은 기존 클래스의 기능을 확장하고, 다형성은 같은 인터페이스로 다른 구현을 제공한다:

```python
# 기본 클래스 (부모 클래스)
class Animal:
    def __init__(self, name):
        self.name = name
        
    def speak(self):
        # 추상 메서드처럼 사용 (자식 클래스에서 구현해야 함)
        raise NotImplementedError("자식 클래스에서 이 메서드를 구현해야 합니다")

# 파생 클래스 (자식 클래스)
class Dog(Animal):
    def speak(self):
        return f"{self.name}이(가) 멍멍!"
        
class Cat(Animal):
    def speak(self):
        return f"{self.name}이(가) 야옹!"
        
# 다형성 예시
def make_speak(animal):
    # animal이 어떤 자식 클래스인지에 상관없이 동일한 인터페이스 사용
    return animal.speak()

dog = Dog("바둑이")
cat = Cat("나비")

print(make_speak(dog))  # 바둑이이(가) 멍멍!
print(make_speak(cat))  # 나비이(가) 야옹!

# isinstance로 상속 관계 확인
print(isinstance(dog, Dog))    # True
print(isinstance(dog, Animal)) # True
print(isinstance(dog, Cat))    # False
```

### b. 캡슐화와 접근 제어

파이썬은 명시적인 접근 제한자가 없지만 관례를 통해 캡슐화를 구현한다:

```python
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner        # 공개 속성
        self._balance = balance    # 보호 속성 (관례상 외부 접근 자제)
        self.__account_number = self.__generate_account_number()  # 비공개 속성
        
    def __generate_account_number(self):  # 비공개 메서드
        # 실제로는 더 복잡한 로직이 들어갈 것
        import random
        return random.randint(10000000, 99999999)
    
    def deposit(self, amount):
        if amount > 0:
            self._balance += amount
            return True
        return False
            
    def withdraw(self, amount):
        if 0 < amount <= self._balance:
            self._balance -= amount
            return True
        return False
            
    def get_balance(self):
        return self._balance
        
    def get_account_info(self):
        # 마스킹된 계좌번호만 반환
        masked = "XXXX-XX" + str(self.__account_number)[-2:]
        return f"소유자: {self.owner}, 계좌번호: {masked}, 잔액: {self._balance}원"

# 사용 예시
account = BankAccount("홍길동", 10000)

# 공개 메서드 사용
account.deposit(5000)
print(account.get_balance())  # 15000

# 보호 속성 접근 (가능하지만 권장하지 않음)
print(account._balance)  # 15000

# 비공개 속성 접근 시도
try:
    print(account.__account_number)  # AttributeError 발생
except AttributeError as e:
    print("비공개 속성에 접근할 수 없습니다")
    
# 이름 맹글링으로 실제 비공개 속성 접근 (권장하지 않음)
print(account._BankAccount__account_number)  # 실제 계좌번호 출력

# 안전한 방법으로 정보 확인
print(account.get_account_info())  # 마스킹된 정보만 제공
```

### c. 특수 메서드와 연산자 오버로딩

파이썬에서는 특수 메서드(던더/매직 메서드)를 정의하여 연산자의 동작을 커스터마이즈할 수 있다:

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    # 문자열 표현 (str() 함수나 print() 사용 시)
    def __str__(self):
        return f"Vector({self.x}, {self.y})"
        
    # 개발자용 표현 (repr() 함수 사용 시)
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
    
    # + 연산자 오버로딩
    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        return NotImplemented
    
    # * 연산자 오버로딩 (벡터와 스칼라 곱)
    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Vector(self.x * scalar, self.y * scalar)
        return NotImplemented
    
    # 오른쪽 곱셈 (스칼라 * 벡터)
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    # == 연산자 오버로딩
    def __eq__(self, other):
        if not isinstance(other, Vector):
            return False
        return self.x == other.x and self.y == other.y
    
    # 길이 계산 (len() 함수 사용 시)
    def __abs__(self):
        return (self.x**2 + self.y**2) ** 0.5
    
    # 컨텍스트 관리자 프로토콜 구현
    def __enter__(self):
        print("벡터 컨텍스트 시작")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("벡터 컨텍스트 종료")
        
    # 이터레이터 프로토콜 구현
    def __iter__(self):
        yield self.x
        yield self.y

# 사용 예시
v1 = Vector(3, 4)
v2 = Vector(2, 3)

# 연산자 사용
v3 = v1 + v2
print(v3)  # Vector(5, 7)

v4 = v1 * 2
print(v4)  # Vector(6, 8)

# 스칼라 곱 (오른쪽 피연산자)
v5 = 3 * v1
print(v5)  # Vector(9, 12)

# 비교 연산
print(v1 == Vector(3, 4))  # True
print(v1 == v2)           # False

# 길이 계산
print(abs(v1))  # 5.0 (벡터의 크기)

# 컨텍스트 관리자로 사용
with Vector(1, 1) as v:
    print(f"컨텍스트 내부: {v}")

# 이터레이터로 사용
for coord in v1:
    print(coord)  # 3, 4 출력
```

### d. 다중 상속과 MRO(Method Resolution Order)

파이썬은 다중 상속을 지원하며, MRO(Method Resolution Order)는 메서드 해석 순서를 결정하는 알고리즘이다:

```python
# 다중 상속 예시
class A:
    def greet(self):
        return "A의 인사"

class B(A):
    def greet(self):
        return "B의 인사"

class C(A):
    def greet(self):
        return "C의 인사"

# 다이아몬드 상속 구조
class D(B, C):
    pass

# MRO 확인
print(D.__mro__)  
# (<class '__main__.D'>, <class '__main__.B'>, <class '__main__.C'>, <class '__main__.A'>, <class 'object'>)

d = D()
print(d.greet())  # "B의 인사" (MRO에 따라 B의 메서드가 우선)
```

#### MRO의 주요 특징

1. __C3 선형화 알고리즘__ - 파이썬 2.3부터 도입된 일관된 메서드 해석 순서 결정 방식
2. __왼쪽에서 오른쪽으로의 우선순위__ - 상속 목록에서 왼쪽에 있는 클래스가 우선
3. __깊이 우선 탐색__ - 하위 클래스가 모든 상위 클래스보다 우선
4. __단일 경로__ - 클래스는 MRO에서 정확히 한 번만 나타남

MRO를 확인하는 방법:

```python
# 클래스의 MRO 확인
print(ClassName.__mro__)  # 튜플 형태로 반환
print(ClassName.mro())    # 리스트 형태로 반환
```

#### 다중 상속 설계 지침

1. __믹스인 패턴__ - 단일 기능을 제공하는 클래스를 조합하여 사용
2. __인터페이스 일관성__ - 상속받는 모든 클래스에서 일관된 메서드 시그니처 유지
3. __상속 구조 단순화__ - 과도하게 복잡한 상속 구조 지양

```python
# 믹스인 패턴 예시
class SerializableMixin:
    def to_json(self):
        import json
        return json.dumps(self.__dict__)

class LoggableMixin:
    def log(self, message):
        print(f"로그: {message}")

class User(SerializableMixin, LoggableMixin):
    def __init__(self, name, email):
        self.name = name
        self.email = email

# 각 믹스인의 기능을 모두 사용 가능
user = User("홍길동", "hong@example.com")
json_data = user.to_json()
user.log("사용자 객체 생성됨")
```

### e. super() 키워드 사용법

`super()` 키워드는 부모 클래스의 메서드를 호출하기 위해 사용되며, 올바른 MRO를 따라 메서드를 찾는다:

```python
class Parent:
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        return f"안녕하세요, {self.name}입니다!"

class Child(Parent):
    def __init__(self, name, age):
        # 부모 클래스의 __init__ 호출
        super().__init__(name)
        self.age = age
    
    def greet(self):
        # 부모 클래스의 greet 메서드를 확장
        parent_greeting = super().greet()
        return f"{parent_greeting} 저는 {self.age}살입니다."

# 사용 예시
child = Child("김철수", 10)
print(child.greet())  # "안녕하세요, 김철수입니다! 저는 10살입니다."
```

#### super()의 주요 특징

1. __매개변수 없는 호출__ - `super()` 만으로 현재 클래스와 인스턴스의 정보를 자동 전달:

   ```python
   super().__init__()  # Python 3 방식
   ```

2. __명시적 매개변수 전달__ - 클래스와 인스턴스를 명시적으로 지정:

   ```python
   super(Child, self).__init__()  # 전통적인 방식
   ```

3. __다중 상속에서의 활용__ - MRO에 따라 다음 클래스의 메서드 호출:

   ```python
   class A:
       def method(self):
           print("A의 메서드")

   class B(A):
       def method(self):
           print("B의 메서드")
           super().method()  # A.method() 호출

   class C(A):
       def method(self):
           print("C의 메서드")
           super().method()  # A.method() 호출

   class D(B, C):
       def method(self):
           print("D의 메서드")
           super().method()  # MRO에 따라 B.method() 호출

   # MRO: D -> B -> C -> A -> object
   D().method()
   # 출력:
   # D의 메서드
   # B의 메서드
   # C의 메서드 
   # A의 메서드
   ```

#### 협동적 다중 상속(Cooperative Multiple Inheritance)

복잡한 다중 상속 구조에서는 각 클래스가 `super()`를 사용해 메서드 체인을 유지하는 '협동적 상속' 패턴이 권장된다:

```python
class Base:
    def __init__(self):
        print("Base 초기화")

class A(Base):
    def __init__(self):
        print("A 초기화 시작")
        super().__init__()
        print("A 초기화 완료")

class B(Base):
    def __init__(self):
        print("B 초기화 시작")
        super().__init__()
        print("B 초기화 완료")

class C(A, B):
    def __init__(self):
        print("C 초기화 시작")
        super().__init__()  # MRO에 따라 A.__init__ 호출
        print("C 초기화 완료")

# 실행 결과
c = C()
# C 초기화 시작
# A 초기화 시작
# B 초기화 시작
# Base 초기화
# B 초기화 완료
# A 초기화 완료
# C 초기화 완료

# MRO 확인
print(C.__mro__)
# (<class '__main__.C'>, <class '__main__.A'>, <class '__main__.B'>, <class '__main__.Base'>, <class 'object'>)
```

이와 같은 협동적 다중 상속 패턴은 믹스인 클래스를 활용할 때 특히 중요하며, 각 믹스인이 `super()`를 통해 메서드 체인을 유지함으로써 모든 기능이 적절히 초기화되고 실행될 수 있게 한다.

## 1.3.5 파이썬의 기타 객체 지향 기능들

### a. 자주 사용되는 특수 메서드들

파이썬의 객체 지향 프로그래밍 시스템은 특수 메서드(던더 메서드, dunder methods)를 통해 다양한 기능을 제공합니다. 이러한 메서드들은 이름이 이중 밑줄(`__`)로 둘러싸여 있으며, 특정 상황에서 자동으로 호출됩니다:

```python
# 자주 사용되는 특수 메서드 요약
class MyClass:
    def __init__(self, value):
        """객체 초기화 (생성자)"""
        self.value = value
    
    def __str__(self):
        """문자열 표현 (str() 함수나 print() 사용 시)"""
        return f"MyClass 객체: {self.value}"
    
    def __repr__(self):
        """개발자용 표현 (repr() 함수 사용 시, 디버깅용)"""
        return f"MyClass({self.value!r})"
    
    def __eq__(self, other):
        """== 연산자 동작 정의"""
        if not isinstance(other, MyClass):
            return NotImplemented
        return self.value == other.value
    
    def __lt__(self, other):
        """< 연산자 동작 정의 (정렬 시 활용)"""
        if not isinstance(other, MyClass):
            return NotImplemented
        return self.value < other.value
    
    def __len__(self):
        """len() 함수 동작 정의"""
        return len(str(self.value))
    
    def __getitem__(self, key):
        """인덱싱 동작 정의 (obj[key])"""
        if key == 0:
            return self.value
        raise IndexError("인덱스 범위 초과")
    
    def __call__(self, *args, **kwargs):
        """함수처럼 호출 가능하게 함 (obj())"""
        return f"호출됨: {self.value}, 인자: {args}, 키워드: {kwargs}"
    
    def __enter__(self):
        """컨텍스트 관리자 진입 (with 문)"""
        print("컨텍스트 시작")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 관리자 종료 (with 문)"""
        print("컨텍스트 종료")
        return False  # 예외를 다시 발생시킴
    
    def __add__(self, other):
        """+ 연산자 동작 정의"""
        if isinstance(other, MyClass):
            return MyClass(self.value + other.value)
        elif isinstance(other, (int, float, str)):
            return MyClass(self.value + other)
        return NotImplemented
    
    def __radd__(self, other):
        """오른쪽 피연산자에 대한 + 연산자 정의 (다른 타입 + self)"""
        if isinstance(other, (int, float, str)):
            return MyClass(other + self.value)
        return NotImplemented

# 특수 메서드 사용 예시
obj = MyClass(42)
print(str(obj))        # MyClass 객체: 42
print(repr(obj))       # MyClass(42)
print(obj == MyClass(42))  # True
print(len(obj))        # 2 (문자열 "42"의 길이)
print(obj[0])          # 42
print(obj(1, 2, x=3))  # 호출됨: 42, 인자: (1, 2), 키워드: {'x': 3}

with obj as context:
    print("컨텍스트 내부 코드")
# 컨텍스트 시작
# 컨텍스트 내부 코드
# 컨텍스트 종료

print(obj + 10)        # MyClass 객체: 52
print(20 + obj)        # MyClass 객체: 62
```

#### 핵심 특수 메서드 목록

다음은 파이썬에서 가장 많이 사용되는 특수 메서드들입니다:

| 카테고리 | 메서드 | 설명 | 관련 연산/함수 |
|---------|-------|------|--------------|
| __객체 생성/초기화__ | `__new__` | 객체 생성 | `클래스()` 호출 |
|  | `__init__` | 객체 초기화 | 생성자 |
|  | `__del__` | 객체 소멸자 | 가비지 컬렉션 시 |
| __문자열 변환__ | `__str__` | 사용자 친화적 문자열 표현 | `str()`, `print()` |
|  | `__repr__` | 개발자용 문자열 표현 | `repr()` |
|  | `__format__` | 포맷 문자열 변환 | `format()`, f-문자열 |
| __산술 연산자__ | `__add__`, `__radd__` | 덧셈 | `+` |
|  | `__sub__`, `__rsub__` | 뺄셈 | `-` |
|  | `__mul__`, `__rmul__` | 곱셈 | `*` |
|  | `__truediv__`, `__rtruediv__` | 나눗셈 | `/` |
|  | `__floordiv__`, `__rfloordiv__` | 정수 나눗셈 | `//` |
|  | `__mod__`, `__rmod__` | 나머지 | `%` |
|  | `__pow__`, `__rpow__` | 거듭제곱 | `**`, `pow()` |
|  | `__neg__` | 단항 부정 | `-obj` |
|  | `__pos__` | 단항 양수 | `+obj` |
|  | `__abs__` | 절댓값 | `abs()` |
| __비교 연산자__ | `__eq__` | 동등 비교 | `==` |
|  | `__ne__` | 불일치 비교 | `!=` |
|  | `__lt__` | 작음 비교 | `<` |
|  | `__le__` | 작거나 같음 비교 | `<=` |
|  | `__gt__` | 큼 비교 | `>` |
|  | `__ge__` | 크거나 같음 비교 | `>=` |
| __컨테이너 동작__ | `__len__` | 길이 계산 | `len()` |
|  | `__getitem__` | 요소 접근 | `obj[key]` |
|  | `__setitem__` | 요소 설정 | `obj[key] = value` |
|  | `__delitem__` | 요소 삭제 | `del obj[key]` |
|  | `__contains__` | 멤버십 테스트 | `in` |
|  | `__iter__` | 이터레이터 반환 | `iter()`, `for` 루프 |
|  | `__next__` | 다음 요소 반환 | `next()`, 이터레이션 |
| __속성 접근__ | `__getattr__` | 존재하지 않는 속성 접근 | `obj.name` |
|  | `__getattribute__` | 모든 속성 접근 | `obj.name` |
|  | `__setattr__` | 속성 설정 | `obj.name = value` |
|  | `__delattr__` | 속성 삭제 | `del obj.name` |
| __기타 동작__ | `__call__` | 함수처럼 호출 | `obj()` |
|  | `__enter__`, `__exit__` | 컨텍스트 관리자 | `with` 문 |
|  | `__hash__` | 해시값 계산 | `hash()` |

특수 메서드를 활용하면 자신만의 클래스가 파이썬 내장 타입처럼 동작하도록 만들 수 있으며, 기존 연산자와 함수에 맞춤형 동작을 부여할 수 있습니다.

### b. 데코레이터(`@`)를 활용한 메서드 유형 지정

파이썬은 데코레이터(`@` 문법)를 사용하여 메서드의 유형을 지정할 수 있습니다. 대표적으로 `@classmethod`와 `@staticmethod`가 있습니다:

```python
class Calculator:
    # 클래스 변수
    pi = 3.14159
    
    def __init__(self, value=0):
        # 인스턴스 변수
        self.value = value
    
    # 인스턴스 메서드 - 첫 번째 인자로 self를 받음
    def add(self, x):
        self.value += x
        return self.value
    
    # 클래스 메서드 - 첫 번째 인자로 cls를 받음
    @classmethod
    def create_zero(cls):
        """0으로 초기화된 새 인스턴스 반환"""
        return cls(0)
    
    # 정적 메서드 - self나 cls를 받지 않음
    @staticmethod
    def is_positive(x):
        """양수 여부 확인"""
        return x > 0
    
    # 속성 접근처럼 사용할 수 있는 메서드
    @property
    def square(self):
        """현재 값의 제곱 계산"""
        return self.value ** 2
    
    # setter 프로퍼티 - 속성 설정 동작 정의
    @square.setter
    def square(self, new_square):
        # 제곱근 계산으로 값 설정
        self.value = new_square ** 0.5

# 사용 예시
calc = Calculator(5)
print(calc.add(3))                # 8 (인스턴스 메서드)
print(Calculator.create_zero())    # <__main__.Calculator object at 0x...> (클래스 메서드)
print(Calculator.is_positive(-1))  # False (정적 메서드)
print(calc.is_positive(10))        # True (인스턴스에서도 정적 메서드 호출 가능)

# 프로퍼티 사용
print(calc.square)                 # 64 (제곱값)
calc.square = 100                  # setter를 통해 값 설정
print(calc.value)                  # 10 (제곱근 계산됨)
```

#### 메서드 유형 비교

| 메서드 유형 | 첫 번째 인자 | 호출 방법 | 주요 용도 |
|------------|------------|---------|---------|
| 인스턴스 메서드 | `self` (인스턴스) | `obj.method()` | 인스턴스 데이터 조작 |
| 클래스 메서드 | `cls` (클래스) | `Class.method()` 또는 `obj.method()` | 대체 생성자, 팩토리 메서드 |
| 정적 메서드 | 특별한 첫 인자 없음 | `Class.method()` 또는 `obj.method()` | 유틸리티 함수, 헬퍼 기능 |
| 프로퍼티 | `self` (인스턴스) | `obj.property` (메서드처럼 보이지 않음) | 계산된 속성, 캡슐화 |

데코레이터를 사용한 메서드 유형 지정은 객체 지향 설계에서 중요한 도구로, 각 메서드의 목적과 책임을 명확히 표현할 수 있게 해줍니다.

---
> [목차로 돌아가기](../../README.md) | [이전: 타입 시스템과 타입 힌트](./1_2_type_system.md) | [다음: 기본 데이터 타입](./1_4_basic_data_types.md)
