# 1.3 객체 시스템과 객체 지향 프로그래밍 지원

> [목차로 돌아가기](../../README.md) | [이전: 타입 시스템과 타입 힌트](./1_2_type_system.md) | [다음: 기본 데이터 타입](./1_4_basic_data_types.md)

## 1.3.1 파이썬의 객체 시스템

a. __파이썬에서 모든 표현 대상은 '객체'이다.__

  '객체'는 '메모리에 저장된 데이터'와 '데이터를 처리하는 함수'를 가지고 있다. '객체'는 '변수'에 할당할 수 있다. '변수'는 '객체'를 가리키는 '레퍼런스'를 가지고 있다.

  'class'로 명시적으로 표현되는 타입뿐만 아니라 'int', 'float', 'list', 'dict' 등의 내장 타입도 '객체'이다. 다음과 같은 코드는 모두 '객체'를 생성한다:

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

b. __파이썬에서 모든 표현 대상은 'object' 클래스를 상속받은 '객체'이다.__

  파이썬에서는 모든 타입이 'object' 클래스를 암묵적으로 상속받는다. 이는 모든 객체가 공통으로 가지는 기본 메서드와 속성이 있다는 것을 의미한다:

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

* `__str__()`: 문자열 표현 (str() 함수가 호출)
* `__repr__()`: 개발자를 위한 상세 문자열 표현
* `__class__`: 객체의 타입 정보
* `__doc__`: 문서화 문자열
* `__dict__`: 객체의 속성 목록 (네임스페이스)
* `__hash__()`: 해시 값 계산 (딕셔너리 키로 사용 가능한지 결정)
* `__eq__()`: 동등성 비교 (== 연산자)

c. __클래스와 인스턴스의 기본 개념__

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

a. __파이썬의 변수는 객체의 참조이다.__

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

b. __객체에는 가변 객체와 불변 객체가 있다.__

  파이썬의 객체는 생성 후 수정 가능 여부에 따라 두 그룹으로 나뉜다:
  
  1. __불변 객체__ (Immutable): 생성 후 내용을 변경할 수 없음
     * 숫자 (int, float, complex)
     * 문자열 (str)
     * 튜플 (tuple)
     * 불변 집합 (frozenset)
     * 바이트 (bytes)

  2. __가변 객체__ (Mutable): 생성 후 내용을 변경할 수 있음
     * 리스트 (list)
     * 딕셔너리 (dict)
     * 집합 (set)
     * 바이트 배열 (bytearray)
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

c. __함수 인자는 객체 참조로 전달된다.__ (Call by Object Reference)

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

a. __가비지 컬렉션과 참조 카운팅으로 메모리를 관리한다.__

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

## 1.3.4 객체 지향 프로그래밍 핵심 원칙

a. __상속과 다형성__

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

b. __캡슐화와 접근 제어__

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

c. __특수 메서드와 연산자 오버로딩__

  (새로운 내용 추가 또는 기존 내용 이동)

파이썬의 객체 지향 특성은 유연하면서도 강력한 프로그래밍 패러다임을 제공한다. 모든 것이 객체인 파이썬에서는 이러한 객체 지향 개념을 일관되게 활용할 수 있다.

> [목차로 돌아가기](../../README.md) | [이전: 타입 시스템과 타입 힌트](./1_2_type_system.md) | [다음: 기본 데이터 타입](./1_4_basic_data_types.md)
