# 1.3 가변(Mutable)과 불변(Immutable) 타입 및 시퀀스

> [목차로 돌아가기](../../README.md) | [이전: 파이썬 기본 자료형](./1_2_data_types.md) | [다음: 객체 지향 특성과 특수 연산자](./1_4_oop_and_operators.md)

## 1.3.1 가변(Mutable)과 불변(Immutable) 타입

* __파이썬의 데이터 타입은 '가변(mutable)'과 '불변(immutable)'으로 나뉜다.__

  이 구분은 객체가 생성된 후 그 내용을 변경할 수 있는지 여부를 결정한다:

  * 불변(Immutable) 타입: 한번 생성되면 내용을 변경할 수 없는 객체
    * int, float, bool, str, tuple, frozenset, bytes
  
  * 가변(Mutable) 타입: 내용을 자유롭게 변경할 수 있는 객체
    * list, dict, set, bytearray, 사용자 정의 클래스
  
  불변 객체는 내용을 변경하는 연산을 수행하면 새로운 객체가 생성된다:

  ```python
  # 문자열(불변 객체) 연산
  s = "hello"
  id_before = id(s)  # 객체의 메모리 주소
  
  s = s + " world"   # 새로운 객체 생성
  id_after = id(s)
  
  print(id_before == id_after)  # False - 다른 객체임
  ```

  반면 가변 객체는 내용 변경 시 같은 객체를 유지한다:

  ```python
  # 리스트(가변 객체) 연산
  lst = [1, 2, 3]
  id_before = id(lst)
  
  lst.append(4)  # 같은 객체의 내용 수정
  id_after = id(lst)
  
  print(id_before == id_after)  # True - 동일한 객체임
  ```

* __가변성이 함수 호출과 변수 할당에 미치는 영향__

  불변 객체는 함수에 전달되거나 다른 변수에 할당될 때 전체가 복사된다(값 전달):
  
  ```python
  def modify_value(x):
      x = x + 1  # 새 객체 생성
      return x
  
  num = 10
  result = modify_value(num)
  print(num)     # 10 (원본 변경 없음)
  print(result)  # 11
  ```
  
  가변 객체는 함수에 전달되거나 다른 변수에 할당될 때 참조가 전달된다(참조 전달):
  
  ```python
  def modify_list(lst):
      lst.append(4)  # 원본 객체 수정
  
  numbers = [1, 2, 3]
  modify_list(numbers)
  print(numbers)  # [1, 2, 3, 4] (원본이 변경됨)
  ```

* __불변성(immutability)의 장점__

  1. 스레드 안전성: 여러 스레드가 동일한 객체에 접근해도 값이 변하지 않음
  2. 예측 가능성: 코드의 다른 부분에서 객체를 변경하지 않을 것이라는 보장
  3. 해시 가능: 딕셔너리 키나 집합의 요소로 사용 가능
  
* __가변성(mutability)의 장점__

  1. 효율성: 큰 데이터의 일부만 수정할 때 전체를 복사할 필요 없음
  2. 메모리 효율: 수정이 필요할 때마다 새 객체를 생성하지 않음
  3. 알고리즘 구현: 특정 알고리즘(정렬, 검색 등)을 더 직관적으로 구현 가능

## 1.3.2 시퀀스 타입과 슬라이싱

* __파이썬의 '시퀀스 타입'은 순서가 있는 데이터 컬렉션이다.__

  파이썬에는 다음과 같은 주요 시퀀스 타입들이 있다:
  
  * 문자열(str): 문자들의 시퀀스
  * 리스트(list): 변경 가능한(mutable) 객체들의 시퀀스
  * 튜플(tuple): 변경 불가능한(immutable) 객체들의 시퀀스
  * 범위(range): 정수 시퀀스(연속적인 숫자들)
  * 바이트(bytes): 바이트의 불변 시퀀스
  * 바이트배열(bytearray): 바이트의 가변 시퀀스
  
  모든 시퀀스 타입은 다음과 같은 공통 특징을 가진다:
  
  * 인덱싱을 통한 접근 가능 (0부터 시작)
  * 슬라이싱 연산 지원
  * len() 함수를 통한 길이 확인 가능
  * 반복문(for)에서 순회 가능
  * * 연산자(concatenation)와 * 연산자(repetition) 지원
  * in 연산자를 통한 포함 여부 확인

* __'슬라이싱'은 시퀀스의 일부분을 추출하는 강력한 기능이다.__

  모든 시퀀스 타입에서 공통적으로 사용할 수 있는 슬라이싱 문법은 다음과 같다:

  ```python
  sequence[start:stop:step]
  ```

  각 매개변수의 의미:
  * start: 시작 인덱스 (포함)
  * stop: 종료 인덱스 (미포함)
  * step: 인덱스 증가량

  슬라이싱의 주요 특징:
  
  ```python
  # 기본 슬라이싱 [시작:끝]
  a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  print(a[2:5])     # 출력: [2, 3, 4] (인덱스 2부터 5 전까지)
  
  # 시작 인덱스 생략 (처음부터)
  print(a[:5])      # 출력: [0, 1, 2, 3, 4]
  
  # 끝 인덱스 생략 (끝까지)
  print(a[5:])      # 출력: [5, 6, 7, 8, 9]
  
  # 스텝 사용 [시작:끝:스텝]
  print(a[1:9:2])   # 출력: [1, 3, 5, 7] (1부터 9 전까지 2 간격으로)
  
  # 음수 인덱스 사용
  print(a[-5:-2])   # 출력: [5, 6, 7] (끝에서 5번째부터 끝에서 2번째 전까지)
  
  # 전체 복사
  b = a[:]          # 리스트의 얕은 복사(shallow copy)
  
  # 리스트 뒤집기
  print(a[::-1])    # 출력: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
  ```

* __다양한 시퀀스 타입에 적용되는 슬라이싱:__

  ```python
  # 문자열 슬라이싱
  s = "Hello Python"
  print(s[6:])      # 출력: "Python"
  print(s[:5])      # 출력: "Hello"
  print(s[::-1])    # 출력: "nohtyP olleH"
  
  # 튜플 슬라이싱
  t = (0, 1, 2, 3, 4, 5)
  print(t[1:5:2])   # 출력: (1, 3)
  
  # range 객체의 슬라이싱 (결과는 다시 range 객체가 됨)
  r = range(10)
  print(list(r[2:8:2]))  # 출력: [2, 4, 6]
  ```

* __슬라이싱은 원본 변경 없이 새로운 객체를 반환한다.__

  원본 시퀀스는 변경되지 않고, 슬라이싱 결과로 원본과 동일한 타입의 새 객체가 생성된다.
  단, 얕은 복사(shallow copy)이므로 리스트나 튜플의 요소가 가변 객체인 경우 요소 자체는 공유된다.
  
  '얕은 복사(shallow copy)'란 최상위 컨테이너는 새로 생성하지만, 내부 요소는 원본 객체와 동일한 참조를 가리키는 것을 말한다:
  
  ```python
  # 얕은 복사 예시
  original = [1, 2, [3, 4]]
  shallow_copy = original[:]  # 슬라이싱으로 얕은 복사
  
  # 최상위 컨테이너는 다른 객체다
  print(original is shallow_copy)  # False
  
  # 내부 요소 변경 시 영향 관계 확인
  shallow_copy[0] = 99  # 불변 객체 변경
  print(original)       # [1, 2, [3, 4]] - 원본에 영향 없음
  
  shallow_copy[2][0] = 33  # 가변 객체(내부 리스트)의 요소 변경
  print(original)          # [1, 2, [33, 4]] - 원본도 함께 변경됨
  ```
  
  반면 '깊은 복사(deep copy)'는 객체 내부의 모든 요소까지 재귀적으로 복사하는 것을 의미하며, 이를 위해서는 `copy` 모듈의 `deepcopy` 함수를 사용한다:
  
  ```python
  import copy
  
  original = [1, 2, [3, 4]]
  deep_copy = copy.deepcopy(original)
  
  deep_copy[2][0] = 33  # 가변 객체(내부 리스트)의 요소 변경
  print(original)       # [1, 2, [3, 4]] - 원본에 영향 없음
  print(deep_copy)      # [1, 2, [33, 4]]
  ```

* __시퀀스 타입과 이터러블(Iterable)의 관계__

  파이썬에서 시퀀스 타입은 모두 '이터러블(Iterable)'의 특성을 가진다. 이터러블이란 하나씩 차례대로 꺼내어 쓸 수 있는 객체를 의미한다. 시퀀스 타입 외에도 여러 이터러블이 존재한다:
  
  * 시퀀스 타입 - 리스트, 튜플, 문자열, range 등
  * 집합 타입 - set, frozenset
  * 사전 타입 - dict
  * 제너레이터(generator)
  * 파일 객체
  * 사용자 정의 이터러블(`__iter__()` 메서드를 구현한 객체)
  
  이터러블의 핵심 특징:
  
  1. `for` 루프에서 순회 가능:

     ```python
     # 여러 이터러블을 순회하는 예
     for item in [1, 2, 3]:  # 리스트
         print(item)
     
     for char in "Python":   # 문자열
         print(char)
     
     for key in {"a": 1, "b": 2}:  # 딕셔너리(키가 순회됨)
         print(key)
     ```
  
  2. 반복자(Iterator) 생성 가능:

     ```python
     # 이터러블로부터 반복자 생성
     my_list = [1, 2, 3]
     iterator = iter(my_list)  # __iter__() 메서드 호출
     
     # 반복자에서 값 가져오기
     print(next(iterator))  # 1
     print(next(iterator))  # 2
     print(next(iterator))  # 3
     # print(next(iterator))  # StopIteration 예외 발생
     ```
  
  3. 컴프리헨션(comprehension)과 함께 사용 가능:

     ```python
     # 리스트 컴프리헨션
     squares = [x**2 for x in range(5)]
     print(squares)  # [0, 1, 4, 9, 16]
     
     # 딕셔너리 컴프리헨션
     word = "hello"
     char_positions = {char: idx for idx, char in enumerate(word)}
     print(char_positions)  # {'h': 0, 'e': 1, 'l': 3, 'o': 4}
     ```
  
  4. 내장 함수와 함께 사용 가능:

     ```python
     print(sum([1, 2, 3, 4]))  # 10
     print(max("python"))      # 'y'
     print(min({5, 3, 8, 1}))  # 1
     print(sorted("hello"))    # ['e', 'h', 'l', 'l', 'o']
     ```

> [목차로 돌아가기](../../README.md) | [이전: 파이썬 기본 자료형](./1_2_data_types.md) | [다음: 객체 지향 특성과 특수 연산자](./1_4_oop_and_operators.md)
