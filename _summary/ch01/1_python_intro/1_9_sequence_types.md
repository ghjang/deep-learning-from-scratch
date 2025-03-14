# 1.9 시퀀스 타입과 슬라이싱

> [목차로 돌아가기](../../README.md) | [이전: 기본 데이터 타입](./1_8_basic_data_types.md) | [다음: 파이썬 2와 3의 주요 차이점](./a_1_ver2_vs_ver3.md)

## 1.9.1 시퀀스 타입

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
  * \+ 연산자(concatenation)와 * 연산자(repetition) 지원
  * in 연산자를 통한 포함 여부 확인

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

## 1.9.2 슬라이싱

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

## 1.9.3 시퀀스 언패킹

* __시퀀스 언패킹은 시퀀스의 요소들을 개별 변수에 할당하는 강력한 기능이다.__

  ```python
  # 기본 언패킹
  a, b, c = [1, 2, 3]
  print(a, b, c)  # 1 2 3
  
  # 더 많은 값 무시하기
  first, *rest = [1, 2, 3, 4, 5]
  print(first, rest)  # 1 [2, 3, 4, 5]
  
  # 중간 값 추출
  first, *middle, last = [1, 2, 3, 4, 5]
  print(first, middle, last)  # 1 [2, 3, 4] 5
  
  # 함수 인자로 활용
  def add_multiply(a, b, c):
      return a + b, b * c
  
  values = [2, 3, 4]
  sum_result, product_result = add_multiply(*values)
  print(sum_result, product_result)  # 5 12
  ```

  언패킹은 코드를 더 간결하고 표현력 있게 만들며, 특히 함수 호출과 결과 처리에 유용하다.

## 1.9.4 바이트 타입 심화

* __바이트 타입(`bytes`와 `bytearray`)은 바이너리 데이터를 처리하는 시퀀스 타입이다.__

  ### a. 바이트 객체 생성

  ```python
  # 다양한 bytes 생성 방법
  b1 = bytes([65, 66, 67])          # ASCII 값으로 생성
  b2 = bytes(b"ABC")                # 리터럴 문법
  b3 = bytes("안녕", encoding="utf-8")  # 문자열로부터 인코딩
  
  print(b1)  # b'ABC'
  print(b3)  # b'\xec\x95\x88\xeb\x85\x95' (UTF-8로 인코딩된 '안녕')
  ```

  ### b. 바이트 시퀀스 연산

  ```python
  # 바이트 시퀀스 인덱싱과 슬라이싱
  b = bytes([104, 101, 108, 108, 111])  # b'hello'
  print(b[0])      # 104 (정수값 반환)
  print(b[1:3])    # b'el' (슬라이스는 바이트 객체 반환)
  print(list(b))   # [104, 101, 108, 108, 111] (정수 리스트로 변환)
  
  # 연결 및 반복
  b1 = b'Hello, '
  b2 = b'World!'
  print(b1 + b2)   # b'Hello, World!'
  print(b'-' * 5)  # b'-----'
  ```

  ### c. 수정 가능한 `bytearray`

  ```python
  # bytearray 생성
  ba = bytearray([65, 66, 67])  # bytearray(b'ABC')
  
  # 요소 수정
  ba[0] = 68  # ASCII 'D'
  print(ba)   # bytearray(b'DBC')
  
  # 메서드 사용
  ba.append(69)  # ASCII 'E'
  ba.extend(b'FG')
  print(ba)   # bytearray(b'DBCEFG')
  
  # bytes와 bytearray 변환
  frozen = bytes(ba)    # 불변 복사본 생성
  mutable = bytearray(b'hello')  # 가변 복사본 생성
  ```
  
  ### d. 인코딩과 디코딩

  ```python
  # 문자열 인코딩 (string → bytes)
  text = "안녕하세요"
  utf8_bytes = text.encode('utf-8')
  print(utf8_bytes)  # b'\xec\x95\x88\xeb\x85\x95\xed\x95\x98\xec\x84\xb8\xec\x9a\x94'
  print(len(text))   # 5 (문자 수)
  print(len(utf8_bytes))  # 15 (바이트 수, UTF-8에서 한글은 문자당 3바이트)
  
  # 다양한 인코딩
  cp949_bytes = text.encode('cp949')  # 윈도우 한글 인코딩
  print(cp949_bytes)  # b'\xbe\xc8\xb3\xe7\xc7\xcf\xbc\xbc\xbf\xe4'
  
  # 바이트 디코딩 (bytes → string)
  decoded = utf8_bytes.decode('utf-8')
  print(decoded)  # '안녕하세요'
  
  # 오류 처리
  invalid_bytes = b'\xff\xfe\xfd'  # 유효하지 않은 UTF-8 시퀀스
  try:
      decoded = invalid_bytes.decode('utf-8')
  except UnicodeDecodeError:
      print("디코딩 오류 발생")
      
  # 오류 처리 옵션 지정
  decoded = invalid_bytes.decode('utf-8', errors='replace')
  print(decoded)  # '���' (유효하지 않은 문자는 �로 대체)
  ```
  
  ### e. 바이트 타입의 활용 사례
  
  ```python
  # 파일 입출력
  with open('image.jpg', 'rb') as f:
      data = f.read()  # bytes 객체로 읽음
      print(type(data))  # <class 'bytes'>
      print(f"파일 크기: {len(data)} 바이트")
  
  # 네트워크 통신
  import socket
  
  # 소켓 통신에서는 바이트 타입으로 데이터 송수신
  msg = b"Hello, server!"
  # client_socket.send(msg)  # 실제 코드에서는 연결된 소켓 필요
  
  # 해시 함수와 암호화
  import hashlib
  data = b"sensitive data"
  hash_value = hashlib.sha256(data).hexdigest()
  print(f"SHA-256 해시: {hash_value}")
  ```

  바이트 타입은 네트워크 프로그래밍, 파일 조작, 암호화, 이미지 처리 등의 바이너리 데이터를 다룰 때 필수적이다.

## 1.9.5 시퀀스 타입별 특화 기능

* __각 시퀀스 타입은 고유한 특징과 메서드를 가진다.__

  ### a. 문자열(str) 고유 메서드
  
  ```python
  text = "  Hello, World!  "
  
  # 대소문자 변환
  print(text.upper())       # "  HELLO, WORLD!  "
  print(text.lower())       # "  hello, world!  "
  print(text.title())       # "  Hello, World!  "
  
  # 공백 처리
  print(text.strip())       # "Hello, World!" (앞뒤 공백 제거)
  print(text.lstrip())      # "Hello, World!  " (왼쪽 공백만 제거)
  print(text.rstrip())      # "  Hello, World!" (오른쪽 공백만 제거)
  
  # 검색과 대체
  print(text.find("World"))  # 9 (위치 반환, 없으면 -1)
  print(text.replace("World", "Python"))  # "  Hello, Python!  "
  
  # 분할과 결합
  parts = "apple,banana,orange".split(",")
  print(parts)              # ['apple', 'banana', 'orange']
  print("|".join(parts))    # "apple|banana|orange"
  ```

  ### b. 리스트(list) 고유 메서드
  
  ```python
  numbers = [1, 2, 3, 4]
  
  # 요소 추가
  numbers.append(5)    # [1, 2, 3, 4, 5]
  numbers.insert(0, 0) # [0, 1, 2, 3, 4, 5]
  numbers.extend([6, 7])  # [0, 1, 2, 3, 4, 5, 6, 7]
  
  # 요소 제거
  numbers.remove(0)    # [1, 2, 3, 4, 5, 6, 7]
  popped = numbers.pop()  # 7, numbers = [1, 2, 3, 4, 5, 6]
  popped_idx = numbers.pop(0)  # 1, numbers = [2, 3, 4, 5, 6]
  
  # 정렬 및 역순
  numbers.sort()       # [2, 3, 4, 5, 6] (원본 변경)
  numbers.reverse()    # [6, 5, 4, 3, 2] (원본 변경)
  ```

  ### c. 튜플(tuple)의 특징과 활용
  
  ```python
  # 튜플은 불변이지만 효율적인 데이터 구조
  point = (10, 20)
  
  # 튜플 언패킹
  x, y = point
  
  # 튜플을 반환하는 함수
  def get_dimensions():
      return (1920, 1080)
  
  width, height = get_dimensions()
  print(f"해상도: {width}x{height}")
  
  # 네임드 튜플 (가독성 향상)
  from collections import namedtuple
  
  Point = namedtuple('Point', ['x', 'y'])
  p = Point(10, 20)
  print(p.x, p.y)  # 10 20
  print(p[0], p[1])  # 10 20 (인덱스로도 접근 가능)
  ```

  ### d. range의 특징과 활용
  
  ```python
  # range는 메모리 효율적인 시퀀스 (실제 리스트를 생성하지 않음)
  r = range(1, 10000000)  # 메모리에 천만 개 정수를 저장하지 않음
  print(f"크기: {r.stop - r.start}")  # 크기: 9999999
  
  # range 속성
  r = range(5, 20, 3)
  print(r.start)  # 5
  print(r.stop)   # 20
  print(r.step)   # 3
  
  # 리스트로 변환하면 실제 값이 모두 생성됨
  print(list(range(5)))  # [0, 1, 2, 3, 4]
  print(list(range(2, 10, 2)))  # [2, 4, 6, 8]
  
  # 음수 스텝
  print(list(range(10, 0, -1)))  # [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
  ```

> [목차로 돌아가기](../../README.md) | [이전: 기본 데이터 타입](./1_8_basic_data_types.md) | [다음: 파이썬 2와 3의 주요 차이점](./a_1_ver2_vs_ver3.md)
