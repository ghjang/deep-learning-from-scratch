# 부록 B: 파이썬 Pickle 객체 직렬화

> [목차로 돌아가기](../../README.md) | [이전: 파이썬 2와 3의 주요 차이점](./appendix_a_ver2_vs_ver3.md)

## B.1 Pickle이란?

__Pickle은 파이썬 객체를 바이트 스트림으로 직렬화하고 역직렬화하는 표준 모듈이다.__ 이를 통해 객체를 파일에 저장하거나, 네트워크를 통해 전송하거나, 나중에 재구성할 수 있다. 특히 딥러닝에서 모델의 가중치나 학습 상태를 저장하는 데 자주 사용된다.

## B.2 기본 사용법

### a. 객체 직렬화 (Pickling)

```python
import pickle

# 직렬화할 객체
data = {
    'a': [1, 2.0, 3, 4+6j],
    'b': ("character string", b"byte string"),
    'c': {None, True, False}
}

# 파일에 저장
with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)

# 바이트 문자열로 직렬화
serialized = pickle.dumps(data)
print(type(serialized))  # <class 'bytes'>
```

### b. 객체 역직렬화 (Unpickling)

```python
# 파일에서 로드
with open('data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

print(loaded_data == data)  # True

# 바이트 문자열에서 역직렬화
deserialized = pickle.loads(serialized)
print(deserialized == data)  # True
```

## B.3 프로토콜 버전

Pickle은 여러 프로토콜 버전을 지원하며, 높은 버전일수록 더 효율적이고 더 많은 기능을 제공한다:

```python
# 기본 프로토콜(현재 파이썬의 기본값은 4)
pickle.dumps(data)

# 특정 프로토콜 지정(0부터 5까지 가능, 파이썬 3.8 기준)
pickle.dumps(data, protocol=4)

# 가장 높은 프로토콜 사용
pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

# 프로토콜 호환성 확보(하위 버전과 호환)
pickle.dumps(data, protocol=2)  # 파이썬 2.3부터 지원
```

각 프로토콜 버전의 특징:

* __프로토콜 0__: ASCII 프로토콜, 사람이 읽을 수 있지만 효율성이 낮음
* __프로토콜 1__: 이진 형식, 파이썬 2.3부터 도입
* __프로토콜 2__: 클래스 및 인스턴스 효율성 개선, 파이썬 2.3부터 도입
* __프로토콜 3__: Python 3 전용으로 bytes 객체 지원 개선
* __프로토콜 4__: 대형 객체 지원 개선, 더 효율적인 데이터 구조 (파이썬 3.4+)
* __프로토콜 5__: 대역 외 데이터와 함께 다양한 개선사항 (파이썬 3.8+)

## B.4 보안 고려사항

pickle 모듈은 신뢰할 수 있는 데이터에만 사용해야 한다. 악의적으로 조작된 pickle 데이터는 불필요한 코드를 실행할 수 있다:

```python
# 주의: 신뢰할 수 없는 출처의 데이터를 unpickle하지 말 것
# 아래 코드는 데모용이며, 실제로 실행하면 안됨
import pickle, os

class Exploit:
    def __reduce__(self):
        # 이 코드는 pickle을 역직렬화할 때 실행됨
        return (os.system, ('echo "보안 경고: 코드가 실행되었습니다!"',))

# exploit = pickle.dumps(Exploit())
# pickle.loads(exploit)  # 경고: 이 코드는 os.system()을 실행함
```

따라서 웹 애플리케이션이나 신뢰할 수 없는 출처의 데이터를 처리할 때는 JSON, XML 등 더 안전한 직렬화 형식을 사용하는 것이 좋다.

## B.5 딥러닝에서의 활용

딥러닝 라이브러리들은 모델과 가중치를 저장하는 데 pickle 또는 이를 기반으로 한 방식을 사용한다:

### a. NumPy 배열 저장

```python
import numpy as np
import pickle

# 가중치 행렬 생성
weights = np.random.randn(1000, 1000)

# pickle로 저장
with open('weights.pkl', 'wb') as f:
    pickle.dump(weights, f)

# 로드
with open('weights.pkl', 'rb') as f:
    loaded_weights = pickle.load(f)

print(np.array_equal(weights, loaded_weights))  # True
```

### b. PyTorch에서의 사용

PyTorch는 내부적으로 pickle을 사용하여 모델을 저장한다:

```python
import torch
import torch.nn as nn

# 간단한 모델 정의
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# 모델 저장 (내부적으로 pickle 사용)
torch.save(model, 'model.pt')

# 모델 불러오기
loaded_model = torch.load('model.pt')
```

### c. 가중치만 저장하기

전체 모델 대신 가중치만 저장하고 불러오는 방식이 더 유연하다:

```python
# 가중치만 저장
torch.save(model.state_dict(), 'model_weights.pt')

# 빈 모델 생성 후 가중치 로드
new_model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)
new_model.load_state_dict(torch.load('model_weights.pt'))
```

## B.6 pickle의 한계와 대안

### a. pickle의 한계

* __호환성__: 다른 버전의 파이썬이나 라이브러리에서 생성한 pickle 파일은 호환되지 않을 수 있음
* __보안__: 신뢰할 수 없는 출처의 데이터 역직렬화 시 위험함
* __파이썬 전용__: 다른 언어와의 상호 운용성이 제한됨
* __디버깅 어려움__: 이진 형식이므로 내용 검사가 어려움

### b. 대안 솔루션

#### JSON

간단한 데이터 구조에 적합하며 사람이 읽을 수 있고 언어 간 호환성이 좋다:

```python
import json

data = {"name": "Alice", "scores": [95, 87, 92]}

# 직렬화
json_str = json.dumps(data)
print(json_str)  # {"name": "Alice", "scores": [95, 87, 92]}

# 역직렬화
loaded_data = json.loads(json_str)
print(loaded_data == data)  # True
```

#### HDF5 (h5py)

대용량 수치 데이터에 특화된 형식:

```python
import h5py
import numpy as np

# 대용량 데이터 생성
data = np.random.rand(1000, 1000)

# HDF5 파일에 저장
with h5py.File('data.h5', 'w') as f:
    f.create_dataset('matrix', data=data)

# 데이터 로드
with h5py.File('data.h5', 'r') as f:
    loaded_data = f[('matrix')][()]

print(np.array_equal(data, loaded_data))  # True
```

#### Protocol Buffers, MessagePack

구조화된 데이터의 효율적인 직렬화를 위한 대안:

```python
# MessagePack 예시
import msgpack

data = {"compact": True, "schema": 0, "values": [1, 2, 3]}

# 직렬화
packed = msgpack.packb(data)

# 역직렬화
unpacked = msgpack.unpackb(packed)
print(unpacked == data)  # True
```

## B.7 Pickle 사용 시 모범 사례

### a. 버전 관리

```python
# 저장 시 버전 정보 함께 저장
import pickle
import sys

data = {"model_weights": weights, "version": "1.0"}

with open('model_with_version.pkl', 'wb') as f:
    pickle.dump(data, f)

# 로드 시 버전 확인
with open('model_with_version.pkl', 'rb') as f:
    loaded = pickle.load(f)
    
version = loaded.get("version")
if version != "1.0":
    print(f"경고: 다른 버전의 모델 ({version})이 로드되었습니다.")
```

### b. 예외 처리

```python
try:
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
except (pickle.PickleError, OSError) as e:
    print(f"파일 로딩 중 오류 발생: {e}")
    # 대체 로직 또는 기본값 사용
    data = default_data
```

### c. 딥러닝에서의 효율적인 저장

```python
# PyTorch 모델 저장 시 CPU로 이동하여 저장
# (다른 장치에서 로드할 수 있도록)
model.to('cpu')
torch.save(model.state_dict(), 'model_cpu.pt')

# 압축 옵션 사용
torch.save(model.state_dict(), 'model_compressed.pt', _use_new_zipfile_serialization=True)
```

### d. 큰 데이터의 효율적인 처리

```python
# 대용량 데이터를 청크(chunk)로 처리
import pickle

# 대용량 데이터를 생성하는 함수
def generate_large_data():
    for i in range(1000):
        yield [i] * 10000  # 각 청크는 10,000개의 항목 포함

# 청크 단위로 직렬화
with open('large_data.pkl', 'wb') as f:
    for chunk in generate_large_data():
        pickle.dump(chunk, f)

# 청크 단위로 역직렬화
chunks = []
with open('large_data.pkl', 'rb') as f:
    while True:
        try:
            chunk = pickle.load(f)
            chunks.append(chunk)
        except EOFError:
            break  # 파일 끝에 도달
```

## B.8 복잡한 객체 직렬화

### a. 사용자 정의 객체 직렬화

```python
import pickle

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = [None] * len(layers)
    
    def __getstate__(self):
        """pickle용 상태 반환 (특별 메서드)"""
        # weights를 제외한 사본 반환
        state = self.__dict__.copy()
        # 가중치는 별도로 처리되어야 한다고 가정
        state['weights'] = None
        return state
    
    def __setstate__(self, state):
        """pickle에서 상태 복원 (특별 메서드)"""
        self.__dict__.update(state)
        # 가중치 초기화 등 추가 작업
        self.weights = [None] * len(self.layers)

# 객체 생성 및 저장
nn = NeuralNetwork([10, 5, 1])
with open('custom_nn.pkl', 'wb') as f:
    pickle.dump(nn, f)

# 객체 로드
with open('custom_nn.pkl', 'rb') as f:
    loaded_nn = pickle.load(f)
```

### b. 참조와 순환 참조

pickle은 동일 객체의 여러 참조와 심지어 순환 참조도 자동으로 처리한다:

```python
import pickle

# 순환 참조가 있는 데이터 구조
a = [1, 2, 3]
b = [4, 5, 6, a]  # b는 a를 참조
a.append(b)       # a는 b를 참조 - 순환 참조 발생

# 직렬화 및 역직렬화
serialized = pickle.dumps(a)
deserialized = pickle.loads(serialized)

print(deserialized[3] is deserialized)  # True: 순환 참조 유지됨
```

pickle은 복잡한 데이터 구조와 객체를 효율적으로 저장하는 강력한 도구이지만, 보안과 호환성에 주의해야 한다. 특히 딥러닝에서는 모델의 가중치와 상태를 저장하는 데 널리 사용되므로 이해하고 적절하게 활용하는 것이 중요하다.

> [목차로 돌아가기](../../README.md) | [이전: 파이썬 2와 3의 주요 차이점](./appendix_a_ver2_vs_ver3.md)
