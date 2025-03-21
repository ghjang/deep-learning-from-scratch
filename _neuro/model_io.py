import os
import json
import numpy as np
from typing import TypeVar, Generic, Any, cast, Callable
from numpy.typing import NDArray
from activation import (
    ActivationFunction,
    identity,
    sigmoid,
    relu,
    leaky_relu,
    elu,
    tanh,
    softmax,
)

# 모델 타입 정의 (타입 가변성 허용)
T = TypeVar("T")


class SaveLoadMixin(Generic[T]):
    """
    모델의 저장 및 로드 기능을 제공하는 믹스인 클래스입니다.

    이 믹스인을 상속받는 클래스는 다음을 구현해야 합니다:
    1. layers 속성: 모델의 레이어 목록
    2. create 메서드: 새 인스턴스 생성
    3. layer 메서드: 레이어 추가
    """

    # 활성화 함수 매핑 테이블을 클래스 상수로 정의
    _ACTIVATION_NAME_MAP: dict[Callable[[NDArray], NDArray] | None, str] = {
        None: "none",
        identity: "identity",
        sigmoid: "sigmoid",
        relu: "relu",
        leaky_relu: "leaky_relu",
        elu: "elu",
        tanh: "tanh",
        softmax: "softmax",
    }

    _ACTIVATION_FUNCTION_MAP: dict[str, Callable[[NDArray], NDArray] | None] = {
        "none": None,
        "identity": identity,
        "sigmoid": sigmoid,
        "relu": relu,
        "leaky_relu": leaky_relu,
        "elu": elu,
        "tanh": "tanh",
        "softmax": softmax,
    }

    def save(self, filepath: str, overwrite: bool = False) -> None:
        """모델 상태를 파일에 저장합니다.

        Args:
            filepath: 저장할 파일 경로 (.npz 또는 .json 확장자)
            overwrite: 파일이 이미 존재할 경우 덮어쓸지 여부 (기본값: False)
                       False인 경우 파일이 이미 존재하면 FileExistsError 발생

        Raises:
            FileExistsError: 파일이 이미 존재하고 overwrite=False인 경우
            ValueError: 지원하지 않는 파일 형식인 경우
        """
        # 파일 존재 여부 확인 및 덮어쓰기 설정 검사
        if os.path.exists(filepath) and not overwrite:
            raise FileExistsError(
                f"파일 '{filepath}'이(가) 이미 존재합니다. 덮어쓰려면 overwrite=True로 설정하세요."
            )

        # "self"는 믹스인을 상속한 클래스의 인스턴스
        model = self

        if filepath.endswith(".npz"):
            # 각 레이어의 데이터를 별도 키로 저장 (NumPy 형식)
            save_dict = {}

            for i, layer in enumerate(model.layers):
                # 레이어 정보 저장
                save_dict[f"layer_{i}_output_size"] = layer.output_size

                # 가중치와 편향 저장
                if layer.weights is not None:
                    save_dict[f"layer_{i}_weights"] = layer.weights

                if layer.biases is not None:
                    save_dict[f"layer_{i}_biases"] = layer.biases

                # 활성화 함수 이름 저장
                activation_name = self._get_activation_name(layer.activation)
                save_dict[f"layer_{i}_activation"] = activation_name

            # 레이어 수 저장
            save_dict["num_layers"] = len(model.layers)

            np.savez(filepath, **save_dict)
            print(f"모델이 {filepath}에 저장되었습니다. (NumPy 형식)")

        elif filepath.endswith(".json"):
            model_data = {"layers": []}

            for layer in model.layers:
                layer_data = {"output_size": layer.output_size}

                # 가중치와 편향을 리스트로 변환
                if layer.weights is not None:
                    layer_data["weights"] = layer.weights.tolist()
                else:
                    layer_data["weights"] = None

                if layer.biases is not None:
                    layer_data["biases"] = layer.biases.tolist()
                else:
                    layer_data["biases"] = None

                # 활성화 함수 이름 저장
                activation_name = self._get_activation_name(layer.activation)
                layer_data["activation"] = activation_name
                model_data["layers"].append(layer_data)

            with open(filepath, "w") as f:
                json.dump(model_data, f, indent=2)
            print(f"모델이 {filepath}에 저장되었습니다. (JSON 형식)")
        else:
            raise ValueError(
                f"지원하지 않는 파일 확장자입니다: {filepath}. '.npz' 또는 '.json' 확장자를 사용하세요."
            )

    @classmethod
    def load(cls, filepath: str) -> T:
        """저장된 모델을 로드합니다.

        Args:
            filepath: 로드할 파일 경로

        Returns:
            로드된 모델 인스턴스

        Raises:
            FileNotFoundError: 파일이 존재하지 않는 경우
            ValueError: 지원하지 않는 파일 형식인 경우
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")

        # 타입 힌트를 위한 캐스팅
        model_class = cast(Any, cls)

        if filepath.endswith(".npz"):
            # NumPy .npz 형식에서 로드
            data = np.load(filepath)
            num_layers = int(data["num_layers"])

            model = model_class.create()

            # 각 레이어 정보 복원
            for i in range(num_layers):
                output_size = int(data[f"layer_{i}_output_size"])
                model.layer(output_size)

                # 마지막으로 추가된 레이어
                current_layer = model.layers[-1]

                # 가중치와 편향 복원
                if f"layer_{i}_weights" in data:
                    current_layer.weights = data[f"layer_{i}_weights"]

                if f"layer_{i}_biases" in data:
                    current_layer.biases = data[f"layer_{i}_biases"]

                # 활성화 함수 복원
                activation_name = str(data[f"layer_{i}_activation"])
                current_layer.activation = cls._get_activation_function(activation_name)

            print(f"모델이 {filepath}에서 로드되었습니다. (NumPy 형식)")
            return model

        elif filepath.endswith(".json"):
            with open(filepath, "r") as f:
                model_data = json.load(f)

            model = model_class.create()

            for layer_data in model_data["layers"]:
                model.layer(layer_data["output_size"])

                # 마지막으로 추가된 레이어
                current_layer = model.layers[-1]

                # 가중치와 편향이 None이 아닌 경우 설정
                if layer_data["weights"] is not None:
                    current_layer.weights = np.array(layer_data["weights"])

                if layer_data["biases"] is not None:
                    current_layer.biases = np.array(layer_data["biases"])

                # 활성화 함수 설정
                activation_name = layer_data["activation"]
                current_layer.activation = cls._get_activation_function(activation_name)

            print(f"모델이 {filepath}에서 로드되었습니다. (JSON 형식)")
            return model

        else:
            raise ValueError(
                f"지원하지 않는 파일 형식입니다: {filepath}. '.npz' 또는 '.json' 확장자를 사용하세요."
            )

    @staticmethod
    def _get_activation_name(activation_function: ActivationFunction | None) -> str:
        """활성화 함수를 이름으로 변환합니다."""
        return SaveLoadMixin._ACTIVATION_NAME_MAP.get(activation_function, "none")

    @staticmethod
    def _get_activation_function(activation_name: str) -> ActivationFunction | None:
        """이름을 활성화 함수로 변환합니다."""
        return SaveLoadMixin._ACTIVATION_FUNCTION_MAP.get(activation_name, None)
