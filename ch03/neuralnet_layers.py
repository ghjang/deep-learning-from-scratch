import numpy as np
from manim import *


class NeuralLayers:
    # Wicked 컬러 테마 상수
    EMERALD = "#088F5B"  # 진한 에메랄드 그린
    EMERALD_LIGHT = "#50C878"  # 밝은 에메랄드
    PINK = "#FF69B4"  # 뮤지컬 Wicked의 Glinda 핑크
    NUMBER_COLOR = "#FFD700"  # 골드 (출력층 숫자용)
    PROB_COLOR = "#B39CD0"  # 확률값 표시용 라벤더 컬러

    @staticmethod
    def create_first_layer(neuron_count, radius, color=None, fill_color=None):
        """첫번째 은닉층 생성"""
        color = color or NeuralLayers.EMERALD
        fill_color = fill_color or NeuralLayers.PINK
        layer = VGroup()
        for _ in range(neuron_count):
            neuron = Circle(
                radius=radius,
                color=color,
                fill_color=fill_color,
                fill_opacity=0,
            )
            layer.add(neuron)

        layer.arrange(RIGHT, buff=radius * 0.5)
        return layer

    @staticmethod
    def create_second_layer(neuron_count, radius, color=None, fill_color=None):
        """두번째 은닉층 생성 (직사각형 형태)"""
        color = color or NeuralLayers.EMERALD
        fill_color = fill_color or NeuralLayers.PINK
        layer = VGroup()

        # 직사각형의 크기 계산
        width = radius * 2  # 원의 지름과 동일
        height = width * 3  # 높이는 너비의 3배

        for _ in range(neuron_count):
            neuron = Rectangle(
                width=width,
                height=height,
                color=color,
                stroke_width=DEFAULT_STROKE_WIDTH / 2,  # 선 두께 조정
                fill_color=fill_color,
                fill_opacity=0,
            )
            layer.add(neuron)

        # 간격을 너비의 0.3배로 설정 (적절한 밀도 유지)
        layer.arrange(RIGHT, buff=width * 0.3)
        return layer

    @staticmethod
    def create_output_layer(neuron_count, radius, color=None, fill_color=None):
        """출력층 생성"""
        color = color or NeuralLayers.EMERALD
        fill_color = fill_color or NeuralLayers.PINK
        layer = VGroup()
        neurons = VGroup()  # 뉴런들만 따로 그룹화
        for i in range(neuron_count):
            neuron = Circle(
                radius=radius,
                color=color,
                fill_color=fill_color,
                fill_opacity=0,
            )
            number = Text(str(i), color=NeuralLayers.NUMBER_COLOR, font_size=36)
            number.move_to(neuron.get_center())
            # 확률값 표시용 텍스트 (초기값 0.0)
            prob = DecimalNumber(
                0.0,
                num_decimal_places=3,
                color=NeuralLayers.PROB_COLOR,
                font_size=24,
            ).next_to(neuron, DOWN, buff=0.1)
            neurons.add(neuron)
            layer.add(VGroup(neuron, number, prob))

        layer.arrange(RIGHT, buff=radius * 0.8)
        return layer, neurons  # 뉴런 그룹도 함께 반환

    @staticmethod
    def create_layer_connections(
        input_positions,
        target_positions,
        weights,
        pixel_values=None,
        opacity_range=(0.1, 0.8),
    ):
        """레이어 간 연결선 생성"""
        connections = VGroup()
        w_min, w_max = weights.min(), weights.max()

        if pixel_values is None:
            pixel_values = np.ones(len(input_positions))

        for i, start_pos in enumerate(input_positions):
            if pixel_values[i] == 0:
                continue

            # 정규화된 픽셀 강도 계산
            pixel_intensity = (
                pixel_values[i] / 255.0 if pixel_values[i] > 1 else pixel_values[i]
            )

            for j, end_pos in enumerate(target_positions):
                # 가중치에 기반한 불투명도 계산
                weight_opacity = NeuralLayers._normalize_opacity(
                    weights[i, j], w_min, w_max, opacity_range
                )
                final_opacity = weight_opacity * pixel_intensity

                line = Line(
                    start=start_pos,
                    end=end_pos,
                    stroke_width=0.15,
                    stroke_color=YELLOW,
                    stroke_opacity=final_opacity,
                )
                connections.add(line)

        return connections

    @staticmethod
    def _normalize_opacity(weight, w_min, w_max, opacity_range=(0.1, 0.8)):
        """가중치값을 불투명도로 정규화"""
        if w_max == w_min:
            return opacity_range[0]
        return opacity_range[0] + (opacity_range[1] - opacity_range[0]) * (
            weight - w_min
        ) / (w_max - w_min)

    @staticmethod
    def update_layer_activation(layer, activation_values, opacity_range=(0.1, 0.9)):
        """뉴런 층의 활성화 값을 기반으로 불투명도 업데이트"""
        for neuron, activation in zip(layer, activation_values):
            neuron.set_fill(opacity=float(activation))

    @staticmethod
    def update_output_probabilities(output_layer, probabilities):
        """출력층의 확률값 업데이트"""
        for i, prob in enumerate(probabilities):
            # VGroup의 세 번째 요소(확률값)를 업데이트
            output_layer[i][2].set_value(float(prob))
