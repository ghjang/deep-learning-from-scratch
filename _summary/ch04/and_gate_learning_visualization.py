import sys
import os
import numpy as np
import operator as op
from typing import TypedDict
from manim import *

# 모듈 검색 경로에 _neuro 디렉토리 추가
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../_neuro"))
)
from neural_net import NeuralNet as NN
from activation import sigmoid
from optimizer import ProgressData

from m_and_gate_logic_table import MANDGateLogicTable
from m_single_layper_perceptron import MSingleLayerPerceptron


class ProgressDataSnapshot(TypedDict):
    """'퍼셉트론 1개'를 사용하여 'AND 게이트 학습'시 진행 상태 저장을 위한 타입"""

    epoch: int  # 현재 에포크 번호
    loss: float  # 현재 에포크의 손실값
    learning_rate: float  # 현재 학습률
    weights: tuple[float, float]  # 현재 가중치
    bias: float  # 현재 바이어스


class AndGateLearningVisualization(Scene):
    # 학습 관련 상수
    LEARNING_RATE = 0.25  # 클래스 상수 값을 실제 사용할 값으로 변경
    EPOCHS = 10000  # 시각화를 위한 에포크 수
    INTERVAL = 20  # 에포크 간격

    TRUTH_TABLE_ZOOM_OUT_SCALE = 0.4
    PERCEPTRON_NETWORK_ZOOM_OUT_SCALE = 0.55

    # 좌표평면 관련 상수
    PLANE_SCALE = 1.9
    X_RANGE = [-4.5, 4.5]
    Y_RANGE = [-4.5, 4.5]
    GRAPH_X_RANGE = [-2.5, 2.5]
    GRAPH_Y_RANGE = [-2.5, 2.5]

    # 시각적 요소 관련 상수
    POINT_RADIUS = 0.1
    LABEL_SCALE = 0.7
    FONT_SIZE = 24
    FILL_OPACITY = 0.15
    LINE_COLOR = YELLOW
    REGION_COLOR = RED
    EQUATION_SCALE = 0.5
    TEXT_Z_INDEX = 10
    EPSILON = 1e-10  # 0에 가까운 값 판단용 상수

    # 색상 관련 상수
    DOT_COLORS = {0: RED, 1: GREEN}
    TEXT_COLOR = "#FFA07A"  # Light Salmon color
    TEXT_LABEL_COLOR = "#FFA07A"  # Light Salmon
    PARAM_COLORS = {
        "w1": "#FF6B6B",  # 붉은 계열
        "w2": "#4ECDC4",  # 청록 계열
        "b": "#FFD93D",  # 노란 계열
    }
    CORRECT_COLOR = GREEN  # 예측 성공 색상
    WRONG_COLOR = RED  # 예측 실패 색상

    # 히스토리 필터링 상수
    LOSS_THRESHOLD = 0.01  # 손실 값의 최소 변화량

    epoch_progress_history: list[ProgressDataSnapshot]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 에포크 데이터를 저장할 리스트 초기화
        self.epoch_progress_history = []

        # 데이터 포인트와 레이블 저장
        self.data_points = []
        self.data_dots = {}
        self.data_labels = {}

    def setup(self):
        self._append_training_epoch_progress_data(
            {
                "epoch": 0,
                "loss": 0,
                "learning_rate": self.LEARNING_RATE,
                "model_ref": None,
            }
        )

        self._setup_AND_gate_model()

    def construct(self):
        self.next_section("Initial Setup", skip_animations=False)

        # AND 게이트 테이블 생성 및 인스턴스 변수로 저장 (다른 메서드에서 접근 가능하도록)
        self.and_gate_table = MANDGateLogicTable().set_z_index(10)
        self.add(self.and_gate_table)
        self.wait(2)
        self.play(
            self.and_gate_table.animate.scale(self.TRUTH_TABLE_ZOOM_OUT_SCALE).to_edge(
                DL
            )
        )

        perceptron_network = MSingleLayerPerceptron().set_z_index(10).shift(RIGHT * 2.2)
        self.add(perceptron_network)
        self.wait(2)
        self.play(
            perceptron_network.animate.scale(
                self.PERCEPTRON_NETWORK_ZOOM_OUT_SCALE
            ).to_edge(DR)
        )

        self.next_section("Show Data Points", skip_animations=False)

        # 'x1, x2' 데이터 표시를 위한 플레인 생성
        number_plane = self._create_number_plane()
        self.add(number_plane)

        # AND 게이트 입력값 4개에 대한 포인트 생성 및 표시
        self._create_and_display_data_points()

        self.wait()

        # 트레이닝 히스토리 표시 섹션
        self.next_section("Show Training History", skip_animations=False)

        # 트레이닝 히스토리 데이터를 사용하여 결정 경계선 표시
        self._display_training_history(number_plane)

        # final wait
        self.wait(3)

    def _create_number_plane(self) -> NumberPlane:
        """좌표 평면을 생성합니다."""
        return NumberPlane(
            axis_config={"stroke_opacity": 0.5},
            background_line_style={"stroke_opacity": 0.25},
        ).scale(self.PLANE_SCALE)

    def _create_and_display_data_points(self) -> None:
        """AND 게이트 입력값 4개에 대한 포인트와 레이블을 생성하고 표시합니다."""
        self.data_points = [
            {"coords": [0, 0], "output": 0, "label_direction": DOWN},
            {"coords": [0, 1], "output": 0, "label_direction": UP},
            {"coords": [1, 0], "output": 0, "label_direction": DOWN},
            {"coords": [1, 1], "output": 1, "label_direction": UP},
        ]

        for point in self.data_points:
            x, y = point["coords"]
            output = point["output"]
            direction = point["label_direction"]

            # 좌표평면 상의 위치로 변환
            position = self.get_plane_coords(x, y)

            # 점 생성 (초기에는 보이지 않게 - 불투명도 0)
            dot = Dot(
                position,
                color=self.WRONG_COLOR,
                radius=self.POINT_RADIUS,
                fill_opacity=0,
            )
            self.data_dots[(x, y)] = dot
            self.add(dot)

            # 레이블 추가 (초기에는 보이지 않게 - 불투명도 0)
            label = (
                MathTex(f"({x},{y})")
                .next_to(dot, direction, buff=0.1)
                .scale(self.LABEL_SCALE)
                .set_opacity(0)
            )
            self.data_labels[(x, y)] = label
            self.add(label)

    def get_plane_coords(self, x: float, y: float) -> np.ndarray:
        """X, Y 좌표값을 화면 좌표로 변환합니다."""
        # 임시적으로 화면 중앙에서 스케일을 적용하여 좌표 계산
        return np.array([x * self.PLANE_SCALE, y * self.PLANE_SCALE, 0])

    def _format_float(self, value: float) -> str:
        """소수점 이하 숫자를 적절히 표시한 문자열 반환"""
        # 총 에러(loss)일 경우 더 많은 소수점 자리수 표시
        if abs(value) < 0.01:
            # 매우 작은 값은 6자리까지 표시 (0.000123 같은 값)
            return f"{value:.6f}"
        elif abs(value) < 0.1:
            # 작은 값은 4자리까지 표시 (0.0123 같은 값)
            return f"{value:.4f}"
        else:
            # 일반적인 값은 기존과 동일하게 2자리까지 표시하고 불필요한 0 제거
            return f"{value:.2f}".rstrip("0").rstrip(".")

    def _create_epoch_text(self, epoch_data: ProgressDataSnapshot) -> VGroup:
        """에포크 데이터를 표시하는 텍스트 그룹을 생성합니다."""
        # 에포크 0인지 확인
        is_epoch_zero = epoch_data["epoch"] == 0

        # 모든 텍스트 요소를 하나의 리스트로 생성
        text_elements = [
            # 기본 정보 (항상 표시)
            MathTex(
                f"\\textrm{{Epoch: }}{epoch_data["epoch"]}",
                font_size=self.FONT_SIZE,
                color=self.TEXT_LABEL_COLOR,
            ),
            MathTex(
                f"\\textrm{{Learning Rate: }}{epoch_data["learning_rate"]}",
                font_size=self.FONT_SIZE,
                color=self.TEXT_LABEL_COLOR,
            ),
            MathTex(
                f"\\textrm{{Weights: }}[{self._format_float(epoch_data["weights"][0])}, {self._format_float(epoch_data["weights"][1])}]",
                font_size=self.FONT_SIZE,
                color=self.TEXT_LABEL_COLOR,
            ),
            MathTex(
                f"\\textrm{{Bias: }}{self._format_float(epoch_data["bias"])}",
                font_size=self.FONT_SIZE,
                color=self.TEXT_LABEL_COLOR,
            ),
        ]

        # 추가 정보 (에포크 0에서는 숨김)
        additional_texts = [
            # Loss 텍스트
            MathTex(
                f"\\textrm{{Loss: }}{self._format_float(epoch_data["loss"])}",
                font_size=self.FONT_SIZE,
                color=self.TEXT_LABEL_COLOR,
            ),
            # 가중치와 바이어스 업데이트 일반식
            MathTex(
                "\\Delta w_i = \\eta \\cdot \\nabla L \\cdot x_i,\\,\\,\\,\\Delta b = \\eta \\cdot \\nabla L",
                font_size=self.FONT_SIZE,
                color=self.TEXT_LABEL_COLOR,
            ),
            # 일반화된 함수식
            MathTex(
                "y = \\left\\{\\frac{-(w_1)}{w_2}\\right\\}\\cdot x + \\left\\{\\frac{-(b)}{w_2}\\right\\}",
                font_size=self.FONT_SIZE,
                color=self.TEXT_LABEL_COLOR,
            ),
        ]

        # 에포크 0에서는 추가 정보 텍스트를 숨기고, 그 외에는 모두 표시
        if is_epoch_zero:
            # 에포크 0에서는 기본 정보만 표시
            all_texts = text_elements
        else:
            # 나머지 에포크에서는 모든 정보 표시
            all_texts = text_elements + additional_texts

        # 모든 텍스트를 그룹으로 결합하고 좌측 정렬로 배치
        text_group = VGroup(*all_texts).arrange(DOWN, aligned_edge=LEFT).to_corner(UL)
        text_group.set_z_index(self.TEXT_Z_INDEX)

        return text_group

    def _create_equation_text(
        self, weights: tuple[float, float], bias: float
    ) -> MathTex:
        """가중치와 바이어스로부터 함수식 레이텍을 생성합니다."""
        w1, w2 = weights
        b = bias

        if abs(w2) < self.EPSILON:  # 수직선의 경우
            if abs(w1) < self.EPSILON:
                return None

            # x = -b/w1 형태
            equation = MathTex(
                "x = \\left\\{\\frac{-("
                + self._format_float(b)
                + ")}{"
                + self._format_float(w1)
                + "}\\right\\}",
                color=self.LINE_COLOR,
            )

            # 수직선이므로 간소화된 수식 추가
            x_val = -b / w1
            simplified_eq = MathTex(
                f"x = {self._format_float(x_val)}",
                color=self.LINE_COLOR,
            )

            # 두 수식을 그룹으로 묶음
            equation_group = VGroup(equation, simplified_eq).arrange(DOWN, buff=0.2)

        else:
            # y = (-w1/w2)x + (-b/w2) 형태
            equation = MathTex(
                "y = "
                "\\left\\{\\frac{-("
                + self._format_float(w1)
                + ")}{"
                + self._format_float(w2)
                + "}\\right\\}"
                "\\cdot x"
                "+\\left\\{\\frac{-("
                + self._format_float(b)
                + ")}{"
                + self._format_float(w2)
                + "}\\right\\}",
                color=self.LINE_COLOR,
            )

            # 실제 값으로 계산된 간소화된 수식 추가
            slope = -w1 / w2
            y_intercept = -b / w2

            sign = "+" if y_intercept >= 0 else "-"
            simplified_eq = MathTex(
                f"y = {self._format_float(slope)} \\cdot x {sign} {self._format_float(abs(y_intercept))}",
                color=self.LINE_COLOR,
            )

            # 두 수식을 그룹으로 묶음
            equation_group = VGroup(equation, simplified_eq).arrange(DOWN, buff=0.5)

        # 방정식을 우상단에 배치
        equation_group.scale(self.EQUATION_SCALE)
        equation_group.to_corner(UR, buff=0.5)  # 우상단(Upper Right)에 배치
        equation_group.set_z_index(self.TEXT_Z_INDEX)

        return equation_group

    def _get_safe_x_position(self, x_val):
        """좌표평면 범위 내의 안전한 X 좌표값을 반환합니다."""
        return max(min(x_val, self.X_RANGE[1] - 1), self.X_RANGE[0] + 1)

    def _calculate_equation_position(self, w1, w2, b):
        """함수식 위치를 계산합니다."""
        slope = -w1 / w2
        y_intercept = -b / w2

        if abs(slope) < 1e-6:
            return 0

        target_y = -1
        x_at_target_y = (target_y - y_intercept) / slope
        return self._get_safe_x_position(x_at_target_y)

    def _draw_decision_boundary_and_region(
        self,
        number_plane: NumberPlane,
        weights: tuple[float, float],
        bias: float,
        loss: float,
    ) -> tuple[VMobject, VMobject, VMobject]:
        """결정 경계선과 아래쪽 영역, 함수식을 그립니다."""
        w1, w2 = weights
        b = bias

        print(f"Drawing decision boundary: w1={w1}, w2={w2}, b={b}, loss={loss}")

        # 모든 가중치와 바이어스가 0인 경우 - y = 0 수평선
        if abs(w1) < self.EPSILON and abs(w2) < self.EPSILON and abs(b) < self.EPSILON:
            # y = 0 형태의 수평선
            line = number_plane.plot_line_graph(
                x_values=self.X_RANGE,
                y_values=[0, 0],  # y = 0
                line_color=self.LINE_COLOR,
                add_vertex_dots=False,
            )

            # 수평선 아래 영역을 채움
            points = [
                number_plane.c2p(self.X_RANGE[0], 0),  # 왼쪽 경계선 점
                number_plane.c2p(self.X_RANGE[1], 0),  # 오른쪽 경계선 점
                number_plane.c2p(
                    self.X_RANGE[1], self.Y_RANGE[0]
                ),  # 오른쪽 아래 모서리
                number_plane.c2p(self.X_RANGE[0], self.Y_RANGE[0]),  # 왼쪽 아래 모서리
            ]

            region = Polygon(
                *points,
                fill_color=self.REGION_COLOR,
                fill_opacity=self.FILL_OPACITY,
                stroke_width=0,
            )

            # 함수식은 y = 0, 우상단에 표시
            equation = VGroup(
                MathTex("y = 0", color=self.LINE_COLOR),
                MathTex(
                    "y = 0", color=self.LINE_COLOR
                ),  # 간소화된 형태가 이미 단순하므로 동일하게 표시
            ).arrange(DOWN, buff=0.2)

            equation.scale(self.EQUATION_SCALE)
            equation.to_corner(UR, buff=0.5)  # 우상단에 배치
            equation.set_z_index(self.TEXT_Z_INDEX)

            # z-index 설정
            region.set_z_index(-1)
            line.set_z_index(0)

            return line, region, equation

        # 특수 케이스 처리: 수직선 (w2 = 0, w1 ≠ 0)
        if abs(w2) < self.EPSILON:
            # x = -b/w1 형태의 수직선
            x_val = -b / w1
            line = number_plane.plot_line_graph(
                x_values=[x_val, x_val],
                y_values=self.Y_RANGE,
                line_color=self.LINE_COLOR,
                add_vertex_dots=False,
            )

            # 수직선의 왼쪽 또는 오른쪽 영역을 채움
            points = [
                number_plane.c2p(x_val, self.Y_RANGE[0]),  # 선의 아래 점
                number_plane.c2p(x_val, self.Y_RANGE[1]),  # 선의 위 점
                number_plane.c2p(self.X_RANGE[0], self.Y_RANGE[1]),  # 왼쪽 위 모서리
                number_plane.c2p(self.X_RANGE[0], self.Y_RANGE[0]),  # 왼쪽 아래 모서리
            ]

        else:
            # 일반적인 경우: y = (-w1*x - b) / w2
            slope = -w1 / w2
            y_intercept = -b / w2

            # 직선 그리기
            line = number_plane.plot_line_graph(
                x_values=self.X_RANGE,
                y_values=[
                    self.X_RANGE[0] * slope + y_intercept,
                    self.X_RANGE[1] * slope + y_intercept,
                ],
                line_color=YELLOW,
                add_vertex_dots=False,
            )

            # 아래쪽 영역을 채우기 위한 점들
            points = [
                number_plane.c2p(
                    self.X_RANGE[0], self.X_RANGE[0] * slope + y_intercept
                ),  # 왼쪽 경계선 점
                number_plane.c2p(
                    self.X_RANGE[1], self.X_RANGE[1] * slope + y_intercept
                ),  # 오른쪽 경계선 점
                number_plane.c2p(
                    self.X_RANGE[1], self.Y_RANGE[0]
                ),  # 오른쪽 아래 모서리
                number_plane.c2p(self.X_RANGE[0], self.Y_RANGE[0]),  # 왼쪽 아래 모서리
            ]

        # 영역 생성
        region = Polygon(
            *points,
            fill_color=self.REGION_COLOR,
            fill_opacity=self.FILL_OPACITY,
            stroke_width=0,
        )

        # 함수식 생성 - 우상단에 표시
        equation = self._create_equation_text(weights, bias)

        # z-index 설정
        region.set_z_index(-1)
        line.set_z_index(0)
        if equation:
            equation.set_z_index(1)

        return line, region, equation

    def _predict_AND_gate_output(
        self,
        weights: tuple[float, float],
        bias: float,
        loss: float,
        inputs: tuple[float, float],
        update_table: bool = False,
    ) -> int:
        """주어진 가중치와 바이어스로 예측 결과를 반환합니다."""

        # '1개 레이어'로만 구성된 모델을 'NN.load' 메쏘드를 이용하여 간단하게 생성
        model = (
            NN.create()
            .layer(weights=np.array(weights).reshape(-1, 1), biases=np.array([bias]))
            .activation(sigmoid)
        )

        # 입력에 대한 예측
        # 시그모이드 출력을 0/1로 변환
        thresholds = {
            (0, 0): (0, 0.1, op.le),
            (0, 1): (0, 0.1, op.le),
            (1, 0): (0, 0.1, op.le),
            (1, 1): (1, 0.9, op.ge),
        }

        target_value, threshold, comparator = thresholds[inputs]

        output = model.forward(np.array(inputs))
        output = output[0, 0]  # '1개의 출력값'만을 갖는 2D 배열 요소를 스칼라로 변환
        org_approx_output = output

        print(f"loss: {loss}, output: {output}, threshold: {threshold}")

        # loss가 충분히 작은 경우에는 threshold값에 도달한 것으로 간주한다.
        SMALL_LOSS_THRESHOLD = 0.001
        output = threshold if loss <= SMALL_LOSS_THRESHOLD else output

        predicted = target_value if comparator(output, threshold) else 1 - target_value

        # AND 게이트 테이블 업데이트 (요청된 경우)
        if update_table and hasattr(self, "and_gate_table"):
            # 입력에 해당하는 행 인덱스 매핑 (2 ~ 5는 데이터 행에 해당)
            row_index_map = {
                (0, 0): 2,  # 첫 번째 데이터 행
                (0, 1): 3,  # 두 번째 데이터 행
                (1, 0): 4,  # 세 번째 데이터 행
                (1, 1): 5,  # 네 번째 데이터 행
            }
            row_idx = row_index_map[inputs]

            # 결과에 따라 녹색/빨간색 마크 선택
            mark_type = "green_circle" if predicted == target_value else "red_circle"

            # 예측 값을 테이블 셀에 업데이트
            org_approx_output_tex = f"\\approx {org_approx_output:.6f}"
            self.and_gate_table.update_result_cell(
                row_idx,
                org_approx_output_tex,
                scene=self,
                mark_type=mark_type,
                mark_buff=0.2,
                current_scale=self.TRUTH_TABLE_ZOOM_OUT_SCALE,
            )

        return predicted

    def _update_data_point_colors(self, weights, bias, loss) -> list[Animation]:
        """현재 가중치와 바이어스로 예측 결과에 따라 데이터 포인트 색상을 업데이트합니다."""
        animations = []

        for point in self.data_points:
            x, y = point["coords"]
            target = point["output"]
            prediction = self._predict_AND_gate_output(
                weights, bias, loss, (x, y), update_table=True
            )

            # 예측 성공/실패에 따른 색상 결정
            new_color = self.CORRECT_COLOR if prediction == target else self.WRONG_COLOR

            # 색상 변경 애니메이션 생성
            dot = self.data_dots[(x, y)]
            animations.append(dot.animate.set_color(new_color))

        return animations

    def _show_data_points(self):
        """데이터 포인트와 레이블을 불투명도 1.0으로 설정하여 보이게 합니다."""
        animations = []

        for coords, dot in self.data_dots.items():
            animations.append(dot.animate.set_fill(opacity=1))

        for coords, label in self.data_labels.items():
            animations.append(label.animate.set_opacity(1))

        return animations

    def _display_training_history(self, number_plane: NumberPlane) -> None:
        """트레이닝 히스토리 데이터를 사용하여 결정 경계선 변화를 애니메이션으로 표시합니다."""

        # loss 값 차이가 LOSS_THRESHOLD 미만인 구간에서는 첫 번째 값만 남기고 필터링
        filtered_history = []
        prev_loss = None
        prev_index = -1

        for i, data in enumerate(self.epoch_progress_history):
            current_loss = data["loss"]

            # 첫 번째 항목은 무조건 포함
            if i == 0 or prev_loss is None:
                filtered_history.append(data)
                prev_loss = current_loss
                prev_index = i
                continue

            # loss 차이가 LOSS_THRESHOLD 이상이면 해당 항목 포함
            if abs(current_loss - prev_loss) >= self.LOSS_THRESHOLD:
                filtered_history.append(data)
                prev_loss = current_loss
                prev_index = i
            # 마지막 히스토리 항목은 포함
            elif i == len(self.epoch_progress_history) - 1 and prev_index != i:
                filtered_history.append(data)

        print(
            f"원본 히스토리 길이: {len(self.epoch_progress_history)}, 필터링 후 길이: {len(filtered_history)}"
        )

        # 필터링된 히스토리 사용
        history_len = len(filtered_history)
        if history_len <= 0:
            return

        # 초기 요소들 생성 (에포크 0)
        first_data = filtered_history[0]

        # 직선과 영역은 생성하되 화면에 표시하지 않음
        current_line, current_region, current_equation = (
            self._draw_decision_boundary_and_region(
                number_plane,
                first_data["weights"],
                first_data["bias"],
                first_data["loss"],
            )
        )
        current_text = self._create_epoch_text(first_data)

        # 에포크 0에서는 텍스트만 표시 (직선과 영역은 표시하지 않음)
        self.play(FadeIn(current_text))
        self.wait()

        # 에포크 1부터는 시각적 요소 모두 표시
        if history_len > 1:
            epoch_data = filtered_history[1]
            new_line, new_region, new_equation = (
                self._draw_decision_boundary_and_region(
                    number_plane,
                    epoch_data["weights"],
                    epoch_data["bias"],
                    epoch_data["loss"],
                )
            )
            new_text = self._create_epoch_text(epoch_data)

            # 데이터 포인트와 레이블을 보이게 만듦 (직접 설정)
            for coords, dot in self.data_dots.items():
                dot.set_fill(opacity=1)  # 직접 불투명도 설정

            for coords, label in self.data_labels.items():
                label.set_opacity(1)  # 직접 불투명도 설정

            # 데이터 포인트 색상 업데이트
            for point in self.data_points:
                x, y = point["coords"]
                target = point["output"]
                # NeuralNet을 사용한 예측 및 테이블 업데이트
                prediction = self._predict_AND_gate_output(
                    epoch_data["weights"],
                    epoch_data["bias"],
                    epoch_data["loss"],
                    (x, y),
                    update_table=True,  # 테이블 업데이트 활성화
                )
                # 색상 설정
                new_color = (
                    self.CORRECT_COLOR if prediction == target else self.WRONG_COLOR
                )
                self.data_dots[(x, y)].set_color(new_color)  # 직접 색상 설정

            # 에포크 1의 모든 요소 표시
            animations = [
                FadeIn(new_region),
                FadeIn(new_line),
                ReplacementTransform(current_text, new_text),
            ]

            if new_equation:
                animations.append(FadeIn(new_equation))

            # 모든 데이터 포인트와 레이블을 애니메이션으로 표시
            animations.append(
                AnimationGroup(
                    *[FadeIn(dot) for dot in self.data_dots.values()]
                    + [FadeIn(label) for label in self.data_labels.values()]
                )
            )

            self.play(*animations)
            self.wait()

            # 현재 객체들 업데이트
            current_line = new_line
            current_region = new_region
            current_equation = new_equation
            current_text = new_text

        # 두 번째 에포크부터는 ReplacementTransform 사용
        for i in range(2, history_len):
            epoch_data = filtered_history[i]
            new_line, new_region, new_equation = (
                self._draw_decision_boundary_and_region(
                    number_plane,
                    epoch_data["weights"],
                    epoch_data["bias"],
                    epoch_data["loss"],
                )
            )
            new_text = self._create_epoch_text(epoch_data)

            # 데이터 포인트 색상 업데이트 애니메이션
            color_animations = self._update_data_point_colors(
                epoch_data["weights"], epoch_data["bias"], epoch_data["loss"]
            )

            if new_line and current_line:
                animations = [
                    ReplacementTransform(current_region, new_region),
                    ReplacementTransform(current_line, new_line),
                    ReplacementTransform(current_text, new_text),
                ]
                animations.extend(color_animations)  # 색상 업데이트

                if current_equation and new_equation:
                    animations.append(
                        ReplacementTransform(current_equation, new_equation)
                    )
                elif new_equation:
                    animations.append(FadeIn(new_equation))

                self.play(*animations)
                self.wait()

                # 현재 객체들 업데이트
                current_line = new_line
                current_region = new_region
                current_equation = new_equation
                current_text = new_text

                # 모든 점이 녹색인지 검사 (모든 예측이 정확한지 확인)
                all_correct = True
                for point in self.data_points:
                    x, y = point["coords"]
                    target = point["output"]
                    prediction = self._predict_AND_gate_output(
                        epoch_data["weights"],
                        epoch_data["bias"],
                        epoch_data["loss"],
                        (x, y),
                    )
                    if prediction != target:  # 하나라도 불일치하면 False
                        all_correct = False
                        break

                # 모든 예측이 정확하면 나머지 에포크 표시 건너뛰기
                if all_correct:
                    print(
                        f"에포크 {epoch_data['epoch']}에서 모든 예측이 정확합니다. 학습 완료!"
                    )
                    break  # 학습이 완료되어 나머지 에포크는 표시하지 않음

    def _setup_AND_gate_model(self):
        """초기 설정 및 데이터 학습 수행"""
        # AND 게이트 학습 데이터
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [0], [0], [1]])

        # 모델 생성 및 설정
        # ; 각 레이어의 초기 가중치와 바이어스는 layer() 메서드 호출 시 자동으로 설정됨.
        model = (
            NN.create()
            .layer(1)
            .activation(sigmoid)
            .learning_rate(self.LEARNING_RATE)
            .batch_size(x.shape[0])
            .epochs(self.EPOCHS)
            .loss("mse")
            .optimizer("gradient_descent")
            .verbose(interval=self.INTERVAL)
            .callback(self._append_training_epoch_progress_data)
        )

        # 모델 학습
        history = model.fit(x, y)

        # 학습 히스토리 결과 출력
        print(
            f"\n학습 기록 수집 완료: {len(self.epoch_progress_history)}개의 에포크 데이터"
        )

        # 결과 테스트 및 출력
        self._print_trained_model_test_results(model, x, y)

    def _print_trained_model_test_results(self, model, x, y):
        """학습 결과를 테스트하고 콘솔에 출력합니다."""
        print("\n결과:")
        predictions = model.predict(x)
        for i in range(len(x)):
            inputs = x[i]
            target = y[i][0]
            pred = predictions[i][0]
            pred_class = 1 if pred > 0.5 else 0
            print(f"입력: {inputs}, 출력: {pred:.4f} ({pred_class}), 정답: {target}")

        # 학습된 가중치와 바이어스 출력
        weights = model.layers[0].weights.flatten()
        bias = model.layers[0].biases[0]
        print(f"\n학습된 가중치: {weights}")
        print(f"학습된 바이어스: {bias}")

    def _append_training_epoch_progress_data(
        self, cur_epoch_progress_data: ProgressData
    ) -> None:
        """에포크 데이터를 기록하고 콘솔에 출력합니다."""

        progress_data: ProgressDataSnapshot = {
            "epoch": cur_epoch_progress_data["epoch"],
            "loss": cur_epoch_progress_data["loss"],
            "learning_rate": cur_epoch_progress_data["learning_rate"],
            "weights": (0, 0),
            "bias": 0,
        }

        if cur_epoch_progress_data["model_ref"] is not None:
            model = cur_epoch_progress_data["model_ref"]

            # copy weights and bias
            progress_data["weights"] = tuple(model.layers[0].weights.flatten())
            progress_data["bias"] = model.layers[0].biases[0]

        self.epoch_progress_history.append(progress_data)

        # 현재 에포크 정보 출력
        print(
            f"Epoch {cur_epoch_progress_data['epoch']}: Loss {cur_epoch_progress_data['loss']:.6f}"
        )

        print(f"Weights: {progress_data['weights']}," f" Bias: {progress_data['bias']}")
