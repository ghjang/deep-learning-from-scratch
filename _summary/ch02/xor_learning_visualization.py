from manim import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os


# 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class XORLearningAnimation(Scene):
    def construct(self):
        # 학습 과정 로드
        if not os.path.exists("xor_weight_history.pkl"):
            self.add(
                Text(
                    "가중치 히스토리 파일이 없습니다. 먼저 xor.py를 실행하세요."
                ).scale(0.5)
            )
            self.wait(2)
            return

        with open("xor_weight_history.pkl", "rb") as f:
            weight_history = pickle.load(f)

        # 좌표 평면 설정 - 오른쪽으로 이동
        axes = Axes(
            x_range=[-0.5, 1.5, 0.5],
            y_range=[-0.5, 1.5, 0.5],
            axis_config={"include_numbers": False},
            x_length=5.5,  # 약간 작게
            y_length=5.5,  # 약간 작게
        ).shift(RIGHT * 2)

        axes_label = axes.get_axis_labels(x_label="x1", y_label="x2")

        # z_index를 높게 설정하여 이미지 위에 표시되도록 함
        axes.set_z_index(2)
        axes_label.set_z_index(2)

        # XOR 데이터 포인트
        data_points = [
            (0, 0, BLUE),  # 0 (파랑)
            (0, 1, RED),  # 1 (빨강)
            (1, 0, RED),  # 1 (빨강)
            (1, 1, BLUE),  # 0 (파랑)
        ]

        # 1. 먼저 좌표축 표시
        self.play(Create(axes), Create(axes_label))

        # 에폭 카운터 - 좌측에 배치 (더 많은 공간 확보)
        epoch_counter = (
            Text("Epoch: 0", font_size=30).to_edge(LEFT, buff=1.0).shift(UP * 0)
        )
        epoch_counter.set_z_index(3)  # 항상 최상위에 보이도록
        self.play(Write(epoch_counter))

        # 2. 데이터 포인트 표시 및 높은 z_index 설정
        dots = [
            Dot(axes.c2p(x, y), color=color, radius=0.1) for x, y, color in data_points
        ]

        # 레이블 색상 변경 - 각 점과 동일한 색상으로 설정
        dot_labels = [
            Text(
                f"({x}, {y}) = {1 if color==RED else 0}", font_size=24, color=color
            ).next_to(axes.c2p(x, y), direction=UP if y > 0.5 else DOWN)
            for x, y, color in data_points
        ]

        # 점과 레이블을 항상 위에 보이게 설정
        for dot in dots:
            dot.set_z_index(3)
        for label in dot_labels:
            label.set_z_index(3)

        self.play(*[FadeIn(dot) for dot in dots])
        self.play(*[FadeIn(label) for label in dot_labels])

        # 3. 결정 경계 시각화 준비
        mesh_size = 100
        x_min, x_max = -0.5, 1.5
        y_min, y_max = -0.5, 1.5
        x_mesh, y_mesh = np.meshgrid(
            np.linspace(x_min, x_max, mesh_size), np.linspace(y_min, y_max, mesh_size)
        )
        mesh_points = np.c_[x_mesh.ravel(), y_mesh.ravel()]

        # 이전 결정 경계 저장용
        prev_decision_boundary = None

        # 에포크 선택 - 에폭 5000 이후 더 큰 간격으로 선택
        selected_indices = []

        # 초반 (0~1000 에포크): 200 간격
        selected_indices.extend(
            [
                i
                for i, weights in enumerate(weight_history)
                if weights["epoch"] <= 1000 and weights["epoch"] % 200 == 0
            ]
        )

        # 중반 (1000~5000 에포크): 500 간격
        selected_indices.extend(
            [
                i
                for i, weights in enumerate(weight_history)
                if 1000 < weights["epoch"] <= 5000 and weights["epoch"] % 500 == 0
            ]
        )

        # 후반 (5000~10000 에포크): 1000 간격 - 수정된 부분
        selected_indices.extend(
            [
                i
                for i, weights in enumerate(weight_history)
                if weights["epoch"] > 5000 and weights["epoch"] % 1000 == 0
            ]
        )

        # 선택된 인덱스에 대해서만 애니메이션 생성
        print(f"총 {len(selected_indices)}개의 프레임을 생성합니다.")

        # 4. 학습 과정에 따른 결정 경계 변화 애니메이션
        for idx in selected_indices:
            weights = weight_history[idx]
            W1, b1 = weights["W1"], weights["b1"]
            W2, b2 = weights["W2"], weights["b2"]
            epoch = weights["epoch"]

            # 결정 경계 계산
            hidden_output = sigmoid(np.dot(mesh_points, W1) + b1)
            output_layer = sigmoid(np.dot(hidden_output, W2) + b2)
            predictions = output_layer.reshape(mesh_size, mesh_size)

            # 결정 경계 0.5 기준 (0.5 이상이면 1, 미만이면 0)
            contour_matrix = (predictions >= 0.5).astype(int)

            # 결정 경계를 이미지로 변환
            img = ImageMobject(np.uint8(contour_matrix * 255))  # 0(검은색), 255(흰색)
            img.set_height(axes.get_height())
            img.set_width(axes.get_width())
            img.move_to(axes.get_center())

            # 투명도 설정
            img.set_opacity(0.4)

            # z-index 낮게 설정 (축과 점 아래에 보이도록)
            img.set_z_index(1)

            # 에폭 카운터 업데이트
            new_epoch_counter = (
                Text(f"Epoch: {epoch}", font_size=30)
                .to_edge(LEFT, buff=1.0)
                .shift(UP * 0)
            )
            new_epoch_counter.set_z_index(3)  # 최상위에 보이도록

            # 이미지 애니메이션
            if prev_decision_boundary is None:
                self.play(FadeIn(img), Transform(epoch_counter, new_epoch_counter))
            else:
                self.play(
                    Transform(prev_decision_boundary, img),
                    Transform(epoch_counter, new_epoch_counter),
                )

            prev_decision_boundary = img

            # 애니메이션 속도 조절 (주요 변화 시점에서는 천천히)
            if epoch == 0 or epoch == 9000:
                self.wait(0.8)  # 처음과 끝에는 조금 더 오래 표시
            else:
                self.wait(0.5)  # 중간 부분은 약간 빠르게

        # 최종 학습 결과 표시 (영어로 변경하고 위치 조정)
        final_text = Text("XOR Learning Completed!", font_size=36, color=GREEN)
        final_text.move_to(epoch_counter.get_center()).shift(RIGHT)
        final_text.set_z_index(3)  # 최상위에 보이도록

        self.play(Transform(epoch_counter, final_text))
        self.wait(3)
