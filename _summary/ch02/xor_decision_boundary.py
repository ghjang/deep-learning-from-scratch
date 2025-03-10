from manim import *
import numpy as np
import matplotlib.pyplot as plt


class XORBoundaryAnimation(Scene):
    def construct(self):
        # 좌표 평면 설정 - 숫자 제거
        axes = Axes(
            x_range=[-0.5, 1.5, 0.5],
            y_range=[-0.5, 1.5, 0.5],
            axis_config={"include_numbers": False},  # 눈금 숫자 제거
        )
        self.play(Create(axes))

        # XOR 데이터 포인트
        data_points = [
            (0, 0, BLUE),
            (0, 1, RED),
            (1, 0, RED),
            (1, 1, BLUE),
        ]

        # 점 추가
        dots = [Dot(axes.c2p(x, y), color=color) for x, y, color in data_points]
        for dot in dots:
            self.play(FadeIn(dot), run_time=0.5)

        # 초기 선형 분리 시도 (직선 하나로는 XOR을 해결 못함)
        fail_line = axes.plot(lambda x: -x + 0.5, color=YELLOW)  # 변경된 부분
        self.play(Create(fail_line))
        self.wait(1)
        self.play(FadeOut(fail_line))

        # 학습을 통해 변화하는 결정 경계 (비선형 분리)
        decision_curve = axes.plot(lambda x: np.sin(3 * x) * 0.3 + 0.5, color=GREEN)
        self.play(Create(decision_curve))
        self.wait(2)
        self.play(FadeOut(decision_curve))

        self.wait(1)
