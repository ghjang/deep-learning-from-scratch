import numpy as np
from manim import *


class HorseSaddle(ThreeDScene):
    def construct(self):
        # 카메라 설정
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        # 좌표축 생성 (먼저 생성해야 함)
        axes = ThreeDAxes()

        # 말 안장 곡면 함수 정의 (z = 0.4 * (x^2 - y^2)) - 완만한 곡선으로 수정
        def param_surface(u, v):
            x = u
            y = v
            z = 0.4 * (x**2 - y**2)  # 계수 0.4를 곱해 곡면을 더 완만하게 만듦
            return np.array([x, y, z])

        # 말 안장 곡면 생성
        horse_saddle = Surface(
            lambda u, v: param_surface(u, v),
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=(30, 30),
            should_make_jagged=False,
        )

        # 곡면 스타일 설정
        horse_saddle.set_style(
            fill_opacity=0.7, fill_color=BLUE_D, stroke_width=0.5, stroke_color=WHITE
        )

        # 그라데이션 적용 - 수정된 부분
        # Z축 값에 따라 색상 변경하는 대신 직접 컬러맵 적용
        colors = [BLUE_D, GREEN, YELLOW, RED]
        horse_saddle.set_color_by_gradient(*colors)

        # 레이블 생성
        x_label = axes.get_x_axis_label("x")
        y_label = axes.get_y_axis_label("y")
        z_label = axes.get_z_axis_label("z")
        labels = VGroup(x_label, y_label, z_label)

        # 제목 생성 - 수정된 방정식 반영
        title_text = (
            Text("Horse Saddle Surface", font_size=36)
            .to_edge(UL)
            .shift(RIGHT * 1.25 + UP * 0.25)
        )
        title_equation = MathTex(r"z = 0.4 \cdot (x^2 - y^2)", font_size=36).next_to(
            title_text, DOWN
        )
        title = VGroup(title_text, title_equation)

        # 씬에 객체 추가
        self.add(axes, labels, title)

        # 곡면 회전 애니메이션 - 반대 방향으로 회전 (음수 rate 사용)
        self.begin_ambient_camera_rotation(rate=-0.375)  # 양수를 음수로 변경
        self.play(Create(horse_saddle), run_time=4)
        self.wait(1)
        self.stop_ambient_camera_rotation()

        # 스타워즈 스타일로 각도 변경 - 절대 각도 수정
        # 약간만 기울이려면 현재 각도(75도)에서 조금만 변경
        self.move_camera(phi=50 * DEGREES, run_time=2, rate_func=smooth)

        # 회전 애니메이션은 끝까지 유지 (멈추지 않음)
        self.wait(2)
