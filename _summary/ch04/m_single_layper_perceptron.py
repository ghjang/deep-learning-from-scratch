from manim import *
from mperceptron import MPerceptron


class MSingleLayerPerceptron(VGroup):
    def __init__(
        self,
        perceptron_radius=1.5,
        perceptron_outer_text=r"\text{P}_h",
        layout_direction="horizontal",
        input_labels=["1", "p", "q"],
        output_label="f(p, q)",
        activation_function="sigmoid()",
        activation_text="h()",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # 수평 배치 퍼셉트론 생성
        self.perceptron = MPerceptron(
            radius=perceptron_radius,
            outer_text=perceptron_outer_text,
            layout_direction=layout_direction,
        )

        self.perceptron.show_inner_circles(r"\text{A}", r"\text{Z}", activation_text)

        # 입력/출력 포인트 생성
        self.input_dot = Dot(color=RED).move_to(self.perceptron.get_input_point())
        self.output_dot = Dot(color=GREEN).move_to(self.perceptron.get_output_point())

        self.sigmoid_text = MathTex(
            f"{activation_text} = {activation_function}", font_size=28
        ).next_to(self.perceptron, DOWN)

        self.circle_group = VGroup(
            self.perceptron,
            self.input_dot,
            self.output_dot,
            self.sigmoid_text,
        ).set_z_index(1)

        # 입력 서클 생성
        input_circle_radius = self.perceptron.radius * 0.25
        input_circle_stroke_color = self.perceptron.stroke_color
        input_circle_stroke_width = self.perceptron.stroke_width
        input_circle_fill_color = self.perceptron.fill_color
        input_circle_fill_opacity = self.perceptron.fill_opacity

        self.input_circles = []
        self.input_texts = []
        self.input_groups = VGroup()

        for label in input_labels:
            circle = Circle(
                radius=input_circle_radius,
                color=input_circle_stroke_color,
                stroke_width=input_circle_stroke_width,
                fill_color=input_circle_fill_color,
                fill_opacity=input_circle_fill_opacity,
            )
            text = MathTex(label, font_size=30).move_to(circle)
            self.input_circles.append(circle)
            self.input_texts.append(text)
            self.input_groups.add(VGroup(circle, text))

        self.input_group = (
            self.input_groups.arrange_in_grid(len(input_labels), 1, buff=0.5)
            .move_to(self.perceptron.get_input_point())
            .shift(LEFT * 2)
            .set_z_index(0)
        )

        # 화살표 생성
        self.arrows = []
        self.weights = []

        # bias 화살표
        bias_arrow = Arrow(
            self.input_circles[0].get_right(),
            self.input_dot.get_left() + UP * 0.125,
            buff=0,
            stroke_width=3,
            color=YELLOW,
            tip_length=0.2,
        )
        b_weight = MathTex("b", font_size=28).move_to(
            bias_arrow.get_center() + UP * 0.35
        )
        self.arrows.append(bias_arrow)
        self.weights.append(b_weight)

        # p 화살표
        p_arrow = Arrow(
            self.input_circles[1].get_right(),
            self.input_dot.get_left(),
            buff=0,
            stroke_width=3,
            color=YELLOW,
            tip_length=0.2,
        )
        w1_weight = MathTex("w_1", font_size=28).move_to(
            p_arrow.get_center() + UP * 0.25
        )
        self.arrows.append(p_arrow)
        self.weights.append(w1_weight)

        # q 화살표
        q_arrow = Arrow(
            self.input_circles[2].get_right(),
            self.input_dot.get_left() + DOWN * 0.125,
            buff=0,
            stroke_width=3,
            color=YELLOW,
            tip_length=0.2,
        )
        w2_weight = MathTex("w_2", font_size=28).move_to(
            q_arrow.get_center() + DOWN * 0.35
        )
        self.arrows.append(q_arrow)
        self.weights.append(w2_weight)

        self.input_arrow_group = VGroup(*self.arrows, *self.weights).set_z_index(0)

        # 출력 서클 생성
        self.output_circle = (
            Circle(
                radius=input_circle_radius * 1.6,
                color=input_circle_stroke_color,
                stroke_width=input_circle_stroke_width,
                fill_color=input_circle_fill_color,
                fill_opacity=input_circle_fill_opacity,
            )
            .move_to(self.perceptron.get_output_point())
            .shift(RIGHT * 2)
        )

        self.output_text = MathTex(output_label, font_size=30).move_to(
            self.output_circle
        )
        self.output_arrow = Arrow(
            self.output_dot.get_right(),
            self.output_circle.get_left(),
            buff=0,
            stroke_width=3,
            color=YELLOW,
            tip_length=0.2,
        )

        self.output_group = VGroup(self.output_circle, self.output_text).set_z_index(0)

        # 모든 요소 추가
        self.add(
            self.circle_group,
            self.input_group,
            self.input_arrow_group,
            self.output_arrow,
            self.output_group,
        )
