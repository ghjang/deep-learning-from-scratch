from manim import *


class ActivationFunction(Scene):
    # 클래스 상수 정의
    PLOT_CONFIG = {
        "z_index": 2,
    }
    PLANE_CONFIG = {
        "x_range": [-3, 3, 1],
        "y_range": [-3, 3, 1],
        "axis_config": {"stroke_opacity": 0.5},
        "background_line_style": {"stroke_opacity": 0.3},
    }
    SCALE_FACTOR = 0.55
    POSITIONS = {
        "step": {"edge": UL, "shift": [RIGHT * 3, DOWN * 0.25, 0]},
        "relu": {"edge": UR, "shift": [LEFT * 3, DOWN * 0.25, 0]},
        "sigmoid": {"edge": DL, "shift": [RIGHT * 3, UP * 0.25, 0]},
        "tanh": {"edge": DR, "shift": [LEFT * 3, UP * 0.25, 0]},
    }
    PLOT_COLORS = {
        "step": "#FF7675",  # 부드러운 빨강
        "relu": "#74B9FF",  # 밝은 파랑
        "sigmoid": "#55EFC4",  # 민트 그린
        "tanh": "#FAC42F",  # 골드 옐로우
    }
    X_RANGE = [-3, 3]
    ANIMATION_RUN_TIME = {
        "color_change": 1,
        "move_to_center": 1.5,
    }

    def _create_base_plane_group(self, title: str = "") -> VGroup:
        base_plane_group = VGroup()

        number_plane = NumberPlane(
            **self.PLANE_CONFIG,
            z_index=0,
        )
        base_plane_group.add(number_plane)

        bounding_box = SurroundingRectangle(
            number_plane,
            buff=0,
            color=GRAY,
            stroke_opacity=0.5,
            z_index=1,
        )
        base_plane_group.add(bounding_box)

        if title:
            title_text = Text(title, font_size=24)
            title_text.next_to(bounding_box, DOWN, buff=0.25)
            base_plane_group.add(title_text)

        return base_plane_group

    def _create_discontinuity_markers(
        self,
        number_plane: NumberPlane,
        x: float,
        y_bottom: float,
        y_top: float,
        color: str,
    ) -> VGroup:
        markers = VGroup()

        dot_bottom = Dot(
            point=number_plane.c2p(x, y_bottom),
            color=color,
            z_index=self.PLOT_CONFIG["z_index"],
        )
        dot_top = Dot(
            point=number_plane.c2p(x, y_top),
            color=color,
            z_index=self.PLOT_CONFIG["z_index"],
        )
        vertical_line = DashedLine(
            start=number_plane.c2p(x, y_bottom),
            end=number_plane.c2p(x, y_top),
            color=color,
            z_index=self.PLOT_CONFIG["z_index"],
        )

        markers.add(vertical_line, dot_bottom, dot_top)
        return markers

    def _plot_function(self, base_group: VGroup, func, color: str = BLUE) -> VGroup:
        function_group = VGroup()
        number_plane = base_group[0]

        graph = number_plane.plot(
            func, color=color, x_range=self.X_RANGE, **self.PLOT_CONFIG
        )

        function_group.add(graph)
        base_group.add(function_group)
        return function_group

    def _setup_function_scene(self, title: str, func) -> VGroup:
        base_plane_group = self._create_base_plane_group(title=title)
        self._plot_function(base_plane_group, func)
        return base_plane_group

    def _position_scene(self, scene_group: VGroup, function_type: str):
        pos = self.POSITIONS[function_type]
        scene_group.scale(self.SCALE_FACTOR).to_edge(pos["edge"], buff=0).shift(
            *pos["shift"]
        )

    def _plot_step_function(self, base_group: VGroup, color: str = BLUE) -> VGroup:
        step_function = VGroup()
        number_plane = base_group[0]

        left_graph = number_plane.plot(
            lambda x: 0, color=color, x_range=[self.X_RANGE[0], 0], **self.PLOT_CONFIG
        )

        right_graph = number_plane.plot(
            lambda x: 1, color=color, x_range=[0, self.X_RANGE[1]], **self.PLOT_CONFIG
        )

        discontinuity = self._create_discontinuity_markers(
            number_plane, x=0, y_bottom=0, y_top=1, color=color
        )

        step_function.add(left_graph, right_graph, discontinuity)
        base_group.add(step_function)

        return step_function

    def _setup_step_function_scene(self) -> VGroup:
        base_plane_group = self._create_base_plane_group(title="Step")
        self._plot_step_function(base_plane_group)
        return base_plane_group

    def _setup_composite_scene(
        self, step_scene, relu_scene, sigmoid_scene, tanh_scene
    ) -> VGroup:
        composite_group = VGroup()

        # 단일 NumberPlane 생성 및 표시
        composite_plane = NumberPlane(**self.PLANE_CONFIG, z_index=-1)
        composite_group.add(composite_plane)
        self.play(FadeIn(composite_plane))

        # 각 함수별 변환 및 플롯
        function_setups = [
            (step_scene, self._plot_step_function, self.PLOT_COLORS["step"]),
            (
                relu_scene,
                lambda g, c: self._plot_function(g, lambda x: max(0, x), c),
                self.PLOT_COLORS["relu"],
            ),
            (
                sigmoid_scene,
                lambda g, c: self._plot_function(g, lambda x: 1 / (1 + np.exp(-x)), c),
                self.PLOT_COLORS["sigmoid"],
            ),
            (
                tanh_scene,
                lambda g, c: self._plot_function(g, np.tanh, c),
                self.PLOT_COLORS["tanh"],
            ),
        ]

        for scene, plot_func, color in function_setups:
            # 색상 변경 및 이동 애니메이션
            self.play(
                scene.animate.set_color(color),
                run_time=self.ANIMATION_RUN_TIME["color_change"],
            )
            self.play(
                scene.animate.move_to(composite_plane.get_center()),
                run_time=self.ANIMATION_RUN_TIME["move_to_center"],
            )
            self.play(FadeOut(scene))

            # 함수 플롯
            composite_graph = plot_func(composite_group, color)
            self.play(Create(composite_graph))

        return composite_group

    def construct(self):
        # Step Function
        self.next_section("step function", skip_animations=False)
        step_scene = self._setup_step_function_scene()
        step_scene.scale(0.55).to_edge(UL, buff=0).shift(DOWN * 0.25 + RIGHT * 3)
        self.add(step_scene)

        # ReLU Function
        self.next_section("ReLU function", skip_animations=False)
        relu_scene = self._setup_function_scene("ReLU", lambda x: max(0, x))
        relu_scene.scale(0.55).to_edge(UR, buff=0).shift(DOWN * 0.25 + LEFT * 3)
        self.add(relu_scene)

        # Sigmoid Function
        self.next_section("sigmoid function", skip_animations=False)
        sigmoid_scene = self._setup_function_scene(
            "Sigmoid", lambda x: 1 / (1 + np.exp(-x))
        )
        sigmoid_scene.scale(0.55).to_edge(DL, buff=0).shift(UP * 0.25 + RIGHT * 3)
        self.add(sigmoid_scene)

        # Tanh Function
        self.next_section("tanh function", skip_animations=False)
        tanh_scene = self._setup_function_scene("tanh", np.tanh)
        tanh_scene.scale(0.55).to_edge(DR, buff=0).shift(UP * 0.25 + LEFT * 3)
        self.add(tanh_scene)

        # Initial Animation
        self.play(
            step_scene.animate.shift(LEFT * 2.75),
            relu_scene.animate.shift(RIGHT * 2.75),
            sigmoid_scene.animate.shift(LEFT * 2.75),
            tanh_scene.animate.shift(RIGHT * 2.75),
        )

        # Composite Scene
        self.next_section("Composite Scene", skip_animations=False)

        # Setup Composite Scene with animations
        composite_scene = self._setup_composite_scene(
            step_scene, relu_scene, sigmoid_scene, tanh_scene
        )

        # Final Zoom In with single bounding box
        self.next_section("Final Zoom In", skip_animations=False)
        sr = SurroundingRectangle(
            composite_scene, buff=0.25, color=GRAY, stroke_opacity=0.5
        )
        composite_scene.add(sr)
        self.play(Create(sr))
        self.play(composite_scene.animate.scale(2))

        self.wait(3)
