from manim import *


class LUActivationFunction(Scene):
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
        "leaky_relu": {"edge": UL, "shift": [RIGHT * 3, DOWN * 0.25, 0]},
        "elu": {"edge": UR, "shift": [LEFT * 3, DOWN * 0.25, 0]},
        "selu": {"edge": DL, "shift": [RIGHT * 3, UP * 0.25, 0]},
        "gelu": {"edge": DR, "shift": [LEFT * 3, UP * 0.25, 0]},
    }
    PLOT_COLORS = {
        "leaky_relu": "#FF7675",  # 부드러운 빨강
        "elu": "#74B9FF",  # 밝은 파랑
        "selu": "#55EFC4",  # 민트 그린
        "gelu": "#FAC42F",  # 골드 옐로우
    }
    # 타이틀과 매개변수 표시 색상
    TITLE_COLOR = GREEN
    PARAM_COLOR = PINK
    X_RANGE = [-3, 3]
    ANIMATION_RUN_TIME = {
        "color_change": 1,
        "move_to_center": 1.5,
    }

    def _create_base_plane_group(
        self,
        title: str = "",
        params: dict = None,
        title_color: str = None,
        param_color: str = None,
    ) -> VGroup:
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
            # 타이틀과 매개변수 텍스트 생성
            title_group = VGroup()

            # 기본 타이틀 텍스트 - 지정된 색상 사용
            title_color = title_color or self.TITLE_COLOR
            title_text = Text(title, font_size=24, color=title_color)
            title_group.add(title_text)

            # 매개변수가 있는 경우 색상으로 강조
            if params:
                param_color = param_color or self.PARAM_COLOR
                param_texts = []
                param_text = " ("

                # 각 매개변수 추가
                for i, (key, value) in enumerate(params.items()):
                    if i > 0:
                        param_text += ", "
                    param_text += f"{key}={value}"
                param_text += ")"

                param_text_obj = Text(param_text, font_size=24, color=param_color)
                param_text_obj.next_to(title_text, RIGHT, buff=0.1)
                title_group.add(param_text_obj)

            title_group.arrange(RIGHT, buff=0.1)
            title_group.next_to(bounding_box, DOWN, buff=0.25)
            base_plane_group.add(title_group)

        return base_plane_group

    def _plot_function(self, base_group: VGroup, func, color: str = BLUE) -> VGroup:
        function_group = VGroup()
        number_plane = base_group[0]

        graph = number_plane.plot(
            func, color=color, x_range=self.X_RANGE, **self.PLOT_CONFIG
        )

        function_group.add(graph)
        base_group.add(function_group)
        return function_group

    def _plot_leaky_relu_function(
        self, base_group: VGroup, alpha: float = 0.01, color: str = BLUE
    ) -> VGroup:
        leaky_relu_function = VGroup()
        number_plane = base_group[0]

        # 음수 영역: f(x) = alpha * x
        left_graph = number_plane.plot(
            lambda x: alpha * x,
            color=color,
            x_range=[self.X_RANGE[0], 0],
            **self.PLOT_CONFIG,
        )

        # 양수 영역: f(x) = x
        right_graph = number_plane.plot(
            lambda x: x, color=color, x_range=[0, self.X_RANGE[1]], **self.PLOT_CONFIG
        )

        leaky_relu_function.add(left_graph, right_graph)
        base_group.add(leaky_relu_function)

        return leaky_relu_function

    def _setup_leaky_relu_function_scene(
        self, alpha: float = 0.01, title_color: str = None, param_color: str = None
    ) -> VGroup:
        base_plane_group = self._create_base_plane_group(
            title="Leaky ReLU",
            params={"α": alpha},
            title_color=title_color,
            param_color=param_color,
        )
        self._plot_leaky_relu_function(base_plane_group, alpha)
        return base_plane_group

    def _plot_elu_function(
        self, base_group: VGroup, alpha: float = 1.0, color: str = BLUE
    ) -> VGroup:
        elu_function = VGroup()
        number_plane = base_group[0]

        # 음수 영역: f(x) = alpha * (exp(x) - 1)
        left_graph = number_plane.plot(
            lambda x: alpha * (np.exp(x) - 1),
            color=color,
            x_range=[self.X_RANGE[0], 0],
            **self.PLOT_CONFIG,
        )

        # 양수 영역: f(x) = x
        right_graph = number_plane.plot(
            lambda x: x, color=color, x_range=[0, self.X_RANGE[1]], **self.PLOT_CONFIG
        )

        elu_function.add(left_graph, right_graph)
        base_group.add(elu_function)

        return elu_function

    def _setup_elu_function_scene(
        self, alpha: float = 1.0, title_color: str = None, param_color: str = None
    ) -> VGroup:
        base_plane_group = self._create_base_plane_group(
            title="ELU",
            params={"α": alpha},
            title_color=title_color,
            param_color=param_color,
        )
        self._plot_elu_function(base_plane_group, alpha)
        return base_plane_group

    def _plot_selu_function(
        self,
        base_group: VGroup,
        lmbda: float = 1.0507,
        alpha: float = 1.6732,
        color: str = BLUE,
    ) -> VGroup:
        selu_function = VGroup()
        number_plane = base_group[0]

        # 음수 영역: f(x) = λ * α * (exp(x) - 1)
        left_graph = number_plane.plot(
            lambda x: lmbda * alpha * (np.exp(x) - 1),
            color=color,
            x_range=[self.X_RANGE[0], 0],
            **self.PLOT_CONFIG,
        )

        # 양수 영역: f(x) = λ * x
        right_graph = number_plane.plot(
            lambda x: lmbda * x,
            color=color,
            x_range=[0, self.X_RANGE[1]],
            **self.PLOT_CONFIG,
        )

        selu_function.add(left_graph, right_graph)
        base_group.add(selu_function)

        return selu_function

    def _setup_selu_function_scene(
        self,
        lmbda: float = 1.0507,
        alpha: float = 1.6732,
        title_color: str = None,
        param_color: str = None,
    ) -> VGroup:
        base_plane_group = self._create_base_plane_group(
            title="SELU",
            params={"λ": lmbda, "α": alpha},
            title_color=title_color,
            param_color=param_color,
        )
        self._plot_selu_function(base_plane_group, lmbda, alpha)
        return base_plane_group

    def _plot_gelu_function(self, base_group: VGroup, color: str = BLUE) -> VGroup:
        gelu_function = VGroup()
        number_plane = base_group[0]

        # tanh 근사를 사용한 GELU 구현
        # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        def gelu_approx(x):
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

        graph = number_plane.plot(
            gelu_approx, color=color, x_range=self.X_RANGE, **self.PLOT_CONFIG
        )

        gelu_function.add(graph)
        base_group.add(gelu_function)

        return gelu_function

    def _setup_gelu_function_scene(
        self, title_color: str = None, param_color: str = None
    ) -> VGroup:
        base_plane_group = self._create_base_plane_group(
            title="GELU",
            params={"approx": "tanh"},
            title_color=title_color,
            param_color=param_color,
        )
        self._plot_gelu_function(base_plane_group)
        return base_plane_group

    def _setup_composite_scene(
        self, leaky_relu_scene, elu_scene, selu_scene, gelu_scene
    ) -> VGroup:
        composite_group = VGroup()

        # 단일 NumberPlane 생성 및 표시
        composite_plane = NumberPlane(**self.PLANE_CONFIG, z_index=-1)
        composite_group.add(composite_plane)
        self.play(FadeIn(composite_plane))

        # 각 함수별 변환 및 플롯
        function_setups = [
            (
                leaky_relu_scene,
                lambda g, c: self._plot_leaky_relu_function(g, alpha=0.01, color=c),
                self.PLOT_COLORS["leaky_relu"],
            ),
            (
                elu_scene,
                lambda g, c: self._plot_elu_function(g, alpha=1.0, color=c),
                self.PLOT_COLORS["elu"],
            ),
            (
                selu_scene,
                lambda g, c: self._plot_selu_function(
                    g, lmbda=1.0507, alpha=1.6732, color=c
                ),
                self.PLOT_COLORS["selu"],
            ),
            (
                gelu_scene,
                lambda g, c: self._plot_gelu_function(g, color=c),
                self.PLOT_COLORS["gelu"],
            ),
        ]

        for scene, plot_func, color in function_setups:
            # 색상 변경 및 이동 애메이션
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
        # 활성화 함수 색상 설정
        title_color = self.TITLE_COLOR
        param_color = self.PARAM_COLOR

        # Leaky ReLU Function
        self.next_section("leaky relu function", skip_animations=False)
        leaky_relu_scene = self._setup_leaky_relu_function_scene(
            alpha=0.01, title_color=title_color, param_color=param_color
        )
        leaky_relu_scene.scale(0.55).to_edge(UL, buff=0).shift(DOWN * 0.25 + RIGHT * 3)
        self.add(leaky_relu_scene)

        # ELU Function
        self.next_section("ELU function", skip_animations=False)
        elu_scene = self._setup_elu_function_scene(
            alpha=1.0, title_color=title_color, param_color=param_color
        )
        elu_scene.scale(0.55).to_edge(UR, buff=0).shift(DOWN * 0.25 + LEFT * 3)
        self.add(elu_scene)

        # SELU Function
        self.next_section("SELU function", skip_animations=False)
        selu_scene = self._setup_selu_function_scene(
            lmbda=1.0507, alpha=1.6732, title_color=title_color, param_color=param_color
        )
        selu_scene.scale(0.55).to_edge(DL, buff=0).shift(UP * 0.25 + RIGHT * 3)
        self.add(selu_scene)

        # GELU Function - tanh 근사 설명 수정
        self.next_section("GELU function", skip_animations=False)
        gelu_scene = self._setup_gelu_function_scene(
            title_color=title_color, param_color=param_color
        )
        gelu_scene.scale(0.55).to_edge(DR, buff=0).shift(UP * 0.25 + LEFT * 3)
        self.add(gelu_scene)

        # initial wait
        self.wait(2)

        # Initial Animation
        self.play(
            leaky_relu_scene.animate.shift(LEFT * 2.75),
            elu_scene.animate.shift(RIGHT * 2.75),
            selu_scene.animate.shift(LEFT * 2.75),
            gelu_scene.animate.shift(RIGHT * 2.75),
        )

        # Composite Scene
        self.next_section("Composite Scene", skip_animations=False)

        # Setup Composite Scene with animations
        composite_scene = self._setup_composite_scene(
            leaky_relu_scene, elu_scene, selu_scene, gelu_scene
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
