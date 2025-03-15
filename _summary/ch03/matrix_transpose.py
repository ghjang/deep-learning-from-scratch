from manim import *


class MatrixTranspose(Scene):
    # 스타일 상수
    BRACKET_COLOR = "#FFD700"  # 골든 옐로우
    ELEMENT_COLOR = "#50C878"  # 에메랄드 그린
    MATRIX_SCALE = 0.7
    MATRIX_BUFF = 0.3
    VERTICAL_BUFF = 1
    HIGHLIGHT_COLOR = "#FF69B4"  # 글린다 핑크 (Hot Pink)
    HIGHLIGHT_OPACITY = 0.15
    RESULT_H_BUFF = 1.75  # 곱셈 결과 행렬의 수평 간격

    # fmt: off
    # 매트릭스 데이터
    MATRIX_A = [1, 2, 3, 4]
    MATRIX_B = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    MATRIX_C = [[1, -1, 2, 0], [0, 3, -2, 1], [4, 5, 6, -3]]
    MATRIX_D = [[2, 3], [-1, 4], [5, 6], [7, 8]]
    # fmt: on

    def create_styled_matrix(self, matrix_data, is_result=False):
        """매트릭스 객체 생성 및 스타일 적용"""
        matrix_config = {
            "element_to_mobject": lambda x: Text(str(x), color=self.ELEMENT_COLOR)
        }
        if is_result:  # 곱셈 결과 행렬인 경우만 수평 간격 조정
            matrix_config["h_buff"] = self.RESULT_H_BUFF

        data = [matrix_data] if not isinstance(matrix_data[0], list) else matrix_data
        matrix = Matrix(data, **matrix_config)
        matrix.get_brackets().set_color(self.BRACKET_COLOR)
        return matrix

    def create_transposed_matrix(self, matrix_data):
        """매트릭스 전치 변환 및 객체 생성"""
        if not isinstance(matrix_data[0], list):
            data = [[x] for x in matrix_data]
        else:
            data = list(map(list, zip(*matrix_data)))
        return self.create_styled_matrix(data)

    def create_matrix_group(self):
        """ABCD 행렬 그룹 생성 및 반환"""
        matrices = [
            self.create_styled_matrix(data)
            for data in [self.MATRIX_A, self.MATRIX_B, self.MATRIX_C, self.MATRIX_D]
        ]
        return VGroup(*matrices).arrange(RIGHT, buff=self.MATRIX_BUFF)

    def create_transposed_matrices_animation(self, org_matrix_group):
        """전치 행렬 생성 및 애니메이션 시퀀스 생성"""
        animations = []
        last_matrix = None
        transposed_matrices = []

        for idx, matrix_data in enumerate(
            [self.MATRIX_A, self.MATRIX_B, self.MATRIX_C, self.MATRIX_D]
        ):
            transposed = self.create_transposed_matrix(matrix_data).scale(
                self.MATRIX_SCALE
            )
            transposed_matrices.append(transposed)

            if idx == 0:
                transposed.next_to(org_matrix_group, DOWN, buff=self.VERTICAL_BUFF)
                transposed.align_to(org_matrix_group, RIGHT).shift(LEFT * 0.35)
            else:
                transposed.next_to(last_matrix, LEFT, buff=self.MATRIX_BUFF)

            animations.append(TransformFromCopy(org_matrix_group[idx], transposed))
            last_matrix = transposed

        # 전치된 행렬들을 VGroup으로 묶기
        transposed_group = VGroup(*reversed(transposed_matrices))  # DCBA 순서로 구성
        return animations, transposed_group

    def create_highlight_rectangle(self, mobjects, buff=0.1):
        """여러 객체를 감싸는 하이라이트 사각형 생성"""
        group = VGroup(*mobjects)
        rect = SurroundingRectangle(
            group,
            buff=buff,
            color=self.HIGHLIGHT_COLOR,
            fill_color=self.HIGHLIGHT_COLOR,
            fill_opacity=self.HIGHLIGHT_OPACITY,
        )
        return rect

    def matrix_multiply(self, matrix1, matrix2):
        """행렬 곱셈 계산"""
        # 1차원 리스트를 2차원으로 변환
        m1 = [matrix1] if not isinstance(matrix1[0], list) else matrix1
        m2 = [matrix2] if not isinstance(matrix2[0], list) else matrix2

        # 행렬 곱셈을 위한 차원 검증
        if len(m1[0]) != len(m2):
            raise ValueError("Matrix dimensions do not match for multiplication")

        result = []
        for i in range(len(m1)):
            row = []
            for j in range(len(m2[0] if isinstance(m2[0], list) else 1)):
                element = sum(
                    m1[i][k] * (m2[k][j] if isinstance(m2[0], list) else m2[k])
                    for k in range(len(m2))
                )
                row.append(element)
            result.append(row)

        # 결과가 1x1 행렬이면 1차원 리스트로 변환
        return result[0] if len(result) == 1 else result

    def construct(self):
        self.next_section("Matrix Transpose Introduction")

        # 행렬 곱셈 수식 생성 및 표시
        multiplication = (
            VGroup(
                MathTex(
                    "A",
                    "\\cdot",
                    "B",
                    "\\cdot",
                    "C",
                    "\\cdot",
                    "D",
                    "=",
                    "E",
                    color=self.BRACKET_COLOR,
                ),
                MathTex(
                    "(A",
                    "\\cdot",
                    "B",
                    "\\cdot",
                    "C",
                    "\\cdot",
                    "D)^T",
                    "=",
                    "E^T",
                    color=self.BRACKET_COLOR,
                ),
                MathTex(
                    "D^T",
                    "\\cdot",
                    "C^T",
                    "\\cdot",
                    "B^T",
                    "\\cdot",
                    "A^T",
                    "=",
                    "E^T",
                    color=self.BRACKET_COLOR,
                ),
                MathTex("(E^T)^T", "=", "E", color=self.BRACKET_COLOR),
            )
            .scale(1.2)
            .arrange(DOWN, buff=0.5)
        )

        for eq in multiplication:
            self.play(Write(eq))
            self.wait(0.5)
        self.wait()
        self.play(FadeOut(multiplication))

        self.next_section("Initial Matrix Setup")

        # 원본 행렬 그룹 생성 및 표시
        org_matrix_group = self.create_matrix_group().scale(self.MATRIX_SCALE)
        self.play(Write(org_matrix_group))
        self.wait()

        # 전치 행렬 변환 애니메이션
        self.next_section("Transpose")
        self.play(org_matrix_group.animate.to_edge(UP).shift(DOWN * 0.5))

        animations, transposed_group = self.create_transposed_matrices_animation(
            org_matrix_group
        )
        for anim in animations:
            self.play(anim)

        self.next_section("Multiplication")
        # AB와 BA 영역 동시에 하이라이트
        highlight_ab = self.create_highlight_rectangle(org_matrix_group[:2])
        highlight_ba = self.create_highlight_rectangle(transposed_group[-2:])

        self.play(Create(highlight_ab), Create(highlight_ba))
        self.wait()

        # AB 행렬곱 결과 생성
        ab_result = self.matrix_multiply(self.MATRIX_A, self.MATRIX_B)
        ab_matrix = self.create_styled_matrix(ab_result, is_result=True).scale(
            self.MATRIX_SCALE
        )
        ab_matrix.move_to(VGroup(*org_matrix_group[:2]).get_center())

        # BA 행렬곱 결과 생성
        transposed_b_data = list(map(list, zip(*self.MATRIX_B)))
        transposed_a_data = [[x] for x in self.MATRIX_A]
        ba_result = self.matrix_multiply(transposed_b_data, transposed_a_data)
        ba_matrix = self.create_styled_matrix(ba_result, is_result=True).scale(
            self.MATRIX_SCALE
        )
        ba_matrix.move_to(VGroup(*transposed_group[-2:]).get_center())

        # 1단계: 곱셈 결과로 변환
        self.play(
            ReplacementTransform(VGroup(*org_matrix_group[:2]), ab_matrix),
            ReplacementTransform(VGroup(*transposed_group[-2:]), ba_matrix),
            FadeOut(highlight_ab),
            FadeOut(highlight_ba),
        )
        self.wait()

        # 2단계: 결과 행렬 이동
        self.play(
            ab_matrix.animate.next_to(org_matrix_group[2], LEFT, buff=self.MATRIX_BUFF),
            ba_matrix.animate.next_to(
                transposed_group[1], RIGHT, buff=self.MATRIX_BUFF
            ),
        )
        self.wait()

        # MC와 CM 영역 하이라이트
        highlight_mc = self.create_highlight_rectangle(
            VGroup(ab_matrix, org_matrix_group[2])
        )
        highlight_cm = self.create_highlight_rectangle(
            VGroup(transposed_group[1], ba_matrix)
        )

        self.play(Create(highlight_mc), Create(highlight_cm))
        self.wait()

        # MC 행렬곱 결과 생성 (원래 위치에)
        mc_result = self.matrix_multiply(ab_result, self.MATRIX_C)
        mc_matrix = self.create_styled_matrix(mc_result, is_result=True).scale(
            self.MATRIX_SCALE
        )
        mc_matrix.move_to(VGroup(ab_matrix, org_matrix_group[2]))

        # CM 행렬곱 결과 생성 (원래 위치에) - C의 전치행렬 사용
        transposed_c_data = list(map(list, zip(*self.MATRIX_C)))  # C 전치
        cm_result = self.matrix_multiply(transposed_c_data, ba_result)
        cm_matrix = self.create_styled_matrix(cm_result, is_result=True).scale(
            self.MATRIX_SCALE
        )
        cm_matrix.move_to(VGroup(transposed_group[1], ba_matrix))

        # 1단계: 곱셈 결과로 변환
        self.play(
            ReplacementTransform(VGroup(ab_matrix, org_matrix_group[2]), mc_matrix),
            ReplacementTransform(VGroup(transposed_group[1], ba_matrix), cm_matrix),
            FadeOut(highlight_mc),
            FadeOut(highlight_cm),
        )
        self.wait()

        # 2단계: 결과 행렬 이동 (D의 좌측과 우측으로)
        self.play(
            mc_matrix.animate.next_to(org_matrix_group[3], LEFT, buff=self.MATRIX_BUFF),
            cm_matrix.animate.next_to(
                transposed_group[0], RIGHT, buff=self.MATRIX_BUFF
            ),
        )
        self.wait()

        # MD와 DM 영역 하이라이트
        highlight_md = self.create_highlight_rectangle(
            VGroup(mc_matrix, org_matrix_group[3])
        )
        highlight_dm = self.create_highlight_rectangle(
            VGroup(transposed_group[0], cm_matrix)
        )

        self.play(Create(highlight_md), Create(highlight_dm))
        self.wait()

        # MD 행렬곱 결과 생성
        md_result = self.matrix_multiply(mc_result, self.MATRIX_D)
        md_matrix = self.create_styled_matrix(md_result, is_result=True).scale(
            self.MATRIX_SCALE
        )
        md_matrix.move_to(VGroup(mc_matrix, org_matrix_group[3]))

        # DM 행렬곱 결과 생성 (전치된 D 사용)
        transposed_d_data = list(map(list, zip(*self.MATRIX_D)))  # D 전치
        dm_result = self.matrix_multiply(transposed_d_data, cm_result)
        dm_matrix = self.create_styled_matrix(dm_result, is_result=True).scale(
            self.MATRIX_SCALE
        )
        dm_matrix.move_to(VGroup(transposed_group[0], cm_matrix))

        # 최종 곱셈 결과로 변환
        self.play(
            ReplacementTransform(VGroup(mc_matrix, org_matrix_group[3]), md_matrix),
            ReplacementTransform(VGroup(transposed_group[0], cm_matrix), dm_matrix),
            FadeOut(highlight_md),
            FadeOut(highlight_dm),
        )

        self.next_section("Conclusion", skip_animations=False)

        # 최종 결과 행렬들을 화면 중앙으로 이동
        self.play(
            md_matrix.animate.move_to([md_matrix.get_center()[0], 0, 0]),
            dm_matrix.animate.move_to([dm_matrix.get_center()[0], 0, 0]),
        )

        # 하단 행렬에 전치 표시 (T) 추가
        transpose_symbol = MathTex("T", color=self.BRACKET_COLOR).scale(0.7)
        transpose_symbol.next_to(dm_matrix, UP + RIGHT, buff=0.1)
        self.play(Write(transpose_symbol))

        # 등호 추가 및 행렬 재배치 (수직 위치 유지)
        equals = MathTex("=", color=self.BRACKET_COLOR).scale(1.2).move_to([0, 0, 0])

        # 현재 y좌표 저장
        current_y = dm_matrix.get_center()[1]

        self.play(Write(equals))
        self.play(
            dm_matrix.animate.move_to([equals.get_left()[0] - 1.25, current_y, 0]),
            transpose_symbol.animate.shift(
                (equals.get_left()[0] - 1.25 - dm_matrix.get_center()[0]) * RIGHT
            ),
            md_matrix.animate.move_to([equals.get_right()[0] + 1.75, current_y, 0]),
        )

        self.wait(3)
