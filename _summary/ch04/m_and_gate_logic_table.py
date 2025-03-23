from manim import *
from truth_table import TruthTable


class MANDGateLogicTable(VGroup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 제목 생성
        self.title = MathTex(
            r"\text{AND Gate Output Approximation}", color=YELLOW, font_size=62
        )

        # AND 게이트 진리표 데이터
        table_data = [
            ["0", "0", "0"],
            ["0", "1", "0"],
            ["1", "0", "0"],
            ["1", "1", "1"],
        ]

        # 열 레이블 정의
        col_labels = ["P", "Q", r"f(P,\; Q) = P \text{ AND } Q"]

        # TruthTable 클래스 인스턴스 생성 (소스 변수는 P와 Q로 2개)
        self.truth_table = TruthTable(
            table_data=table_data,
            col_labels=col_labels,
            source_vars_count=2,
            element_to_mobject=lambda text: MathTex(
                text, font_size=TruthTable.FONT_SIZE
            ),
        )

        # 테이블 그룹 가져오기 (이중선 포함)
        self.table_group = self.truth_table.get_table_group()

        # 결과 셀에 접근하기 쉽도록 저장 - 인덱스 범위 수정 (1베이스드 인덱스)
        self.result_cells = [self.truth_table.get_entries((i + 1, 3)) for i in range(5)]

        # 제목과 테이블을 함께 배치
        self.add(self.title, self.table_group)
        self.arrange(DOWN, buff=0.5)

    def update_result_cell(
        self,
        row_idx,
        new_value,
        scene=None,
        animate=False,
        mark_type=None,
        mark_buff=0.1,
        current_scale=1,
    ):
        """
        특정 행의 결과 셀 값을 업데이트

        Parameters:
        -----------
        row_idx : int
            업데이트할 행 인덱스 (1-5, 1: 헤더, 2-5: 데이터 행)
        new_value : str
            새로운 결과 값 (LaTeX 문자열)
        scene : Scene, optional
            셀 업데이트를 적용할 씬 객체
        animate : bool
            애니메이션 사용 여부
        mark_type : str, optional
            마크 유형 ('red_circle', 'green_circle', None)
        mark_buff : float, optional
            마크 표시 간격 (기본값: 0.1)

        Returns:
        --------
        Animation 또는 None:
            scene이 None이고 animate=True인 경우 애니메이션 객체 반환
        """
        # 행 인덱스 검증 (1-5 범위)
        if 1 <= row_idx <= 5:
            # result_cells 배열에서는 인덱스가 0부터 시작하므로 조정 필요
            cell_idx = row_idx - 1
            old_cell = self.result_cells[cell_idx]

            # 이전 셀에 연결된 마크가 있는지 확인
            has_old_mark = False
            old_mark = None

            # 이전 셀에 마크가 있는지 확인하고 제거
            if hasattr(old_cell, "mark"):
                has_old_mark = True
                old_mark = old_cell.mark

            # 새 셀 생성
            new_cell = MathTex(
                new_value,
                font_size=TruthTable.FONT_SIZE * current_scale,
                color=TruthTable.RESULT_COLOR,
            )
            new_cell.move_to(old_cell.get_center())

            # 새 마크 생성 (요청된 경우)
            new_mark = None
            if mark_type == "red_circle":
                new_mark = Circle(
                    radius=0.15 * current_scale,
                    color=RED,
                    fill_opacity=0.8,
                    stroke_width=0,
                )
                new_mark.next_to(new_cell, RIGHT, buff=mark_buff * current_scale)
                new_cell.mark = new_mark
            elif mark_type == "green_circle":
                new_mark = Circle(
                    radius=0.15 * current_scale,
                    color=GREEN,
                    fill_opacity=0.8,
                    stroke_width=0,
                )
                new_mark.next_to(new_cell, RIGHT, buff=mark_buff * current_scale)
                new_cell.mark = new_mark

            # 씬이 제공된 경우
            if scene:
                # 기존 요소들 제거
                scene.remove(old_cell)
                if has_old_mark:
                    scene.remove(old_mark)

                # 새 요소들 추가
                scene.add(new_cell)
                if new_mark:
                    scene.add(new_mark)

                # 내부 배열 업데이트
                self.result_cells[cell_idx] = new_cell

                # 애니메이션이 필요한 경우
                if animate:
                    animations = [
                        FadeOut(old_cell, run_time=0.5),
                        FadeIn(new_cell, run_time=0.5),
                    ]

                    if has_old_mark:
                        animations.append(FadeOut(old_mark, run_time=0.5))

                    if new_mark:
                        animations.append(FadeIn(new_mark, run_time=0.5))

                    return AnimationGroup(*animations)
                return None
            else:
                # 씬이 제공되지 않은 경우
                if animate:
                    animations = [
                        FadeOut(old_cell, run_time=0.5),
                        FadeIn(new_cell, run_time=0.5),
                    ]

                    if has_old_mark:
                        animations.append(FadeOut(old_mark, run_time=0.5))

                    if new_mark:
                        animations.append(FadeIn(new_mark, run_time=0.5))

                    return AnimationGroup(*animations)

                # 애니메이션 없이 바로 교체를 위한 객체들 반환
                result = [new_cell, old_cell]

                if has_old_mark:
                    result.append(old_mark)

                if new_mark:
                    result.append(new_mark)

                return tuple(result)
        return None

    def highlight_row(self, row_idx, color=YELLOW, stroke_width=4):
        """
        특정 행을 하이라이트

        Parameters:
        -----------
        row_idx : int
            하이라이트할 행 인덱스 (1-5, 1: 헤더, 2-5: 데이터 행)
        """
        if 1 <= row_idx <= 5:
            row = self.truth_table.get_rows()[row_idx]
            rect = SurroundingRectangle(row, color=color, stroke_width=4)
            return rect
        return None
