from typing import Any
from manim import *


class TruthTable(Table):
    # 스타일 관련 상수
    FONT_SIZE = 62
    LABEL_COLOR = YELLOW
    SOURCE_COLOR = GREEN
    RESULT_COLOR = PINK
    LINE_COLOR = BLUE_B
    LINE_STROKE_WIDTH = 1

    # 간격 조정 상수
    HEADER_LINE_SHIFT = 0.05
    VERTICAL_LINE_SHIFT = 0.05

    def __init__(
        self,
        table_data: list[list[str]],
        col_labels: list[str],
        source_vars_count: int = 1,  # 소스 논리 변수 개수
        **kwargs: Any
    ) -> None:
        super().__init__(
            table_data,
            col_labels=[
                MathTex(label, font_size=self.FONT_SIZE) for label in col_labels
            ],
            include_outer_lines=True,
            line_config={
                "color": self.LINE_COLOR,
                "stroke_width": self.LINE_STROKE_WIDTH,
            },
            **kwargs
        )

        self.source_vars_count = source_vars_count
        self._style_table()
        self._add_double_lines()

    def _style_table(self) -> None:
        self.get_labels().set_color(self.LABEL_COLOR)
        for row in self.get_rows()[1:]:
            # 소스 변수 컬럼들의 텍스트를 녹색으로 설정
            for i in range(self.source_vars_count):
                row[i].set_color(self.SOURCE_COLOR)
            # 결과 컬럼의 텍스트를 분홍색으로 설정
            row[self.source_vars_count :].set_color(self.RESULT_COLOR)

    def _add_double_lines(self) -> None:
        # 헤더 이중선
        second_h_line = self.get_horizontal_lines()[2]
        self.second_h_line_copy = second_h_line.copy().shift(
            UP * self.HEADER_LINE_SHIFT
        )

        # 소스 변수들과 결과를 구분하는 수직 이중선
        v_lines = self.get_vertical_lines()
        self.v_line_copies = []

        # 소스 변수 개수에 따라 수직 구분선 위치 결정
        divider_index = self.source_vars_count + 1
        source_divider_v_line = v_lines[divider_index]
        v_line_copy = source_divider_v_line.copy().shift(
            LEFT * self.VERTICAL_LINE_SHIFT
        )
        self.v_line_copies.append(v_line_copy)

    def get_table_group(self) -> VGroup:
        return VGroup(self, self.second_h_line_copy, *self.v_line_copies)
