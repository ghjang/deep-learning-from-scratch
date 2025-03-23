from manim import *
from m_and_gate_logic_table import MANDGateLogicTable


class ANDGateLogicTable(Scene):
    def construct(self):
        # 커스텀 VGroup 생성 및 추가
        and_gate_table = MANDGateLogicTable()

        # 씬에 추가
        self.add(and_gate_table)

        self.wait()

        # 마지막 데이터 행(인덱스 5, 1 AND 1 = 1) 변경 - 빨간 원 마크 추가 (더 가까운 간격)
        and_gate_table.update_result_cell(
            5, "?", scene=self, mark_type="red_circle", mark_buff=0.05
        )
        self.wait(0.5)

        # 다시 원래 값으로 변경하고 녹색 원 마크 추가 (기본 간격)
        and_gate_table.update_result_cell(5, "1", scene=self, mark_type="green_circle")
        self.wait(0.5)

        # 첫 번째 데이터 행(인덱스 2, 0 AND 0 = 0) 변경 - 마크 없음
        and_gate_table.update_result_cell(2, "X", scene=self)
        self.wait(0.5)

        # 애니메이션을 사용한 셀 업데이트 예시 - 마크 추가 (더 넓은 간격)
        anim = and_gate_table.update_result_cell(
            3, "Y", scene=self, animate=True, mark_type="green_circle", mark_buff=0.2
        )
        if anim:
            self.play(anim)
        self.wait(0.5)

        # 이전에 추가한 마크가 제거되는지 확인
        anim = and_gate_table.update_result_cell(3, "0", scene=self, animate=True)
        if anim:
            self.play(anim)
        self.wait(0.5)

        self.wait(2)
