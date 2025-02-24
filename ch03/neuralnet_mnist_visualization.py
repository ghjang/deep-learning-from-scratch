# fmt: off
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# fmt: on

import numpy as np
from manim import *
from dataset.mnist import load_mnist
from PIL import Image


class NeuralNetMNISTVisualization(Scene):
    # 이미지 관련 상수
    MNIST_IMAGE_HEIGHT = 6
    HORIZONTAL_IMAGE_WIDTH = MNIST_IMAGE_HEIGHT * 28  # 수평 이미지 폭
    FINAL_HORIZONTAL_WIDTH = 14
    HORIZONTAL_HIGHLIGHT_WIDTH = (
        MNIST_IMAGE_HEIGHT  # 28픽셀 폭 (원본 이미지 높이와 동일)
    )

    # 위치 관련 상수
    VERTICAL_GAP = 2  # 원본과 수평 이미지 사이 간격

    # 그리드 관련 상수
    GRID_STROKE_WIDTH = 0.5
    GRID_COLOR = BLUE_A
    BORDER_COLOR = BLUE

    # 애니메이션 관련 상수
    ANIMATION_TIME = {
        "transform": 0.15,
        "highlight_move": 0.2,
        "scroll": 0.15,
        "wait": 0.4,
    }

    def __init__(self):
        super().__init__()
        self.mnist_image = None
        self.img_2d = None

    def construct(self):
        # 섹션 1: 28x28 이미지 표시
        self.next_section("Display MNIST Image with Grid", skip_animations=False)
        self._display_mnist_image_with_grid()

        # 섹션 2: 784x1 수평 변환
        self.next_section("Flatten MNIST Image to 784x1", skip_animations=False)
        horizontal_image, grid_group = self._display_horizontal_mnist_image()

        # 섹션 3: 이미지 재배치
        self.next_section("Rearrange Images", skip_animations=False)
        self._rearrange_images(horizontal_image, grid_group)

        # final wait
        self.wait(3)

    def _load_mnist_image(self):
        """MNIST 데이터 로드 및 이미지 준비"""
        (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
        target_indices = np.where(t_test == 6)[0]
        target_img = x_test[target_indices[0]]
        return target_img.reshape(28, 28).astype(np.uint8)

    def _create_grid_for_image(self, image_mob, num_cells, stroke_width):
        """이미지에 대한 그리드 생성"""
        grid = VGroup()
        for i in range(num_cells + 1):
            # 수직선
            grid.add(
                Line(
                    start=UP * image_mob.height / 2
                    + LEFT * image_mob.width / 2
                    + RIGHT * (i / num_cells) * image_mob.width,
                    end=DOWN * image_mob.height / 2
                    + LEFT * image_mob.width / 2
                    + RIGHT * (i / num_cells) * image_mob.width,
                    stroke_width=stroke_width,
                    color=self.GRID_COLOR,
                )
            )
            # 수평선
            grid.add(
                Line(
                    start=LEFT * image_mob.width / 2
                    + UP * image_mob.height / 2
                    - UP * (i / num_cells) * image_mob.height,
                    end=RIGHT * image_mob.width / 2
                    + UP * image_mob.height / 2
                    - UP * (i / num_cells) * image_mob.height,
                    stroke_width=stroke_width,
                    color=self.GRID_COLOR,
                )
            )
        return grid

    def _create_horizontal_grid_for_image(self, image_mob, num_cells, stroke_width):
        """수평 이미지를 위한 그리드 생성 (수직선 + 상하 경계선)"""
        grid = VGroup()

        # 수직선들 (784개 셀을 나누는 785개 선)
        for i in range(num_cells + 1):
            grid.add(
                Line(
                    start=UP * image_mob.height / 2
                    + LEFT * image_mob.width / 2
                    + RIGHT * (i / num_cells) * image_mob.width,
                    end=DOWN * image_mob.height / 2
                    + LEFT * image_mob.width / 2
                    + RIGHT * (i / num_cells) * image_mob.width,
                    stroke_width=stroke_width,
                    color=self.GRID_COLOR,
                )
            )

        # 상하 수평 경계선
        grid.add(
            Line(
                start=LEFT * image_mob.width / 2 + UP * image_mob.height / 2,
                end=RIGHT * image_mob.width / 2 + UP * image_mob.height / 2,
                stroke_width=stroke_width,
                color=self.GRID_COLOR,
            )
        )
        grid.add(
            Line(
                start=LEFT * image_mob.width / 2 + DOWN * image_mob.height / 2,
                end=RIGHT * image_mob.width / 2 + DOWN * image_mob.height / 2,
                stroke_width=stroke_width,
                color=self.GRID_COLOR,
            )
        )

        return grid

    def _display_mnist_image_with_grid(self):
        """28x28 MNIST 이미지 표시 및 그리드 애니메이션"""
        self.img_2d = self._load_mnist_image()

        # 이미지 객체 생성 및 설정
        self.mnist_image = ImageMobject(self.img_2d)
        self.mnist_image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])

        # 약간 아래로 내린 위치에 배치
        self.play(FadeIn(self.mnist_image))
        self.play(
            self.mnist_image.animate.set_height(self.MNIST_IMAGE_HEIGHT).shift(
                DOWN * 0.5
            )  # 수직 위치 조정
        )

        # 바운딩 박스와 그리드 생성
        rect = SurroundingRectangle(self.mnist_image, color=self.BORDER_COLOR, buff=0)
        grid = self._create_grid_for_image(self.mnist_image, 28, self.GRID_STROKE_WIDTH)
        grid.move_to(self.mnist_image)

        self.play(FadeIn(rect), FadeIn(grid))

    def _setup_horizontal_image(self):
        """수평 이미지와 그리드 초기화"""
        empty_img = np.zeros((1, 784), dtype=np.uint8)
        horizontal_image = ImageMobject(empty_img)
        horizontal_image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        horizontal_image.width = self.HORIZONTAL_IMAGE_WIDTH

        # 원본 이미지 상단에서 0.75 위에 배치
        horizontal_image.move_to(self.mnist_image.get_top() + UP * 0.75)
        horizontal_image.align_to(self.mnist_image, LEFT)

        h_rect = SurroundingRectangle(horizontal_image, color=self.BORDER_COLOR, buff=0)
        h_grid = self._create_horizontal_grid_for_image(horizontal_image, 784, 0.5)
        h_grid.move_to(horizontal_image)

        # 고정된 28픽셀 하이라이트 박스
        fixed_highlight = Rectangle(
            width=self.HORIZONTAL_HIGHLIGHT_WIDTH,
            height=horizontal_image.height,
            stroke_width=0,
            fill_color=YELLOW,
            fill_opacity=0.3,
        ).set_z_index(
            3
        )  # 가장 위에 표시되도록 z-index 설정
        fixed_highlight.move_to(
            horizontal_image.get_left() + RIGHT * self.HORIZONTAL_HIGHLIGHT_WIDTH / 2
        )
        fixed_highlight.align_to(horizontal_image, UP)

        return horizontal_image, Group(h_rect, h_grid), fixed_highlight

    def _setup_row_highlight(self):
        """원본 이미지의 행 하이라이트 초기화"""
        highlight_height = self.mnist_image.height / 28
        row_highlight = Rectangle(
            width=self.mnist_image.width,
            height=highlight_height,
            stroke_width=0,
            fill_color=YELLOW,
            fill_opacity=0.3,
        )
        row_highlight.move_to(
            self.mnist_image.get_top() + DOWN * (highlight_height / 2)
        )
        row_highlight.align_to(self.mnist_image, LEFT)
        return row_highlight

    def _display_horizontal_mnist_image(self):
        """784x1 수평 MNIST 이미지 표시"""
        img_784x1 = self.img_2d.reshape(-1)
        horizontal_image, grid_group, fixed_highlight = self._setup_horizontal_image()
        row_highlight = self._setup_row_highlight()
        current_position = horizontal_image.get_center()

        # 초기 객체들 표시 (fixed_highlight 포함)
        self.play(
            FadeIn(horizontal_image),
            FadeIn(grid_group),
            FadeIn(row_highlight),
            FadeIn(fixed_highlight),
        )

        # 데이터 업데이트 및 스크롤 애니메이션
        self._animate_data_updates(
            img_784x1, horizontal_image, grid_group, row_highlight, current_position
        )

        # 행 하이라이트만 제거 (fixed_highlight는 유지)
        self.play(FadeOut(row_highlight))

        return horizontal_image, grid_group

    def _animate_data_updates(
        self,
        img_784x1,  # 1차원 배열 (784,)
        horizontal_image,
        grid_group,
        row_highlight,
        current_position,
    ):
        """데이터 업데이트 및 스크롤 애니메이션"""
        highlight_height = self.mnist_image.height / 28
        scroll_amount = self.HORIZONTAL_IMAGE_WIDTH / 28

        for i in range(28):
            update_img = np.zeros((1, 784), dtype=np.uint8)
            current_pixels = img_784x1[: (i + 1) * 28]  # 현재까지의 픽셀
            update_img[0, : len(current_pixels)] = current_pixels  # 수평으로 채움

            new_image = ImageMobject(update_img)
            new_image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
            new_image.width = self.HORIZONTAL_IMAGE_WIDTH
            new_image.move_to(current_position)

            if i == 0:
                self.play(
                    Transform(horizontal_image, new_image),
                    run_time=self.ANIMATION_TIME["transform"],
                )
            else:
                # 원본 이미지의 행 하이라이트는 아래로 이동 (원래 방식)
                self.play(
                    Transform(horizontal_image, new_image),
                    row_highlight.animate.shift(DOWN * highlight_height),
                    run_time=self.ANIMATION_TIME["highlight_move"],
                )

                # 수평 이미지는 좌측으로 스크롤
                current_position += LEFT * scroll_amount
                self.play(
                    horizontal_image.animate.shift(LEFT * scroll_amount),
                    grid_group.animate.shift(LEFT * scroll_amount),
                    run_time=self.ANIMATION_TIME["scroll"],
                )

            self.wait(self.ANIMATION_TIME["wait"])

    def _rearrange_images(self, horizontal_image, grid_group):
        """정사각형 이미지 제거 및 수평 이미지 재배치"""
        # grid_group에서 바운딩 박스와 그리드 분리
        bounding_box = grid_group[0]  # 첫 번째 요소는 바운딩 박스
        grid = grid_group[1]  # 두 번째 요소는 그리드

        # 바운딩 박스와 고정 하이라이트 즉시 제거
        fixed_highlight = None
        for obj in self.mobjects:
            if isinstance(obj, Rectangle) and obj.get_fill_opacity() > 0:
                fixed_highlight = obj
                break

        if fixed_highlight:
            self.remove(fixed_highlight)
        self.remove(bounding_box)

        # 원본 이미지 관련 객체들은 페이드아웃으로 처리
        objects_to_fadeout = []
        for obj in self.mobjects:
            if obj not in [horizontal_image, grid, grid_group]:
                objects_to_fadeout.append(obj)

        # 이미지와 그리드만으로 새로운 그룹 생성
        horizontal_group = Group(horizontal_image, grid)

        # 새로운 크기 계산
        new_width = self.FINAL_HORIZONTAL_WIDTH
        scale_factor = new_width / horizontal_image.width

        # 변환 실행: 원본 이미지는 페이드아웃, 수평 이미지는 변환
        self.play(
            *[FadeOut(obj) for obj in objects_to_fadeout],
            horizontal_group.animate.move_to(ORIGIN).scale(scale_factor).to_edge(UP),
            run_time=1.5
        )

        self.wait(0.5)
