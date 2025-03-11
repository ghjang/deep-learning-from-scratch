import numpy as np
import json
import matplotlib.pyplot as plt
from manim import *

# Load optimized training log (filtered for significant changes)
with open("xor_training_log.json", "r") as f:
    training_data = json.load(f)


class XORTrainingAnimation(Scene):
    def construct(self):
        axes = Axes(
            x_range=[-0.2, 1.2, 0.2],
            y_range=[-0.2, 1.2, 0.2],
            axis_config={"include_tip": False},
        )
        self.add(axes)

        # XOR input points
        points = [(0, 0, BLUE), (1, 0, RED), (0, 1, RED), (1, 1, BLUE)]
        for x, y, color in points:
            self.add(Dot(axes.c2p(x, y), color=color))

        # Initial separating lines (will be updated dynamically)
        line1 = always_redraw(
            lambda: self.get_decision_boundary(axes, training_data[0][:2], YELLOW)
        )
        line2 = always_redraw(
            lambda: self.get_decision_boundary(axes, training_data[0][2:], GREEN)
        )
        self.add(line1, line2)

        # Animate training updates
        for epoch, (w1, b1, w2, b2) in enumerate(training_data):
            self.play(
                line1.animate.become(
                    self.get_decision_boundary(axes, (w1, b1), YELLOW)
                ),
                line2.animate.become(self.get_decision_boundary(axes, (w2, b2), GREEN)),
                run_time=0.1,  # Adjust speed
            )

    def get_decision_boundary(self, axes, wb, color):
        """Generate a line for given weights and bias."""
        w, b = wb
        if w == 0:
            return Line(axes.c2p(-0.2, -b), axes.c2p(1.2, -b), color=color)
        x_vals = np.array([-0.2, 1.2])
        y_vals = -(w * x_vals + b)
        return Line(axes.c2p(*x_vals, *y_vals), color=color)
