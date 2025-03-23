from manim import *
from m_single_layper_perceptron import MSingleLayerPerceptron


class MPerceptronAndGateTest(Scene):
    def construct(self):
        perceptron_network = MSingleLayerPerceptron()

        self.add(perceptron_network)
        self.play(perceptron_network.animate.scale(0.5).to_edge(DR))

        self.wait(3)
