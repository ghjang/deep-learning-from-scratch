from typing import Self, override
from numpy.typing import NDArray
from neural_net import NeuralNet as NN
from activation import ActivationFunction as AF, sigmoid


class MulNode(NN):
    def __init__(self, output_size: int):
        super().__init__()
        self.layer(output_size)

    @override
    def forward(self, x, auto_init_bias=False):
        return super().forward(x, auto_init_bias=auto_init_bias)


type NodeContent = NN | AF | MulNode


class Node:
    content: NodeContent

    def __init__(self, content: NodeContent):
        self.content = content

    def forward(self, x):
        if isinstance(self.content, NN) or isinstance(self.content, MulNode):
            return self.content.forward(x)
        elif isinstance(self.content, AF):
            return self.content(x)
        else:
            raise Exception("Invalid Node Content")


class Graph:
    node_list: list[Node]

    def __init__(self, node_list: list[Node]):
        self.node_list = node_list

    def forward(self, x):
        for node in self.node_list:
            x = node.forward(x)
        return x


class Neuro:
    neuro_graph: Graph = Graph([])

    @staticmethod
    def create() -> Self:
        return Neuro()

    def affine(self, output_size: int) -> Self:
        # NOTE: 기본적으로 NN이 affine 레이어를 생성하게 되어 있음.
        affine_node = NN.create().layer(output_size)
        self.neuro_graph.node_list.append(affine_node)
        return self

    def sigmoid(self) -> Self:
        self.neuro_graph.node_list.append(sigmoid)
        return self

    def forward(self, x):
        return self.neuro_graph.forward(x)
