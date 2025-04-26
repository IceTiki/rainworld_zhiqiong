import numpy as np
from typing import Iterable
from numpy.typing import NDArray

Vec2d = NDArray[np.float64]  # shape = (2, ), dtype = np.float64


class EndPoint:
    def __init__(self, box: "Box", posi: Vec2d):
        self.box: Box = box
        self.rel_posi: Vec2d = posi

    @property
    def glo_posi(self) -> Vec2d:
        return self.box.position + self.rel_posi

    @property
    def distance_left(self):
        return self.rel_posi[0]

    @property
    def distance_down(self):
        return self.rel_posi[1]

    @property
    def distance_right(self):
        return self.box.size[0] - self.rel_posi[0]

    @property
    def distance_up(self):
        return self.box.size[1] - self.rel_posi[1]

    def to_edge_vecs(self) -> np.ndarray:
        """
        Shape
        ---
        (4, 2)
        """
        return np.array(
            [
                [-self.distance_left, 0],
                [self.distance_right, 0],
                [0, -self.distance_down],
                [0, self.distance_up],
            ]
        )

    def to_vertex_vecs(self):
        return np.array(
            [
                [-self.distance_left, -self.distance_down],
                [self.distance_right, -self.distance_down],
                [-self.distance_left, self.distance_up],
                [self.distance_right, self.distance_up],
            ]
        )


class Edge:
    def __init__(self, end_point_1: EndPoint, end_point_2: EndPoint):
        self.end_points: list[EndPoint] = [end_point_1, end_point_2]

    @property
    def boxes(self):
        return [i.box for i in self.end_points]

    @property
    def end_point_1(self):
        return self.end_points[0]

    @property
    def end_point_2(self):
        return self.end_points[1]


class Box:
    @classmethod
    def combined_big_box(cls, boxes: Iterable["Box"]):
        boxes = list(boxes)
        left_downs = np.array([i.position for i in boxes])
        right_tops = left_downs + np.array([i.size for i in boxes])

        left_down = left_downs.min(axis=0)
        right_top = right_tops.max(axis=0)
        size = right_top - left_down
        return cls(left_down, size)

    def __init__(self, position: Vec2d, size: Vec2d, edges: list[Edge] = None):
        self.position: Vec2d = position
        self.size: Vec2d = size
        self.edges: list[Edge] = edges if edges is not None else []

    @property
    def width(self):
        return self.size[0]

    @property
    def height(self):
        return self.size[1]

    @property
    def area(self):
        return np.prod(self.size)

    @property
    def left_down(self):
        return self.position

    @property
    def right_top(self):
        return self.position + self.size

    @property
    def left_top(self):
        return self.position + self.size * np.array([0, 1])

    @property
    def right_down(self):
        return self.position + self.size * np.array([1, 0])

    @property
    def center(self):
        return self.position + self.size / 2

    def append_edge(self, edge: Edge):
        self.edges.append(edge)

    def coord_to_global(self, coord: Vec2d):
        return self.position + coord

    def is_intersect(self, other: "Box") -> bool:
        x1, y1 = self.left_down
        x1_, y1_ = self.right_top
        x2, y2 = other.left_down
        x2_, y2_ = other.right_top

        return not (x1 > x2_ or x1_ < x2 or y1 > y2_ or y1_ < y2)

