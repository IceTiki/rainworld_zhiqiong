import typing
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import constants as c
from typing import TypedDict, TypeVar, Literal, Optional, Iterable, Annotated, Generator
from matplotlib.patches import Rectangle
from dataclasses import dataclass, field
from loguru import logger
import random
from tqdm import tqdm
from pprint import pprint
import utils
import math
import itertools
from collections import deque
from numpy.typing import NDArray
from numpy.linalg import norm

DEBUG = False

# if typing.TYPE_CHECKING:
#     from rooms import Room, Connection, Region, RegionTeleportConnection
# else:
#     Room = TypeVar("Room")
#     Connection = TypeVar("Connection")
#     Region = TypeVar("Region")
#     RegionTeleportConnection = TypeVar("RegionTeleportConnection")


# Vec2d = NDArray[np.float64]
# RoomMap = dict[Room, "Box"]
# ConnectionMap = dict[Connection, "Edge"]


# class EndPoint:
#     def __init__(self, box: "Box", posi: Vec2d):
#         self.box: Box = box
#         self.ep_posi: Vec2d = posi

#     @property
#     def distance_left(self):
#         return self.ep_posi[0]

#     @property
#     def distance_down(self):
#         return self.ep_posi[1]

#     @property
#     def distance_right(self):
#         return self.box.size[0] - self.ep_posi[0]

#     @property
#     def distance_top(self):
#         return self.box.size[1] - self.ep_posi[1]

#     def to_edge_vecs(self):
#         return np.array(
#             [
#                 [-self.distance_left, 0],
#                 [self.distance_right, 0],
#                 [0, -self.distance_down],
#                 [0, self.distance_top],
#             ]
#         )

#     def to_vertex_vecs(self):
#         return np.array(
#             [
#                 [-self.distance_left, -self.distance_down],
#                 [self.distance_right, -self.distance_down],
#                 [-self.distance_left, self.distance_top],
#                 [self.distance_right, self.distance_top],
#             ]
#         )


# class Edge:
#     def __init__(self, end_point_1: EndPoint, end_point_2: EndPoint):
#         self.end_points: list[EndPoint] = [end_point_1, end_point_2]

#     @property
#     def boxes(self):
#         return [i.box for i in self.end_points]


# class Box:
#     @classmethod
#     def combined_big_box(cls, boxes: Iterable["Box"]):
#         boxes = list(boxes)
#         left_downs = np.array([i.position for i in boxes])
#         right_tops = left_downs + np.array([i.size for i in boxes])

#         left_down = left_downs.min(axis=0)
#         right_top = right_tops.max(axis=0)
#         size = right_top - left_down
#         return cls(left_down, size)

#     @classmethod
#     def from_room(cls, room: Room):
#         return cls(room.box_position.copy(), room.box_size.copy())

#     @classmethod
#     def build_graph(
#         cls, rooms: Iterable[Room], conns: Iterable[Connection]
#     ) -> tuple[RoomMap, ConnectionMap]:
#         room_map: RoomMap = {r: cls.from_room(r) for r in rooms}
#         conn_map: ConnectionMap = {}
#         for conn in conns:
#             for room in (conn.room1, conn.room2):
#                 if room not in room_map:
#                     room_map[room] = cls.from_room(room)

#             edge = Edge(
#                 EndPoint(room_map[conn.room1], conn.room1_posi),
#                 EndPoint(room_map[conn.room2], conn.room2_posi),
#             )
#             conn_map[conn] = edge
#             for room in (conn.room1, conn.room2):
#                 box = room_map[room]
#                 box.append_edge(edge)

#         return room_map, conn_map

#     def __init__(self, position: Vec2d, size: Vec2d, edges: list[Edge] = None):
#         self.position: Vec2d = position
#         self.size: Vec2d = size
#         self.edges: list[Edge] = edges if edges is not None else []

#     @property
#     def area(self):
#         return np.prod(self.size)

#     @property
#     def left_down(self):
#         return self.position

#     @property
#     def right_top(self):
#         return self.position + self.size

#     def append_edge(self, edge: Edge):
#         self.edges.append(edge)

#     def coord_to_global(self, coord: Vec2d):
#         return self.position + coord

#     def is_intersect(self, other: "Box") -> bool:
#         x1, y1 = self.left_down
#         x1_, y1_ = self.right_top
#         x2, y2 = other.left_down
#         x2_, y2_ = other.right_top

#         return not (x1 > x2_ or x1_ < x2 or y1 > y2_ or y1_ < y2)

from colls import Box, Edge, EndPoint


class BaseOpt:
    # def optimizing_rooms(
    #     self, rooms: Iterable[Room], connections: Iterable[Connection]
    # ):
    #     room_map, conn_map = Box.build_graph(rooms, connections)
    #     self(list(room_map.values()), list(conn_map.values()))
    #     for room, box in room_map.items():
    #         room.box_position = box.position

    # def optimizing_regions(
    #     self, regions: Iterable[Region], rt_conns: Iterable[RegionTeleportConnection]
    # ):
    #     region_map: dict[Region, Box] = {i: i.region_box for i in regions}
    #     conn_map: dict[RegionTeleportConnection, Edge] = {}
    #     for conn in rt_conns:
    #         for region in (conn.region1, conn.region2):
    #             if region not in region_map:
    #                 region_map[region] = region.region_box

    #         edge = Edge(
    #             EndPoint(region_map[conn.region1], conn.reg1_posi),
    #             EndPoint(region_map[conn.region2], conn.reg2_posi),
    #         )
    #         conn_map[conn] = edge
    #         for region in (conn.region1, conn.region2):
    #             box = region_map[region]
    #             box.append_edge(edge)

    #     self(list(region_map.values()), list(conn_map.values()))
    #     for region, box in region_map.items():
    #         region.position = box.position

    def __init__(self):
        pass

    def __call__(self, boxes: list[Box], edges: list[Edge]) -> None:
        self.run(boxes, edges)

    def run(self, boxes: list[Box], edges: list[Edge]) -> None:
        return

    _TEST_PLOT_COUNT = 0

    def _test_plot(
        self, boxes: list[Box], edges: list[Edge], output: Path | None | bool = None
    ):
        if not DEBUG:
            return
        if output is None:
            import constants as cons
            import time

            output = cons.TMP_OUTPUT_PATH / f"{time.time_ns()}.png"
            self._TEST_PLOT_COUNT += 1

        fig, ax = plt.subplots(facecolor="white")
        ax: plt.Axes

        for edge in edges:
            ep1, ep2 = edge.end_points
            ax.plot(
                [ep1.glo_posi[0], ep2.glo_posi[0]],
                [ep1.glo_posi[1], ep2.glo_posi[1]],
                color="black",
            )

        for box in boxes:
            rect = Rectangle(
                box.position,
                box.width,
                box.height,
                linewidth=1,
                edgecolor="black",
            )
            ax.add_patch(rect)

        big_box = Box.combined_big_box(boxes)
        x0, y0 = big_box.left_down
        x1, y1 = big_box.right_top

        delta_x = big_box.size[0]
        delta_y = big_box.size[1]
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

        ax.set_aspect(1)

        R = 20

        r = delta_x / delta_y
        new_h = R / math.sqrt(r)
        new_w = R * math.sqrt(r)

        fig.set_size_inches(new_w, new_h, forward=True)

        fig.tight_layout()

        if output is False:
            plt.show()
            plt.close()
        else:
            fig.savefig(
                output,
                dpi=40,
            )
            plt.close()


class BfsBoxes:
    def __init__(self, boxes: list[Box]):
        self.boxes: list[Box] = list(boxes)
        self.reset()

    @property
    def done_boxes(self):
        return set(self.boxes) - self.todo_boxes

    def reset(self):
        self.todo_boxes = set(self.boxes)
        self.done_edges = set()

    def choice(self) -> Box:
        return max(self.todo_boxes, key=lambda x: len(x.edges))

    def __iter__(self) -> Generator[tuple[EndPoint, EndPoint] | EndPoint, None, None]:
        queue: deque[Edge] = deque()
        # BFS
        while True:
            if not queue:
                if not self.todo_boxes:
                    break
                start_box = self.choice()
                queue.extend(
                    sorted(
                        start_box.edges, key=lambda x: x.boxes[0].area + x.boxes[1].area
                    )
                )
                yield EndPoint(start_box, np.array([0, 0]))  # 非联通图的首个节点

                self.todo_boxes.remove(start_box)
                if not queue:
                    continue

            edge = queue.popleft()
            if edge in self.done_edges:
                continue
            self.done_edges.add(edge)
            for i, end_point in enumerate(edge.end_points):
                box = end_point.box
                if box not in self.todo_boxes:
                    continue

                queue.extend(box.edges)

                yield edge.end_points[1 - i], end_point
                self.todo_boxes.remove(box)


class DfsBoxes:
    def __init__(self, boxes: list[Box]):
        self.boxes: list[Box] = list(boxes)
        self.reset()

    @property
    def done_boxes(self):
        return set(self.boxes) - self.todo_boxes

    def reset(self):
        self.todo_boxes = set(self.boxes)
        self.done_edges = set()

    def choice(self) -> Box:
        return max(self.todo_boxes, key=lambda x: len(x.edges))

    def __iter__(self) -> Generator[tuple[EndPoint, EndPoint] | EndPoint, None, None]:
        stack: list[Edge] = []
        while True:
            if not stack:
                if not self.todo_boxes:
                    break
                start_box = self.choice()
                stack.extend(
                    sorted(
                        start_box.edges,
                        key=lambda x: x.boxes[0].area + x.boxes[1].area,
                        reverse=True,
                    )
                )

                yield EndPoint(start_box, np.array([0, 0]))  # 非联通图的首个节点
                self.todo_boxes.remove(start_box)

                if not stack:
                    continue

            edge = stack.pop()
            if edge in self.done_edges:
                continue
            self.done_edges.add(edge)

            for i, end_point in enumerate(edge.end_points):
                box = end_point.box
                if box not in self.todo_boxes:
                    continue
                stack.extend(
                    sorted(
                        box.edges,
                        key=lambda x: x.boxes[0].area + x.boxes[1].area,
                        reverse=True,
                    )
                )

                yield edge.end_points[1 - i], end_point
                self.todo_boxes.remove(box)


class AlignOpt(BaseOpt):
    def __init__(self, search: Literal["bfs", "dfs"] = "bfs"):
        super().__init__()
        self.search = search

    def run(self, boxes: list[Box], edges: list[Edge]):
        logger.debug("start align opt")
        searcher = DfsBoxes(boxes) if self.search == "dfs" else BfsBoxes(boxes)
        i = 0
        for eps in searcher:
            if isinstance(eps, EndPoint):
                if not searcher.done_boxes:
                    continue
                big_box = Box.combined_big_box(searcher.done_boxes)
                eps.box.position = big_box.position - np.array(
                    [0, eps.box.height], np.float64
                )

                continue
            ep_main, ep_sub = eps
            self.align(ep_main, ep_sub)
            # self.repel(ep_sub.box, searcher.done_boxes)
            self.repel2(ep_main.box, ep_sub.box, searcher.done_boxes)
            self._test_plot(boxes, edges)
            i += 1
        logger.debug("end align opt")

    @staticmethod
    def cal_best_edge_vec_pair(end_point_main: EndPoint, end_point_sub: EndPoint):
        best_pair = None
        for vec1, vec2 in itertools.product(
            end_point_main.to_edge_vecs(), end_point_sub.to_edge_vecs()
        ):
            cos_theta = (vec1 @ vec2) / (norm(vec1) * norm(vec2))
            if cos_theta > -1 + 1e-6:
                continue
            if best_pair is None:
                best_pair = (vec1, vec2)
            elif norm(vec1 - vec2) < norm(best_pair[0] - best_pair[1]):
                best_pair = (vec1, vec2)

        return best_pair

    @classmethod
    def align(cls, end_point_main: EndPoint, end_point_sub: EndPoint):
        if end_point_main.box is end_point_sub.box:
            return

        vec1, vec2 = cls.cal_best_edge_vec_pair(end_point_main, end_point_sub)
        best_conn = vec1 - vec2
        end_point_sub.box.position = (
            end_point_main.box.position
            + end_point_main.rel_posi
            + best_conn
            - end_point_sub.rel_posi
        )
        return

        conn_vec: list[np.ndarray] = []
        for vec1, vec2 in itertools.product(
            end_point_main.to_edge_vecs(), end_point_sub.to_edge_vecs()
        ):
            cos_theta = (vec1 @ vec2) / (norm(vec1) * norm(vec2))
            if cos_theta > -1 + 1e-6:
                continue
            conn_vec.append(vec1 - vec2)

        for vec1, vec2 in itertools.product(
            end_point_main.to_vertex_vecs(), end_point_sub.to_vertex_vecs()
        ):
            cos_theta = (vec1 @ vec2) / (norm(vec1) * norm(vec2))
            if cos_theta > 0:
                continue
            conn_vec.append(vec1 - vec2)

        best_conn: np.ndarray = min(conn_vec, key=lambda x: norm(x))

        end_point_sub.box.position = (
            end_point_main.box.position
            + end_point_main.rel_posi
            + best_conn
            - end_point_sub.rel_posi
        )

    def repel(
        self,
        box: Box,
        fixed_boxes: set[Box],
        step=1,
    ):
        have_intersect = True

        count_ = 0
        np.random.seed(42)
        while have_intersect:
            count_ += 1
            have_intersect = False

            for b in fixed_boxes:
                if b is box:
                    continue
                if not box.is_intersect(b):
                    continue
                have_intersect = True
                delta = box.position - b.position
                if np.linalg.norm(delta) == 0:
                    delta = np.random.randn(2)
                direction = delta / np.linalg.norm(delta) * step
                box.position += direction

            if count_ % 500 == 0:
                step *= 1.1
                box.position += np.random.randn(2) * step * 4
                if count_ % 20000 == 0:
                    logger.debug(f"Repel iter {count_}, {box.position=}")

    def repel2(
        self,
        main_box: Box,
        sub_box: Box,
        done_boxes: set[Box],
        step=1,
    ):

        fixed_boxes: set[Box] = {main_box}
        new_posi_boxes: set[Box] = {sub_box}
        while new_posi_boxes:
            box = new_posi_boxes.pop()
            self.repel(box, fixed_boxes)
            fixed_boxes.add(box)
            for b in done_boxes:
                if b is box:
                    continue
                if not box.is_intersect(b):
                    continue
                if b in fixed_boxes:
                    continue

                new_posi_boxes.add(b)


class ForceOpt(BaseOpt):
    @dataclass
    class BoxOptParams:
        direction: np.ndarray = field(
            default_factory=lambda: np.array([0, 0], dtype=np.float64)
        )

    @staticmethod
    def cal_inbox_force(ep: EndPoint, delta: np.ndarray) -> np.ndarray:
        dx, dy = delta

        if dx < 0:
            fx = -ep.distance_left / (ep.distance_left + ep.distance_right)
        else:
            fx = ep.distance_right / (ep.distance_left + ep.distance_right)

        if dy < 0:
            fy = -ep.distance_down / (ep.distance_up + ep.distance_down)
        else:
            fy = ep.distance_up / (ep.distance_up + ep.distance_down)

        return np.array([fx, fy])

    def __init__(self, iters: int | None = None):
        super().__init__()
        self.iters: int | None = iters

    def run(self, boxes: list[Box], edges: list[Edge]):
        box_notes = {i: self.BoxOptParams() for i in boxes}

        iters = self.iters if self.iters is not None else len(boxes) * 30
        for i in tqdm(range(iters), desc="force opt"):
            ip = i / iters
            # grow_ip = (1 - np.cos(ip * np.pi)) / 2
            # decrease_ip = (np.cos(ip * np.pi) + 1) / 2
            self.step2(
                boxes,
                edges,
                box_notes=box_notes,
                pull_factor=(1 - ip) * 0.5,
                repel_factor=ip * 10,
            )
            if i % 20 == 0:
                self._test_plot(boxes, edges)

    @staticmethod
    def cal_box_repel(stable_box: Box, repel_box: Box):
        x1, y1 = stable_box.left_down
        x1_, y1_ = stable_box.right_top
        x2, y2 = repel_box.left_down
        x2_, y2_ = repel_box.right_top

        # no overlap
        if x1 >= x2_ or x1_ <= x2 or y1 >= y2_ or y1_ <= y2:
            return np.array([0, 0], np.float64)

        # calculate overlap in four directions
        dx_left = x1 - x2_  # move repel_box to left
        dx_right = x2 - x1_  # move repel_box to right
        dy_down = y1 - y2_  # move down
        dy_up = y2 - y1_  # move up

        # pick the shortest move
        candidates = [
            np.array([dx_left, 0], dtype=np.float64),
            np.array([dx_right, 0], dtype=np.float64),
            np.array([0, dy_down], dtype=np.float64),
            np.array([0, dy_up], dtype=np.float64),
        ]

        best = min(candidates, key=lambda v: np.linalg.norm(v))
        return best

    @staticmethod
    def edge_weight(edge: Edge, box: Box):
        # return 1 / (len(box.edges) + 1)
        all_box = {j for i in box.edges for j in i.boxes}

        weight = (sum(map(lambda x: x.area, edge.boxes)) - box.area) / sum(
            map(lambda x: x.area, all_box)
        )
        assert 0 <= weight <= 1
        return weight

    @staticmethod
    def box_weight(box: Box):
        return 1 / (len(box.edges) + 1)

    # def step(
    #     self,
    #     boxes: list[Box],
    #     edges: list[Edge],
    #     *,
    #     box_notes: dict[Box, BoxOptParams],
    #     step_length: float = 1,
    # ):
    #     for edge in edges:
    #         ep1, ep2 = edge.end_points
    #         delta = ep2.glo_posi - ep1.glo_posi
    #         delta_unit = utils.uniform(delta)

    #         weight_factor = ep1.box.area + ep2.box.area
    #         inbox_force = (
    #             self.cal_inbox_force(ep1, delta_unit)
    #             - self.cal_inbox_force(ep2, -delta_unit)
    #         ) * weight_factor

    #         force = inbox_force

    #         force_1 = force / ep1.box.area + 0.25 * delta  / len(ep1.box.edges)
    #         force_2 = -force / ep2.box.area - 0.25 * delta  / len(ep2.box.edges)

    #         box_notes[ep1.box].direction += force_1 * step_length
    #         box_notes[ep2.box].direction += force_2 * step_length

    #     for i1, box1 in enumerate(boxes):
    #         for i2 in range(i1, len(boxes)):
    #             box2 = boxes[i2]
    #             if not box1.is_intersect(box2):
    #                 continue
    #             delta = box1.center - box2.center
    #             if np.linalg.norm(delta) == 0:
    #                 delta = np.random.randn(2)
    #             direction = delta / np.linalg.norm(delta)
    #             force_1 += direction * step_length * 100
    #             force_2 -= direction * step_length * 100

    #     for box, param in box_notes.items():
    #         box.position += param.direction
    #         # param.direction = np.array([0, 0], dtype=np.float64)
    #         param.direction = param.direction * 0.5

    def step2(
        self,
        boxes: list[Box],
        edges: list[Edge],
        *,
        box_notes: dict[Box, BoxOptParams],
        pull_factor: float = 0.5,
        repel_factor=1,
    ):
        """
        pull_factor : 0~0.5
        """
        np.random.seed(42)
        for edge in edges:
            ep1, ep2 = edge.end_points
            best_pair = AlignOpt.cal_best_edge_vec_pair(ep1, ep2)
            delta = (ep2.glo_posi + best_pair[1]) - (ep1.glo_posi + best_pair[0])

            force_1 = pull_factor * delta * self.box_weight(ep1.box)
            force_2 = -pull_factor * delta * self.box_weight(ep2.box)

            box_notes[ep1.box].direction += force_1
            box_notes[ep2.box].direction += force_2

        for i1, box1 in enumerate(boxes):
            for i2 in range(i1 + 1, len(boxes)):
                box2 = boxes[i2]
                if not box1.is_intersect(box2):
                    continue

                delta = box1.center - box2.center
                # delta = self.cal_box_repel(box2, box1)
                if np.linalg.norm(delta) == 0:
                    delta = np.random.randn(2)
                direction = delta / np.linalg.norm(delta)
                box_notes[box1].direction += (
                    direction * repel_factor * self.box_weight(ep1.box)
                )
                box_notes[box2].direction -= (
                    direction * repel_factor * self.box_weight(ep2.box)
                )

        for box, param in box_notes.items():
            box.position += param.direction
            param.direction = np.array([0, 0], dtype=np.float64)
            # param.direction = param.direction * 0.1


class AvoidOverlap(BaseOpt):
    def __init__(self):
        super().__init__()

    def run(self, boxes, edges):
        have_intersection = True
        bar = iter(tqdm(itertools.count(), desc="checking intersection"))
        while have_intersection:
            next(bar)
            have_intersection = False
            for i1, box1 in enumerate(boxes):
                for i2 in range(i1 + 1, len(boxes)):
                    box2 = boxes[i2]
                    if not box1.is_intersect(box2):
                        continue
                    have_intersection = True

                    delta = box1.center - box2.center
                    if np.linalg.norm(delta) == 0:
                        delta = np.random.randn(2)
                    direction = delta / np.linalg.norm(delta)
                    box1.position += direction
                    box2.position -= direction


class NonConnGraphLayout(BaseOpt):
    def __init__(self):
        super().__init__()

    def run(self, boxes: list[Box], edges):
        groups: list[set[Box]] = []
        group = None
        bfs = BfsBoxes(boxes)
        for i in bfs:
            if isinstance(i, EndPoint):
                if group is not None:
                    groups.append(group)
                group = {i.box}
                continue
            group.add(i[0].box)
            group.add(i[1].box)
        if group is not None:
            groups.append(group)

        groups: list[tuple[Box, set[Box]]] = [
            (Box.combined_big_box(i), i) for i in groups
        ]
        groups.sort(key=lambda x: x[0].area)

        if len(groups) == 0:
            return

        big_box, box_set = groups.pop()

        delta = np.array([0, 0], np.float64) - big_box.position

        for i in box_set:
            i.position += delta

        done_boxes = [Box.combined_big_box(box_set)]

        while groups:
            big_box, box_set = groups.pop()

            may_posi = [
                j for i in done_boxes for j in [i.right_down, i.left_top, i.right_top]
            ]
            may_posi.sort(key=lambda x: max(x))

            for i in may_posi:
                for j in may_posi:
                    if j[0] > i[0] and j[1] > i[1]:
                        break
                else:
                    best_post = i
                    break

            delta = best_post - big_box.position
            print(best_post)
            for i in box_set:
                i.position += delta

            done_boxes.append(Box.combined_big_box(box_set))


# class _Old_OptConn1:
#     @staticmethod
#     def cal_to_edge_vecs(conn_posi: np.ndarray, room: Room):
#         """
#         点到房间四个边的向量
#         根据模长排序
#         """
#         rel_align_points = np.array(
#             [
#                 [conn_posi[0], 0],
#                 [conn_posi[0], room.box_size[1]],
#                 [0, conn_posi[1]],
#                 [room.box_size[0], conn_posi[1]],
#             ]
#         )
#         conn_to_edge = rel_align_points - conn_posi

#         return conn_to_edge[np.argsort(np.linalg.norm(conn_to_edge, axis=1))]

#     @staticmethod
#     def cal_to_vertex_vecs(conn_posi: np.ndarray, room: Room):
#         """
#         点到房间四个顶点的向量
#         根据模长排序
#         """
#         rel_align_points = np.array(
#             [
#                 [0, 0],
#                 [0, room.box_size[1]],
#                 [room.box_size[0], 9],
#                 [room.box_size[0], room.box_size[1]],
#             ]
#         )
#         conn_to_edge = rel_align_points - conn_posi

#         return conn_to_edge[np.argsort(np.linalg.norm(conn_to_edge, axis=1))]

#     @staticmethod
#     def cal_vec_relationship(
#         v1: np.ndarray, v2: np.ndarray, tol=1e-6
#     ) -> Literal[-1, 0, 1, None]:
#         """
#         1: 同向
#         -1: 反向
#         0: 垂直
#         None: 其他
#         """
#         v1 = np.asarray(v1)
#         v2 = np.asarray(v2)

#         if np.linalg.norm(v1) < tol or np.linalg.norm(v2) < tol:
#             return None

#         # 单位向量
#         u1 = v1 / np.linalg.norm(v1)
#         u2 = v2 / np.linalg.norm(v2)

#         dot = np.dot(u1, u2)

#         if np.abs(dot - 1) < tol:
#             return 1
#         elif np.abs(dot + 1) < tol:
#             return -1
#         elif np.abs(dot) < tol:
#             return 0
#         else:
#             return None

#     @classmethod
#     def opt_one(cls, conn: Connection, all_rooms: list[Room] = [], move_limit=np.inf):
#         global_conn_posi_1 = conn.room1.box_position + conn.room1_posi
#         global_conn_posi_2 = conn.room2.box_position + conn.room2_posi

#         weight1, weight2 = map(np.prod, (conn.room1.box_size, conn.room2.box_size))

#         r1_edge_vecs = cls.cal_to_edge_vecs(conn.room1_posi, conn.room1)
#         r2_edge_vecs = cls.cal_to_edge_vecs(conn.room2_posi, conn.room2)

#         dis = np.inf
#         for vec1, vec2 in itertools.product(r1_edge_vecs, r2_edge_vecs):
#             if not cls.cal_vec_relationship(vec1, vec2) == -1:
#                 continue
#             new_dis = np.linalg.norm(vec1 - vec2)
#             if not new_dis < dis:
#                 continue
#             dis = new_dis

#             center = (
#                 (conn.room1.box_position + conn.room1_posi + vec1) * weight1
#                 + (conn.room2.box_position + conn.room2_posi + vec2) * weight2
#             ) / (weight1 + weight2)
#             delta_r1 = (center - (conn.room1_posi + vec1)) - conn.room1.box_position
#             delta_r2 = (center - (conn.room2_posi + vec2)) - conn.room2.box_position
#             conn.room1.box_position += utils.uniform(delta_r1) * min(
#                 move_limit, np.linalg.norm(delta_r1)
#             )
#             conn.room2.box_position += utils.uniform(delta_r2) * min(
#                 move_limit, np.linalg.norm(delta_r2)
#             )

#             # for _ in range(10):
#             #     for i in all_rooms:
#             #         if i.name == conn.room1.name:
#             #             continue
#             #         if i.intersects(conn.room1):
#             #             conn.room1.box_position = (
#             #                 conn.room1.box_position * 0.5 + old_room1_posi * 0.5
#             #             )
#             #             break
#             #     else:
#             #         break
#             # for _ in range(10):
#             #     for i in all_rooms:
#             #         if i.name == conn.room2.name:
#             #             continue
#             #         if i.intersects(conn.room2):
#             #             conn.room2.box_position = (
#             #                 conn.room2.box_position * 0.5 + old_room2_posi * 0.5
#             #             )
#             #             break

#     @classmethod
#     def opt_a_lot(
#         cls,
#         room_map: dict[str, Room],
#         connections: list[Connection],
#         opt=0,
#         repel=10000,
#         iter_=100,
#     ):
#         random.seed(42)
#         rooms = list(room_map.values())
#         for i in rooms:
#             i.box_position /= 2
#         weights = [i.norm for i in connections]
#         linespace = np.arange(len(weights))
#         for i in tqdm(range(iter_), leave=False):
#             for _ in range(opt):
#                 conn_idx: int = random.choices(linespace, weights)[0]
#                 conn = connections[conn_idx]
#                 _Old_OptConn1.opt_one(
#                     conn, rooms, move_limit=2000 * (1 - (i / iter_)) ** 2
#                 )

#                 weights[conn_idx] = conn.norm

#             for _ in range(repel):
#                 room1, room2 = random.sample(rooms, k=2)
#                 if room1.intersects(room2):
#                     cls.repel_room(room1, room2)

#     @staticmethod
#     def repel_room(a: Room, b: Room, strength=1):
#         delta = a.box_position - b.box_position
#         if np.linalg.norm(delta) == 0:
#             delta = np.random.randn(2)
#         direction = delta / np.linalg.norm(delta)
#         # a.box_position += direction * strength / np.sqrt(np.prod(a.box_size))
#         # b.box_position += -direction * strength / np.sqrt(np.prod(b.box_size))
#         a.box_position += direction * strength
#         b.box_position += -direction * strength


# class _Old_OptConn0:
#     @staticmethod
#     def cal_to_edge_vecs(conn_posi: np.ndarray, room: Room):
#         """
#         点到房间四个边的向量
#         根据模长排序
#         """
#         rel_align_points = np.array(
#             [
#                 [conn_posi[0], 0],
#                 [conn_posi[0], room.box_size[1]],
#                 [0, conn_posi[1]],
#                 [room.box_size[0], conn_posi[1]],
#             ]
#         )
#         conn_to_edge = rel_align_points - conn_posi

#         return conn_to_edge[np.argsort(np.linalg.norm(conn_to_edge, axis=1))]

#     @staticmethod
#     def cal_to_vertex_vecs(conn_posi: np.ndarray, room: Room):
#         """
#         点到房间四个顶点的向量
#         根据模长排序
#         """
#         rel_align_points = np.array(
#             [
#                 [0, 0],
#                 [0, room.box_size[1]],
#                 [room.box_size[0], 9],
#                 [room.box_size[0], room.box_size[1]],
#             ]
#         )
#         conn_to_edge = rel_align_points - conn_posi

#         return conn_to_edge[np.argsort(np.linalg.norm(conn_to_edge, axis=1))]

#     @staticmethod
#     def cal_vec_relationship(
#         v1: np.ndarray, v2: np.ndarray, tol=1e-6
#     ) -> Literal[-1, 0, 1, None]:
#         """
#         1: 同向
#         -1: 反向
#         0: 垂直
#         None: 其他
#         """
#         v1 = np.asarray(v1)
#         v2 = np.asarray(v2)

#         if np.linalg.norm(v1) < tol or np.linalg.norm(v2) < tol:
#             return None

#         # 单位向量
#         u1 = v1 / np.linalg.norm(v1)
#         u2 = v2 / np.linalg.norm(v2)

#         dot = np.dot(u1, u2)

#         if np.abs(dot - 1) < tol:
#             return 1
#         elif np.abs(dot + 1) < tol:
#             return -1
#         elif np.abs(dot) < tol:
#             return 0
#         else:
#             return None

#     @classmethod
#     def opt_one(cls, conn: Connection, all_rooms: list[Room] = []):
#         global_conn_posi_1 = conn.room1.box_position + conn.room1_posi
#         global_conn_posi_2 = conn.room2.box_position + conn.room2_posi

#         weight1, weight2 = map(np.prod, (conn.room1.box_size, conn.room2.box_size))

#         r1_edge_vecs = cls.cal_to_edge_vecs(conn.room1_posi, conn.room1)
#         r2_edge_vecs = cls.cal_to_edge_vecs(conn.room2_posi, conn.room2)

#         dis = np.inf
#         for vec1, vec2 in itertools.product(r1_edge_vecs, r2_edge_vecs):
#             if not cls.cal_vec_relationship(vec1, vec2) == -1:
#                 continue
#             new_dis = np.linalg.norm(vec1 - vec2)
#             if not new_dis < dis:
#                 continue
#             dis = new_dis

#             center = (
#                 (conn.room1.box_position + conn.room1_posi + vec1) * weight1
#                 + (conn.room2.box_position + conn.room2_posi + vec2) * weight2
#             ) / (weight1 + weight2)
#             old_room1_posi = conn.room1.box_position
#             old_room2_posi = conn.room2.box_position
#             conn.room1.box_position = center - (conn.room1_posi + vec1)
#             conn.room2.box_position = center - (conn.room2_posi + vec2)

#             # for _ in range(10):
#             #     for i in all_rooms:
#             #         if i.name == conn.room1.name:
#             #             continue
#             #         if i.intersects(conn.room1):
#             #             conn.room1.box_position = (
#             #                 conn.room1.box_position * 0.5 + old_room1_posi * 0.5
#             #             )
#             #             break
#             #     else:
#             #         break
#             # for _ in range(10):
#             #     for i in all_rooms:
#             #         if i.name == conn.room2.name:
#             #             continue
#             #         if i.intersects(conn.room2):
#             #             conn.room2.box_position = (
#             #                 conn.room2.box_position * 0.5 + old_room2_posi * 0.5
#             #             )
#             #             break

#     @classmethod
#     def opt_a_lot(
#         cls,
#         room_map: dict[str, Room],
#         connections: list[Connection],
#         iter=10000,
#         repel=100000,
#     ):
#         random.seed(42)
#         rooms = list(room_map.values())
#         weights = [i.norm for i in connections]
#         linespace = np.arange(len(weights))
#         for _ in tqdm(range(iter)):
#             conn_idx: int = random.choices(linespace, weights)[0]
#             conn = connections[conn_idx]
#             _Old_OptConn0.opt_one(conn, rooms)

#             weights[conn_idx] = conn.norm

#         for _ in tqdm(range(repel)):
#             room1, room2 = random.sample(rooms, k=2)
#             if room1.intersects(room2):
#                 cls.repel_room(room1, room2, strength=5)

#     @staticmethod
#     def repel_room(a: Room, b: Room, strength=10):
#         delta = a.box_position - b.box_position
#         if np.linalg.norm(delta) == 0:
#             delta = np.random.randn(2)
#         direction = delta / np.linalg.norm(delta)
#         a.box_position += direction * strength


if __name__ == "__main__":
    main()
