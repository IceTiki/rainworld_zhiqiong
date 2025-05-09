from pathlib import Path
from typing import Literal, Generator, Callable, Hashable, TypeVar, Generic
from dataclasses import dataclass, field
from collections import deque
import math
import typing
import itertools


from loguru import logger
from numpy.linalg import norm
from pprint import pprint
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
import numpy as np

from colls import Box, Edge, EndPoint

DEBUG = False


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

    @logger.catch
    def _test_plot(
        self,
        boxes: list[Box],
        edges: list[Edge],
        output: Path | None | typing.Literal[False] = None,
    ):
        if not DEBUG:
            return
        if output is None:
            import constants as cons
            import time

            output = cons.TMP_OUTPUT_PATH / f"{time.time_ns()}.png"
            self._TEST_PLOT_COUNT += 1
        if isinstance(output, Path):
            output.parent.mkdir(exist_ok=True, parents=True)

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
        logger.debug("Start AlignOpt.")
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
            self.repel2(ep_main.box, ep_sub.box, searcher.done_boxes)
            self._test_plot(boxes, edges)
            i += 1
        logger.debug("End AlignOpt.")

    @staticmethod
    def cal_best_edge_vec_pair(end_point_main: EndPoint, end_point_sub: EndPoint):
        cbkey = [
            (end_point_main.distance_left + end_point_sub.distance_right, 0),
            (end_point_main.distance_right + end_point_sub.distance_left, 1),
            (end_point_main.distance_up + end_point_sub.distance_down, 2),
            (end_point_main.distance_down + end_point_sub.distance_up, 3),
        ]
        cbkey.sort()
        key = cbkey[0][1]
        if key == 0:
            return np.array([end_point_main.distance_left, 0]), np.array(
                [-end_point_sub.distance_right, 0]
            )
        elif key == 1:
            return (
                np.array([-end_point_main.distance_right, 0]),
                np.array([end_point_sub.distance_left, 0]),
            )
        elif key == 2:
            return (
                np.array([0, end_point_main.distance_up]),
                np.array([0, -end_point_sub.distance_down]),
            )
        elif key == 3:
            return (
                np.array([0, -end_point_main.distance_down]),
                np.array([0, end_point_sub.distance_up]),
            )

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

    def __init__(self, iters: int | None = None, repel_factor=10):
        super().__init__()
        self.iters: int | None = iters
        self.repel_factor = repel_factor

    def run(self, boxes: list[Box], edges: list[Edge]):
        logger.debug("Start ForceOpt.")
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
                repel_factor=ip * self.repel_factor,
            )
            if i % 20 == 0:
                self._test_plot(boxes, edges)

        logger.debug("End ForceOpt.")

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


class AvoidOverlap(BaseOpt):
    def __init__(self, step=1):
        super().__init__()
        self.step = step

    def run(self, boxes, edges):
        logger.debug("Start AvoidOverlap.")
        have_intersection = True
        bar = iter(tqdm(itertools.count(), desc="checking intersection"))
        while have_intersection:
            self._test_plot(boxes, edges)
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
                    box1.position += direction * self.step
                    box2.position -= direction * self.step

        logger.debug("End AvoidOverlap.")


class NonConnGraphLayout(BaseOpt):
    # TODO: 堆叠方式可能有问题
    def __init__(self):
        super().__init__()

    def run(self, boxes: list[Box], edges):
        logger.debug("Start NonConnGraphLayout.")
        self._test_plot(boxes, edges)
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
            for i in box_set:
                i.position += delta

            done_boxes.append(Box.combined_big_box(box_set))

        self._test_plot(boxes, edges)
        logger.debug("End NonConnGraphLayout.")


_T_Box = TypeVar("_T_Box", bound=Box)


class BoxPositionCache(BaseOpt, Generic[_T_Box]):
    _T_BoxKey = TypeVar("_T_BoxKey", bound=Hashable)
    _T_Vec2d = list[float]
    _T_Cache = dict[_T_BoxKey, _T_Vec2d]

    def __init__(
        self,
        opts: list[BaseOpt],
        box_key: Callable[[_T_Box], _T_BoxKey],
        position_cache: _T_Cache | None = None,
    ):
        super().__init__()
        self.opts: list[BaseOpt] = opts
        self.position_cache: BoxPositionCache._T_Cache | None = position_cache
        self.box_key = box_key

    def run(self, boxes: list[_T_Box], edges: list[Edge]):
        if self.position_cache is None:
            for opt in self.opts:
                opt.run(boxes=boxes, edges=edges)
            self.position_cache = {self.box_key(i): list(i.position) for i in boxes}
            return

        for box in boxes:
            key = self.box_key(box)
            box.position = np.array(self.position_cache[key], np.float64)
