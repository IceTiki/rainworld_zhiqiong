import typing
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import constants as c
from typing import TypedDict, TypeVar, Literal, Optional, Iterable, Annotated, Generator
from matplotlib.patches import Rectangle
from dataclasses import dataclass
from loguru import logger
import random
from tqdm import tqdm
from pprint import pprint
import utils
import math
import itertools
from numpy.typing import NDArray
from numpy.linalg import norm

if typing.TYPE_CHECKING:
    from rooms import Room, Connection
else:
    Room = TypeVar("Room")
    Connection = TypeVar("Connection")


Vec2d = NDArray[np.float64]
RoomMap = dict[Room, "Box"]
ConnectionMap = dict[Connection, "Edge"]


class EndPoint:
    def __init__(self, box: "Box", posi: Vec2d):
        self.box: Box = box
        self.ep_posi: Vec2d = posi

    @property
    def distance_left(self):
        return self.ep_posi[0]

    @property
    def distance_down(self):
        return self.ep_posi[1]

    @property
    def distance_right(self):
        return self.box.size[0] - self.ep_posi[0]

    @property
    def distance_top(self):
        return self.box.size[1] - self.ep_posi[1]

    def to_edge_vecs(self):
        return np.array(
            [
                [-self.distance_left, 0],
                [self.distance_right, 0],
                [0, -self.distance_down],
                [0, self.distance_top],
            ]
        )

    def to_vertex_vecs(self):
        return np.array(
            [
                [-self.distance_left, -self.distance_down],
                [self.distance_right, -self.distance_down],
                [-self.distance_left, self.distance_top],
                [self.distance_right, self.distance_top],
            ]
        )


class Edge:
    def __init__(self, end_point_1: EndPoint, end_point_2: EndPoint):
        self.end_points: list[EndPoint] = [end_point_1, end_point_2]

    @property
    def boxes(self):
        return [i.box for i in self.end_points]


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

    @classmethod
    def from_room(cls, room: Room):
        return cls(room.box_position.copy(), room.box_size.copy())

    @classmethod
    def build_graph(
        cls, rooms: Iterable[Room], conns: Iterable[Connection]
    ) -> tuple[RoomMap, ConnectionMap]:
        room_map: RoomMap = {r: cls.from_room(r) for r in rooms}
        conn_map: ConnectionMap = {}
        for conn in conns:
            for room in (conn.room1, conn.room2):
                if room not in room_map:
                    room_map[room] = cls.from_room(room)

            edge = Edge(
                EndPoint(room_map[conn.room1], conn.room1_posi),
                EndPoint(room_map[conn.room2], conn.room2_posi),
            )
            conn_map[conn] = edge
            for room in (conn.room1, conn.room2):
                box = room_map[room]
                box.append_edge(edge)

        return room_map, conn_map

    def __init__(self, position: Vec2d, size: Vec2d, edges: list[Edge] = None):
        self.position: Vec2d = position
        self.size: Vec2d = size
        self.edges: list[Edge] = edges if edges is not None else []

    @property
    def area(self):
        return np.prod(self.size)

    @property
    def left_down(self):
        return self.position

    @property
    def right_top(self):
        return self.position + self.size

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


class BaseOpt:
    @staticmethod
    def use_opt(
        opt: "BaseOpt", rooms: Iterable[Room], connections: Iterable[Connection]
    ):
        room_map, conn_map = Box.build_graph(rooms, connections)
        opt(list(room_map.values()), list(conn_map.values()))
        for room, box in room_map.items():
            room.box_position = box.position

    def __init__(self, *, boxes: list[Box] = None, edges: list[Edge] = None):
        self.boxes: list[Box] = boxes if boxes is not None else []
        self.edges: list[Edge] = edges if edges is not None else []

    def __call__(self, boxes: list[Box], edges: list[Edge]) -> None:
        self.boxes: list[Box] = boxes
        self.edges: list[Edge] = edges
        self.run()

    def run(self) -> None:
        return


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
        return max(self.todo_boxes, key=lambda x: x.area)

    def __iter__(self) -> Generator[tuple[EndPoint, EndPoint], None, None]:
        stack: list[Edge] = []
        # BFS
        old_start = None
        while True:
            if not stack:
                if not self.todo_boxes:
                    break
                start_box = self.choice()
                stack.extend(start_box.edges)
                if old_start is not None:
                    yield EndPoint(old_start, np.array([0, 0])), EndPoint(
                        start_box, np.array([0, 0])
                    )

                self.todo_boxes.remove(start_box)
                old_start = start_box

            edge = stack.pop()
            if edge in self.done_edges:
                continue
            self.done_edges.add(edge)
            for i, end_point in enumerate(edge.end_points):
                box = end_point.box
                if box not in self.todo_boxes:
                    continue
                self.todo_boxes.remove(box)
                stack.extend(box.edges)

                yield edge.end_points[1 - i], end_point


class InitOpt(BaseOpt):
    def run(self):
        bfs = BfsBoxes(self.boxes)
        for ep_main, ep_sub in bfs:
            self.align(ep_main, ep_sub)
            self.repel(ep_sub, bfs.done_boxes)

    def align(self, end_point_main: EndPoint, end_point_sub: EndPoint):
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
            + end_point_main.ep_posi
            + best_conn
            - end_point_sub.ep_posi
        )

    def repel(
        self,
        end_point: EndPoint,
        fixed_boxes: set[Box],
        step=1,
    ):
        box = end_point.box
        have_intersect = True

        count_ = 0
        random.seed(42)
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


class _Old_OptConn1:
    @staticmethod
    def cal_to_edge_vecs(conn_posi: np.ndarray, room: Room):
        """
        点到房间四个边的向量
        根据模长排序
        """
        rel_align_points = np.array(
            [
                [conn_posi[0], 0],
                [conn_posi[0], room.box_size[1]],
                [0, conn_posi[1]],
                [room.box_size[0], conn_posi[1]],
            ]
        )
        conn_to_edge = rel_align_points - conn_posi

        return conn_to_edge[np.argsort(np.linalg.norm(conn_to_edge, axis=1))]

    @staticmethod
    def cal_to_vertex_vecs(conn_posi: np.ndarray, room: Room):
        """
        点到房间四个顶点的向量
        根据模长排序
        """
        rel_align_points = np.array(
            [
                [0, 0],
                [0, room.box_size[1]],
                [room.box_size[0], 9],
                [room.box_size[0], room.box_size[1]],
            ]
        )
        conn_to_edge = rel_align_points - conn_posi

        return conn_to_edge[np.argsort(np.linalg.norm(conn_to_edge, axis=1))]

    @staticmethod
    def cal_vec_relationship(
        v1: np.ndarray, v2: np.ndarray, tol=1e-6
    ) -> Literal[-1, 0, 1, None]:
        """
        1: 同向
        -1: 反向
        0: 垂直
        None: 其他
        """
        v1 = np.asarray(v1)
        v2 = np.asarray(v2)

        if np.linalg.norm(v1) < tol or np.linalg.norm(v2) < tol:
            return None

        # 单位向量
        u1 = v1 / np.linalg.norm(v1)
        u2 = v2 / np.linalg.norm(v2)

        dot = np.dot(u1, u2)

        if np.abs(dot - 1) < tol:
            return 1
        elif np.abs(dot + 1) < tol:
            return -1
        elif np.abs(dot) < tol:
            return 0
        else:
            return None

    @classmethod
    def opt_one(cls, conn: Connection, all_rooms: list[Room] = [], move_limit=np.inf):
        global_conn_posi_1 = conn.room1.box_position + conn.room1_posi
        global_conn_posi_2 = conn.room2.box_position + conn.room2_posi

        weight1, weight2 = map(np.prod, (conn.room1.box_size, conn.room2.box_size))

        r1_edge_vecs = cls.cal_to_edge_vecs(conn.room1_posi, conn.room1)
        r2_edge_vecs = cls.cal_to_edge_vecs(conn.room2_posi, conn.room2)

        dis = np.inf
        for vec1, vec2 in itertools.product(r1_edge_vecs, r2_edge_vecs):
            if not cls.cal_vec_relationship(vec1, vec2) == -1:
                continue
            new_dis = np.linalg.norm(vec1 - vec2)
            if not new_dis < dis:
                continue
            dis = new_dis

            center = (
                (conn.room1.box_position + conn.room1_posi + vec1) * weight1
                + (conn.room2.box_position + conn.room2_posi + vec2) * weight2
            ) / (weight1 + weight2)
            delta_r1 = (center - (conn.room1_posi + vec1)) - conn.room1.box_position
            delta_r2 = (center - (conn.room2_posi + vec2)) - conn.room2.box_position
            conn.room1.box_position += utils.uniform(delta_r1) * min(
                move_limit, np.linalg.norm(delta_r1)
            )
            conn.room2.box_position += utils.uniform(delta_r2) * min(
                move_limit, np.linalg.norm(delta_r2)
            )

            # for _ in range(10):
            #     for i in all_rooms:
            #         if i.name == conn.room1.name:
            #             continue
            #         if i.intersects(conn.room1):
            #             conn.room1.box_position = (
            #                 conn.room1.box_position * 0.5 + old_room1_posi * 0.5
            #             )
            #             break
            #     else:
            #         break
            # for _ in range(10):
            #     for i in all_rooms:
            #         if i.name == conn.room2.name:
            #             continue
            #         if i.intersects(conn.room2):
            #             conn.room2.box_position = (
            #                 conn.room2.box_position * 0.5 + old_room2_posi * 0.5
            #             )
            #             break

    @classmethod
    def opt_a_lot(
        cls,
        room_map: dict[str, Room],
        connections: list[Connection],
        opt=0,
        repel=10000,
        iter_=100,
    ):
        random.seed(42)
        rooms = list(room_map.values())
        for i in rooms:
            i.box_position /= 2
        weights = [i.norm for i in connections]
        linespace = np.arange(len(weights))
        for i in tqdm(range(iter_), leave=False):
            for _ in range(opt):
                conn_idx: int = random.choices(linespace, weights)[0]
                conn = connections[conn_idx]
                _Old_OptConn1.opt_one(
                    conn, rooms, move_limit=2000 * (1 - (i / iter_)) ** 2
                )

                weights[conn_idx] = conn.norm

            for _ in range(repel):
                room1, room2 = random.sample(rooms, k=2)
                if room1.intersects(room2):
                    cls.repel_room(room1, room2)

    @staticmethod
    def repel_room(a: Room, b: Room, strength=1):
        delta = a.box_position - b.box_position
        if np.linalg.norm(delta) == 0:
            delta = np.random.randn(2)
        direction = delta / np.linalg.norm(delta)
        # a.box_position += direction * strength / np.sqrt(np.prod(a.box_size))
        # b.box_position += -direction * strength / np.sqrt(np.prod(b.box_size))
        a.box_position += direction * strength
        b.box_position += -direction * strength


class _Old_OptConn0:
    @staticmethod
    def cal_to_edge_vecs(conn_posi: np.ndarray, room: Room):
        """
        点到房间四个边的向量
        根据模长排序
        """
        rel_align_points = np.array(
            [
                [conn_posi[0], 0],
                [conn_posi[0], room.box_size[1]],
                [0, conn_posi[1]],
                [room.box_size[0], conn_posi[1]],
            ]
        )
        conn_to_edge = rel_align_points - conn_posi

        return conn_to_edge[np.argsort(np.linalg.norm(conn_to_edge, axis=1))]

    @staticmethod
    def cal_to_vertex_vecs(conn_posi: np.ndarray, room: Room):
        """
        点到房间四个顶点的向量
        根据模长排序
        """
        rel_align_points = np.array(
            [
                [0, 0],
                [0, room.box_size[1]],
                [room.box_size[0], 9],
                [room.box_size[0], room.box_size[1]],
            ]
        )
        conn_to_edge = rel_align_points - conn_posi

        return conn_to_edge[np.argsort(np.linalg.norm(conn_to_edge, axis=1))]

    @staticmethod
    def cal_vec_relationship(
        v1: np.ndarray, v2: np.ndarray, tol=1e-6
    ) -> Literal[-1, 0, 1, None]:
        """
        1: 同向
        -1: 反向
        0: 垂直
        None: 其他
        """
        v1 = np.asarray(v1)
        v2 = np.asarray(v2)

        if np.linalg.norm(v1) < tol or np.linalg.norm(v2) < tol:
            return None

        # 单位向量
        u1 = v1 / np.linalg.norm(v1)
        u2 = v2 / np.linalg.norm(v2)

        dot = np.dot(u1, u2)

        if np.abs(dot - 1) < tol:
            return 1
        elif np.abs(dot + 1) < tol:
            return -1
        elif np.abs(dot) < tol:
            return 0
        else:
            return None

    @classmethod
    def opt_one(cls, conn: Connection, all_rooms: list[Room] = []):
        global_conn_posi_1 = conn.room1.box_position + conn.room1_posi
        global_conn_posi_2 = conn.room2.box_position + conn.room2_posi

        weight1, weight2 = map(np.prod, (conn.room1.box_size, conn.room2.box_size))

        r1_edge_vecs = cls.cal_to_edge_vecs(conn.room1_posi, conn.room1)
        r2_edge_vecs = cls.cal_to_edge_vecs(conn.room2_posi, conn.room2)

        dis = np.inf
        for vec1, vec2 in itertools.product(r1_edge_vecs, r2_edge_vecs):
            if not cls.cal_vec_relationship(vec1, vec2) == -1:
                continue
            new_dis = np.linalg.norm(vec1 - vec2)
            if not new_dis < dis:
                continue
            dis = new_dis

            center = (
                (conn.room1.box_position + conn.room1_posi + vec1) * weight1
                + (conn.room2.box_position + conn.room2_posi + vec2) * weight2
            ) / (weight1 + weight2)
            old_room1_posi = conn.room1.box_position
            old_room2_posi = conn.room2.box_position
            conn.room1.box_position = center - (conn.room1_posi + vec1)
            conn.room2.box_position = center - (conn.room2_posi + vec2)

            # for _ in range(10):
            #     for i in all_rooms:
            #         if i.name == conn.room1.name:
            #             continue
            #         if i.intersects(conn.room1):
            #             conn.room1.box_position = (
            #                 conn.room1.box_position * 0.5 + old_room1_posi * 0.5
            #             )
            #             break
            #     else:
            #         break
            # for _ in range(10):
            #     for i in all_rooms:
            #         if i.name == conn.room2.name:
            #             continue
            #         if i.intersects(conn.room2):
            #             conn.room2.box_position = (
            #                 conn.room2.box_position * 0.5 + old_room2_posi * 0.5
            #             )
            #             break

    @classmethod
    def opt_a_lot(
        cls,
        room_map: dict[str, Room],
        connections: list[Connection],
        iter=10000,
        repel=100000,
    ):
        random.seed(42)
        rooms = list(room_map.values())
        weights = [i.norm for i in connections]
        linespace = np.arange(len(weights))
        for _ in tqdm(range(iter)):
            conn_idx: int = random.choices(linespace, weights)[0]
            conn = connections[conn_idx]
            _Old_OptConn0.opt_one(conn, rooms)

            weights[conn_idx] = conn.norm

        for _ in tqdm(range(repel)):
            room1, room2 = random.sample(rooms, k=2)
            if room1.intersects(room2):
                cls.repel_room(room1, room2, strength=5)

    @staticmethod
    def repel_room(a: Room, b: Room, strength=10):
        delta = a.box_position - b.box_position
        if np.linalg.norm(delta) == 0:
            delta = np.random.randn(2)
        direction = delta / np.linalg.norm(delta)
        a.box_position += direction * strength


if __name__ == "__main__":
    main()
