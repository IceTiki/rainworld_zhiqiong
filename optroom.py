import typing
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import constants as c
from typing import TypedDict
from matplotlib.patches import Rectangle
from dataclasses import dataclass
from loguru import logger
import random
from typing import Literal, Optional
from tqdm import tqdm
from itertools import product
from pprint import pprint
import utils
import math

if typing.TYPE_CHECKING:
    from rooms import Room, Connection


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
        for vec1, vec2 in product(r1_edge_vecs, r2_edge_vecs):
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
        for vec1, vec2 in product(r1_edge_vecs, r2_edge_vecs):
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
