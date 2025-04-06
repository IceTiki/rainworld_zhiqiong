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

plt.rcParams["font.sans-serif"] = ["MicroSoft YaHei"]


def rgba_pixel(color: str = "#ffffff", alpha: float = 1):
    carr = [int(color[i : i + 2], 16) for i in (1, 3, 5)] + [alpha * 255]
    return np.array([[carr]], np.uint8)


class RoomTxt:
    CHANNEL = 8 + 1

    @classmethod
    def from_file(cls, path: str | Path):
        return cls(Path(path).read_text())

    def __init__(self, text: str):
        self.text = text
        self.lines = text.splitlines()
        self.name = self.lines[0].upper()

        size, water_level, _ = self.lines[1].split("|")
        self.width, self.height = map(int, size.split("*"))
        self.water_level = int(water_level)

        # ===
        map_str = self.lines[11].split("|")[:-1]
        map_num = [tuple(map(int, i.split(","))) for i in map_str]
        map_layer = np.array(
            [list(map(lambda x: x in i, range(self.CHANNEL))) for i in map_num],
            dtype=np.uint8,
        )
        self.map_matrix = np.reshape(
            map_layer,
            (self.width, self.height, self.CHANNEL),
        ).transpose((1, 0, 2))
        """
        layer
        0 : 可活动区域?
        1 : 墙体?
        2 : 横向钢筋?
        3 : 管道路线
        4 : 管道入口
        5 : 废弃管道入口?
        6 : 背景?
        7 : ?
        8 : 废弃管道？
        """

        water = np.zeros((self.height, self.width, 1), dtype=np.uint8)
        if self.water_level >= 0:
            water[-self.water_level :, :, :] = 1
        self.water_mask = water

    @property
    def wall_im(self):
        return self.map_matrix[:, :, 1:2]

    @property
    def background_im(self):
        return self.map_matrix[:, :, 6:7]

    @property
    def entrance_im(self):
        return self.map_matrix[:, :, 4:5]

    @property
    def pipe_im(self):
        return self.map_matrix[:, :, 3:4]

    @property
    def bar_im(self):
        return self.wall_im & self.background_im | self.map_matrix[:, :, 2:3]

    def test_show_all(self):

        # fig, axs = plt.subplots(
        #     nrows=self.CHANNEL // 4 + 1,
        #     ncols=4,
        #     figsize=(16, 16),
        #     constrained_layout=True,
        # )
        # ax_list: list[plt.Axes] = axs.flatten().tolist()

        # for i, ax in zip(range(self.CHANNEL), ax_list):
        #     ax.imshow(self.map_matrix[:, :, i], cmap="gray")
        #     ax.set_title(str(i))
        #     ax.axis("off")

        # ax = ax_list[-1]
        fig, ax = plt.subplots()
        ax: plt.Axes
        ax.imshow(self.water_mask * rgba_pixel("#007AAE"))
        ax.imshow(self.wall_im * rgba_pixel("#000000"))
        ax.imshow(self.background_im * rgba_pixel("#000000", 0.25))
        ax.imshow(self.pipe_im * rgba_pixel("#ffff00", 0.25))
        ax.imshow(self.entrance_im * rgba_pixel("#ffff00", 0.5))
        ax.imshow(self.bar_im * rgba_pixel("#880015", 1))
        ax.imshow(self.map_matrix[:, :, 5:6] * rgba_pixel("#A349A4", 1))
        ax.imshow(self.map_matrix[:, :, 8:9] * rgba_pixel("#3F48CC", 1))
        plt.show()


class RoomSettingTxt:
    @classmethod
    def from_file(cls, path: str | Path):
        return cls(Path(path).read_text())

    def __init__(self, text: str):
        self.text = text
        self.lines = text.splitlines()
        self.name = self.lines[0].upper()
        self.data: dict[str, str] = {}
        for line in self.lines:
            if line == "":
                continue
            key, value = line.split(": ")
            self.data[key] = value

        self.placed_objects = []
        if "PlacedObjects" in self.data:
            for obj in self.data["PlacedObjects"].split(", "):
                if obj == "":
                    continue
                name, x, y, other = obj.split("><")
                x, y = map(float, (x, y))
                other = other.split("~")
                self.placed_objects.append([name, x, y, other])


class Room:
    def __init__(
        self,
        roomtxt: RoomTxt,
        box_position: np.ndarray = np.array([0, 0]),
        box_size: np.ndarray = None,
        *,
        room_setting: RoomSettingTxt | None = None,
    ):
        self.rootxt = roomtxt
        self.box_position = box_position
        self.box_size = (
            np.array([self.width, self.height]) if box_size is None else box_size
        )
        self.room_setting = room_setting

    def intersects(self, other: "Room") -> bool:
        ax1, ay1 = self.box_position
        ax2, ay2 = self.box_position + self.box_size

        bx1, by1 = other.box_position
        bx2, by2 = other.box_position + other.box_size

        return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

    @property
    def name(self):
        return self.rootxt.name

    @property
    def en_name(self):
        return c.ZONE_ID_2_EN.get(self.name.upper(), self.name.upper())

    @property
    def cn_name(self):
        return c.EN_2_CN.get(self.en_name, self.en_name)

    @property
    def width(self):
        return self.rootxt.width

    @property
    def height(self):
        return self.rootxt.height

    @property
    def box_extent(self):
        return np.array(
            [
                self.box_position[0],
                self.box_position[0] + self.box_size[0],
                self.box_position[1],
                self.box_position[1] + self.box_size[1],
            ]
        )

    def plot(self, ax: plt.Axes):
        extent = self.box_extent
        ax.imshow(self.rootxt.water_mask * rgba_pixel("#007AAE"), extent=extent)
        ax.imshow(self.rootxt.wall_im * rgba_pixel("#000000"), extent=extent)
        ax.imshow(
            self.rootxt.background_im * rgba_pixel("#000000", 0.25), extent=extent
        )
        ax.imshow(self.rootxt.pipe_im * rgba_pixel("#ffff00", 0.25), extent=extent)
        ax.imshow(self.rootxt.entrance_im * rgba_pixel("#ffff00", 0.5), extent=extent)
        ax.imshow(self.rootxt.bar_im * rgba_pixel("#880015", 1), extent=extent)
        ax.imshow(
            self.rootxt.map_matrix[:, :, 5:6] * rgba_pixel("#A349A4", 1), extent=extent
        )
        ax.imshow(
            self.rootxt.map_matrix[:, :, 8:9] * rgba_pixel("#3F48CC", 1), extent=extent
        )

        rect = Rectangle(
            self.box_position,
            self.width,
            self.height,
            linewidth=1,
            edgecolor="#138535",
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.text(*self.box_position, self.cn_name, c="yellow", alpha=1, fontsize=3)

        if self.room_setting is not None:
            for obj in self.room_setting.placed_objects:
                name, x, y = obj[:3]
                property: list[str] = obj[-1]

                comments = ""
                fontsize = 3
                color = "#FF7F27"
                if name in {"SpinningTopSpot", "WarpPoint"}:
                    name = c.PLACE_OBJECT_NAME[name]
                    target_map = c.zone_id_2_cn(property[4])
                    target_room = property[5].upper()
                    comments = f"({target_map}{target_room})"
                elif name in c.PLACE_OBJECT_NAME:
                    name = c.PLACE_OBJECT_NAME[name]
                else:
                    continue

                x = self.box_position[0] + x / 20
                y = self.box_position[1] + y / 20
                ax.text(x, y, f"{name}{comments}", fontsize=fontsize, c=color)


class MapTxt:
    class _Room(TypedDict):
        room_name: str
        canon_pos_x: float
        canon_pos_y: float
        dev_pos_x: float
        dev_pos_y: float
        layer: int
        subregion_name: str
        room_width: int
        room_height: int

    class _Connection(TypedDict):
        room_name_1: str
        room_name_2: str
        room_pos_1_x: int
        room_pos_1_y: int
        room_pos_2_x: int
        room_pos_2_y: int
        room_1_direction: int
        room_2_direction: int

    @classmethod
    def from_file(cls, path: str | Path):
        return cls(Path(path).read_text())

    def __init__(self, text: str):
        self.text = text
        self.lines = text.splitlines()
        self.rooms: list[MapTxt._Room] = []
        self.connections: list[MapTxt._Connection] = []

        for line in self.lines:
            head, content = line.split(": ")
            if head == "Connection":
                (
                    room_name_1,
                    room_name_2,
                    room_pos_1_x,
                    room_pos_1_y,
                    room_pos_2_x,
                    room_pos_2_y,
                    room_1_direction,
                    room_2_direction,
                ) = content.split(",")
                (
                    room_pos_1_x,
                    room_pos_1_y,
                    room_pos_2_x,
                    room_pos_2_y,
                    room_1_direction,
                    room_2_direction,
                ) = map(
                    int,
                    (
                        room_pos_1_x,
                        room_pos_1_y,
                        room_pos_2_x,
                        room_pos_2_y,
                        room_1_direction,
                        room_2_direction,
                    ),
                )
                self.connections.append(
                    {
                        "room_name_1": room_name_1.upper(),
                        "room_name_2": room_name_2.upper(),
                        "room_pos_1_x": room_pos_1_x,
                        "room_pos_1_y": room_pos_1_y,
                        "room_pos_2_x": room_pos_2_x,
                        "room_pos_2_y": room_pos_2_y,
                        "room_1_direction": room_1_direction,
                        "room_2_direction": room_2_direction,
                    }
                )
                continue
            (
                canon_pos_x,
                canon_pos_y,
                dev_pos_x,
                dev_pos_y,
                layer,
                subregion_name,
                room_width,
                room_height,
            ) = content.split("><")
            canon_pos_x, canon_pos_y, dev_pos_x, dev_pos_y = map(
                float, (canon_pos_x, canon_pos_y, dev_pos_x, dev_pos_y)
            )
            layer, room_width, room_height = map(int, (layer, room_width, room_height))
            self.rooms.append(
                {
                    "room_name": head.upper(),
                    "canon_pos_x": canon_pos_x,
                    "canon_pos_y": canon_pos_y,
                    "dev_pos_x": dev_pos_x,
                    "dev_pos_y": dev_pos_y,
                    "layer": layer,
                    "subregion_name": subregion_name,
                    "room_width": room_width,
                    "room_height": room_height,
                }
            )


class Connection:
    DIRECTION_MAP = {
        0: np.array([1, 0]),  # +x
        1: np.array([0, 1]),  # +y
        2: np.array([-1, 0]),  # -x
        3: np.array([0, -1]),  # -y
    }

    def __init__(
        self,
        room1: Room,
        room2: Room,
        room1_posi: np.ndarray,
        room2_posi: np.ndarray,
        room1_direct: int,
        room2_direct: int,
    ):
        self.room1 = room1
        self.room2 = room2
        self.room1_posi = room1_posi
        self.room2_posi = room2_posi
        self.room1_direct = room1_direct
        self.room2_direct = room2_direct

    @property
    def norm(self):
        start = self.room1.box_position + self.room1_posi
        end = self.room2.box_position + self.room2_posi
        return np.linalg.norm(start - end)

    def plot(self, ax: plt.Axes):
        start = self.room1.box_position + self.room1_posi
        end = self.room2.box_position + self.room2_posi
        ax.plot([start[0], end[0]], [start[1], end[1]],             c="#54C1F0",
            linestyle="-",
            linewidth=1,
            alpha=0.5)


def load_maptxt(world_path: Path = c.WORLD_PATH, name="wara"):
    name = name.upper()
    map_txt = world_path / name.lower() / f"map_{name.lower()}.txt"
    room_folder = world_path / f"{name.lower()}-rooms"

    room_map: dict[str, Room] = {}
    map_txt = MapTxt.from_file(map_txt)
    for room in map_txt.rooms:
        room_txt_path = room_folder / (room["room_name"].lower() + ".txt")
        room_setting_path = room_folder / (room["room_name"].lower() + "_settings.txt")
        if not room_txt_path.is_file():
            logger.warning(f"{room_txt_path} not exists")
            continue
        room_map[room["room_name"].upper()] = Room(
            RoomTxt.from_file(room_txt_path),
            np.array([room["canon_pos_x"], room["canon_pos_y"]]),
            room_setting=(
                RoomSettingTxt.from_file(room_setting_path)
                if room_setting_path.is_file()
                else None
            ),
        )

    connections: list[Connection] = []
    for conn in map_txt.connections:
        if (
            conn["room_name_1"].upper() not in room_map
            or conn["room_name_2"].upper() not in room_map
        ):
            logger.warning(conn)
            continue

        connections.append(
            Connection(
                room_map[conn["room_name_1"].upper()],
                room_map[conn["room_name_2"].upper()],
                np.array([conn["room_pos_1_x"], conn["room_pos_1_y"]]),
                np.array([conn["room_pos_2_x"], conn["room_pos_2_y"]]),
                conn["room_1_direction"],
                conn["room_2_direction"],
            )
        )

    return room_map, connections


class OptConn:
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
            OptConn.opt_one(conn, rooms)

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


import networkx as nx


def spring_layout_optimize(room_map: dict[str, Room], connections: list[Connection]):
    G = nx.Graph()

    # 添加房间为节点
    for room_name, room in room_map.items():
        G.add_node(room_name)

    # 添加连接为边
    for conn in connections:
        G.add_edge(conn.room1.name, conn.room2.name)

    # 使用spring_layout优化节点坐标
    pos = nx.spring_layout(G, scale=500, seed=42)  # scale可以调节整体图尺寸

    # 更新房间位置
    for room_name, room in room_map.items():
        # 设置房间左上角 box_position = pos - size / 2
        center_pos = np.array(pos[room_name])
        room.box_position = center_pos - room.box_size / 2

@logger.catch
def plot_map(world_path, output, name="ward"):
    room_map, connections = load_maptxt(world_path=world_path, name=name)
    fig, ax = plt.subplots(figsize=(16, 16))
    ax: plt.Axes

    OptConn.opt_a_lot(room_map, connections)

    # for room in room_map.values():
    #     if room.name.upper() in c.ROOM_RECOMMAND_POSITION:
    #         room.box_position = c.ROOM_RECOMMAND_POSITION[room.name.upper()]
    #     else:
    #         logger.warning(room.name)

    for room in room_map.values():
        room.plot(ax)

    for conn in connections:
        conn.plot(ax)

    x0, x1, y0, y1 = np.inf, -np.inf, np.inf, -np.inf
    for room in room_map.values():
        ext = room.box_extent
        x0 = min(x0, ext[0])
        x1 = max(x1, ext[1])
        y0 = min(y0, ext[2])
        y1 = max(y1, ext[3])

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    ax.set_aspect(1)
    ax.axis("off")
    fig.suptitle(f"{c.zone_id_2_cn(name)}", fontsize=30)
    plt.tight_layout()
    plt.savefig(output, dpi=600)
    plt.close()
    # plt.show()


if __name__ == "__main__":
    for i in c.WORLD_PATH.iterdir():
        if i.name == "gates" or i.name.endswith("-rooms"):
            continue
        name = i.name
        plot_map(
            c.WORLD_PATH,
            output=c.OUTPUT_PATH / f"{c.zone_id_2_cn(name)}.png",
            name=name,
        )
