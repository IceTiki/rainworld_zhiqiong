from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import constants as c
from typing import TypedDict
from matplotlib.patches import Rectangle
from dataclasses import dataclass
from loguru import logger
import random


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


class Room:
    def __init__(
        self,
        roomtxt: RoomTxt,
        box_position: np.ndarray = np.array([0, 0]),
        box_size: np.ndarray = None,
    ):
        self.rootxt = roomtxt
        self.box_position = box_position
        self.box_size = (
            np.array([self.width, self.height]) if box_size is None else box_size
        )

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

        ax.text(*self.box_position, self.cn_name, c="yellow", alpha=1, fontsize=12)


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

    @staticmethod
    def cal_cloest_edge_point(conn_posi: np.array, room: Room):
        rel_align_points = np.array(
            [
                [conn_posi[0], 0],
                [conn_posi[0], room.box_size[1]],
                [0, conn_posi[1]],
                [room.box_size[0], conn_posi[1]],
            ]
        )
        conn_to_edge = rel_align_points - conn_posi

        best = np.argmin(np.linalg.norm(conn_to_edge, axis=1))
        return rel_align_points[best]

    @property
    def room1_cloest_edge_point(self):
        return self.cal_cloest_edge_point(self.room1_posi, self.room1)

    @property
    def room2_cloest_edge_point(self):
        return self.cal_cloest_edge_point(self.room2_posi, self.room2)

    def plot(self, ax: plt.Axes):
        start = self.room1.box_position + self.room1_posi
        end = self.room2.box_position + self.room2_posi
        ax.plot([start[0], end[0]], [start[1], end[1]], c="blue")


def load_maptxt(world_path: Path = c.WORLD_PATH, name="wara"):
    name = name.upper()
    map_txt = world_path / name.lower() / f"map_{name.lower()}.txt"
    room_folder = world_path / f"{name.lower()}-rooms"

    room_map: dict[str, Room] = {}
    map_txt = MapTxt.from_file(map_txt)
    for room in map_txt.rooms:
        room_txt_path = room_folder / (room["room_name"].lower() + ".txt")
        if not room_txt_path.is_file():
            continue
        room_map[room["room_name"].upper()] = Room(
            RoomTxt.from_file(room_txt_path),
            np.array([room["canon_pos_x"], room["canon_pos_y"]]),
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


def opt_conn(conn: Connection):
    conn_posi_1 = conn.room1.box_position + conn.room1_posi
    conn_posi_2 = conn.room2.box_position + conn.room2_posi

    weight1, weight2 = map(np.prod, (conn.room1.box_size, conn.room2.box_size))
    center = (conn_posi_1 * weight1 + conn_posi_2 * weight2) / (weight1 + weight2)

    norm = np.linalg.norm(
        sum(
            map(
                lambda x: (x) / np.linalg.norm(x),
                (
                    conn.room1_cloest_edge_point - conn.room1_posi,
                    conn.room2_cloest_edge_point - conn.room2_posi,
                ),
            )
        )
    )

    for room, cloest_edge_point in (
        (conn.room1, conn.room1_cloest_edge_point),
        (conn.room2, conn.room2_cloest_edge_point),
    ):
        if norm == 0:
            room.box_position = center - cloest_edge_point


if __name__ == "__main__":
    room_map, connections = load_maptxt()
    fig, ax = plt.subplots()
    ax: plt.Axes

    for _ in range(10000):
        conn = random.choice(connections)
        opt_conn(conn)

    x0, x1, y0, y1 = 0, 0, 0, 0
    for room in room_map.values():
        room.plot(ax)

    for conn in connections:
        conn.plot(ax)

    x0, x1, y0, y1 = 0, 0, 0, 0
    for room in room_map.values():
        ext = room.box_extent
        x0 = min(x0, ext[0])
        x1 = max(x1, ext[1])
        y0 = min(y0, ext[2])
        y1 = max(y1, ext[3])

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    ax.set_aspect(1)
    plt.show()
