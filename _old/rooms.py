from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import constants as c
from typing import TypedDict, Optional, Generator
from matplotlib.patches import Rectangle
import matplotlib as mpl
from loguru import logger
from tqdm import tqdm
import math
import utils
import optim
from pprint import pprint
from matplotlib.patches import FancyArrowPatch

import assets

mpl.use("Agg")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
TEST_CATCHER = []
# mpl.rcParams['font.family'] = 'Microsoft YaHei'  # 或 SimHei
# mpl.use("pgf")  # 使用 pgf 后端才能走 xelatex
# mpl.rcParams.update(
#     {
#         "text.usetex": True,
#         "pgf.texsystem": "xelatex",
#         "pgf.rcfonts": False,
#         "pgf.preamble": "\n".join(
#             [
#                 r"\usepackage{xeCJK}",
#                 r"\setCJKmainfont{Microsoft YaHei}",
#                 r"\usepackage{xcolor}",
#             ]
#         ),
#     }
# )


class RoomTxt:
    CHANNEL = 8 + 1

    @classmethod
    def from_file(cls, path: str | Path):
        return cls(Path(path).read_text())

    def __init__(self, text: str):
        self.text = text
        self.lines = text.splitlines()
        self.name = self.lines[0].upper()

        line1 = self.lines[1].split("|")
        if len(line1) != 2:
            water_level = -1
            size = line1[0]
        else:
            size, water_level = self.lines[1].split("|")
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
        #     ax.imshow(self.map_matrix[:, :, i], cmap="gray",clip_on=False)
        #     ax.set_title(str(i))
        #     ax.axis("off")

        # ax = ax_list[-1]
        fig, ax = plt.subplots()
        ax: plt.Axes
        ax.imshow(self.water_mask * utils.rgba_pixel("#007AAE"), clip_on=False)
        ax.imshow(self.wall_im * utils.rgba_pixel("#000000"), clip_on=False)
        ax.imshow(self.background_im * utils.rgba_pixel("#000000", 0.25), clip_on=False)
        ax.imshow(self.pipe_im * utils.rgba_pixel("#ffff00", 0.25), clip_on=False)
        ax.imshow(self.entrance_im * utils.rgba_pixel("#ffff00", 0.5), clip_on=False)
        ax.imshow(self.bar_im * utils.rgba_pixel("#880015", 1), clip_on=False)
        ax.imshow(
            self.map_matrix[:, :, 5:6] * utils.rgba_pixel("#A349A4", 1), clip_on=False
        )
        ax.imshow(
            self.map_matrix[:, :, 8:9] * utils.rgba_pixel("#3F48CC", 1), clip_on=False
        )
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

        self.placed_objects: list[str, float, float, list[str]] = []
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
        map_txt_info: Optional["MapTxt._Room"] = None,
    ):
        self.rootxt = roomtxt
        self.box_position = box_position
        self.box_size = (
            np.array([self.width, self.height]) if box_size is None else box_size
        )
        self.room_setting: RoomSettingTxt | None = room_setting
        self.map_txt_info: "MapTxt._Room" | None = (
            map_txt_info if map_txt_info is not None else {}
        )

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

    @property
    def subregion_name(self) -> str:
        if self.map_txt_info is not None:
            return self.map_txt_info.get("subregion_name", "")
        return ""

    def plot(self, ax: plt.Axes):
        extent = self.box_extent
        ax.imshow(
            self.rootxt.water_mask * utils.rgba_pixel("#007AAE"),
            extent=extent,
            clip_on=False,
        )
        ax.imshow(
            self.rootxt.wall_im * utils.rgba_pixel("#000000"),
            extent=extent,
            clip_on=False,
        )
        ax.imshow(
            self.rootxt.background_im * utils.rgba_pixel("#000000", 0.25),
            extent=extent,
            clip_on=False,
        )
        ax.imshow(
            self.rootxt.pipe_im * utils.rgba_pixel("#ffff00", 0.25),
            extent=extent,
            clip_on=False,
        )
        ax.imshow(
            self.rootxt.entrance_im * utils.rgba_pixel("#ffff00", 0.5),
            extent=extent,
            clip_on=False,
        )
        ax.imshow(
            self.rootxt.bar_im * utils.rgba_pixel("#880015", 1),
            extent=extent,
            clip_on=False,
        )
        ax.imshow(
            self.rootxt.map_matrix[:, :, 5:6] * utils.rgba_pixel("#A349A4", 1),
            extent=extent,
            clip_on=False,
        )
        ax.imshow(
            self.rootxt.map_matrix[:, :, 8:9] * utils.rgba_pixel("#3F48CC", 1),
            extent=extent,
            clip_on=False,
        )

        edgecolor = utils.color_hash(self.subregion_name)  # edgecolor="#138535",
        rect = Rectangle(
            self.box_position,
            self.width,
            self.height,
            linewidth=2,
            edgecolor=edgecolor,
            facecolor="none",
        )
        ax.add_patch(rect)

        text = self.cn_name
        fontsize = 3
        color = "white"
        bbox = dict(
            facecolor="#000000aa",
            edgecolor="#00000000",
            boxstyle="square,pad=0",
        )

        if self.name.upper() in c.SPECIAL_ROOMS:
            sp_type = c.SPECIAL_ROOMS[self.name.upper()]
            sp_name = c.SPECIAL_ROOM_TYPE_2_CN.get(sp_type, sp_type)
            color = "white"
            fontsize *= 2
            text += f"({sp_name})"
            bbox = dict(
                facecolor="#ff0000aa",
                edgecolor="#00000000",
                boxstyle="square,pad=0",
            )
        ax.text(
            *self.box_position, text, c=color, alpha=1, fontsize=fontsize, bbox=bbox
        )

        if self.room_setting is not None:
            for obj in self.room_setting.placed_objects:
                name, x, y = obj[:3]
                property: list[str] = obj[-1]

                comments = ""
                fontsize = 3
                color = "#ffffff"
                if name in {"SpinningTopSpot", "WarpPoint"}:
                    fontsize *= 2
                    name = c.PLACE_OBJECT_NAME[name]
                    map_idx = (
                        (len(property) - 1 - property[::-1].index("Watcher")) + 1
                        if "Watcher" in property
                        else 4
                    )
                    target_map = c.zone_id_2_cn(property[map_idx])
                    target_room = property[map_idx + 1].upper()

                    if self.name.upper() == "WAUA_BATH":
                        comments = f"(古人线结局, 一次性传送)\n(上古城市|WAUA_TOYS)"
                    elif self.name.upper() == "WARA_P09":
                        comments = f"(上古城市|WAUA_E01|需要满级业力)"
                    elif self.name.upper() == "WAUA_TOYS":
                        comments = f"(古人线结局)"
                    elif self.name.upper()[:4] in {
                        "WSUR",
                        "WHIR",
                        "WDSR",
                        "WGWR",
                        "WSSR",
                    }:
                        comments = f"(外缘)"
                    elif target_map != "NULL":
                        comments = f"({target_map}|{target_room})"
                    else:
                        comments = f"(恶魔|WRSA_L01|需要满级业力)"

                    TEST_CATCHER.append([x, y, self.name, target_room, comments, name])

                elif name == "PrinceBulb":
                    fontsize *= 2
                    name = "王子"
                elif name in c.PLACE_OBJECT_NAME:
                    name = c.PLACE_OBJECT_NAME[name]
                else:
                    continue

                x = self.box_position[0] + x / 20
                y = self.box_position[1] + y / 20

                ax.text(
                    x,
                    y,
                    f"{name}{comments}",
                    fontsize=fontsize,
                    c=color,
                    bbox=dict(
                        facecolor="#ff0000aa",
                        edgecolor="#00000000",
                        boxstyle="square,pad=0",
                    ),
                )


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

        room_names = {i["room_name"] for i in self.rooms}
        self.rooms = [i for i in self.rooms if (i["room_name"] + "W") not in room_names]


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
        conn_r1 = self.room1.box_position + self.room1_posi
        conn_r2 = self.room2.box_position + self.room2_posi

        dir1 = self.DIRECTION_MAP[self.room1_direct]
        dir2 = self.DIRECTION_MAP[self.room2_direct]

        dir_len = 15
        dir_len = min(dir_len, np.linalg.norm(conn_r2 - conn_r1) / 3)
        if (
            utils.calculate_cos_theta(conn_r2 - conn_r1, dir1) < -0.8
            or utils.calculate_cos_theta(conn_r1 - conn_r2, dir2) < -0.8
        ):
            dir_len = 0
        dir1, dir2 = map(
            lambda x: x * dir_len,
            (dir1, dir2),
        )

        linewidth = 0.6
        alpha = 1
        utils.cubic_bezier_curve(
            conn_r1,
            conn_r2,
            dir1,
            dir2,
            ax,
            dict(
                c="#000000",
                linestyle="-",
                linewidth=linewidth,
                alpha=alpha,
            ),
        )
        utils.cubic_bezier_curve(
            conn_r1,
            conn_r2,
            dir1,
            dir2,
            ax,
            dict(
                c="#FFF200",
                linestyle="--",
                linewidth=linewidth * 0.5,
                alpha=alpha,
            ),
        )


class Region:
    @staticmethod
    def load_maptxt(world_path: Path = c.WORLD_PATH, name="wara"):
        name = name.upper()
        map_txt = world_path / name.lower() / f"map_{name.lower()}.txt"
        map_txt_watcher = world_path / name.lower() / f"map_{name.lower()}-watcher.txt"
        if map_txt_watcher.is_file():
            map_txt = map_txt_watcher
        room_folder = world_path / f"{name.lower()}-rooms"

        room_map: dict[str, Room] = {}
        map_txt = MapTxt.from_file(map_txt)
        for room in map_txt.rooms:
            room_txt_path = room_folder / (room["room_name"].lower() + ".txt")
            room_setting_path = utils.world_file_locator(
                Path(f"{name.lower()}-rooms")
                / (room["room_name"].lower() + "_settings.txt")
            )
            if not room_txt_path.is_file():
                logger.warning(f"{room_txt_path} not exists")
                continue
            room_map[room["room_name"].upper()] = Room(
                RoomTxt.from_file(room_txt_path),
                np.array([room["canon_pos_x"], room["canon_pos_y"]]),
                room_setting=(
                    RoomSettingTxt.from_file(room_setting_path)
                    if room_setting_path is not None
                    else None
                ),
                map_txt_info=room,
            )

        connections: list[Connection] = []
        for conn in map_txt.connections:
            if (
                conn["room_name_1"].upper() not in room_map
                or conn["room_name_2"].upper() not in room_map
            ):
                logger.warning(
                    f"{conn["room_name_1"].upper()} and {conn["room_name_2"].upper()} not in room_map."
                )
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

    @classmethod
    # @logger.catch
    def from_maptxt(cls, world_path: Path, name="ward"):
        room_map, connections = cls.load_maptxt(world_path=world_path, name=name)
        return cls(room_map=room_map, connections=connections, name=name)

    def __init__(
        self, room_map: dict[str, Room], connections: list[Connection], name: str
    ):
        self.room_map: dict[str, Room] = room_map
        self.connections: list[Connection] = connections
        self.__name: str
        self.name: str = name

    def get_room(self, key: str) -> Room | None:
        return self.room_map.get(key.upper())

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, new_name: str):
        self.__name = new_name.upper()

    @property
    def rooms(self) -> list[Room]:
        return self.room_map.values()

    @property
    def region_box(self) -> optim.Box:
        return optim.Box.combined_big_box(map(optim.Box.from_room, self.rooms))

    @property
    def position(self) -> np.ndarray:
        return self.region_box.position

    @position.setter
    def position(self, new_position: np.ndarray):
        old_position = self.position
        delta = new_position - old_position
        for r in self.rooms:
            r.box_position += delta

    @property
    def sub_regions(self) -> set[str]:
        result = set()
        for r in self.rooms:
            result.add(r.subregion_name)
        result.remove("")
        return result

    @property
    def region_name_en(self) -> str:
        return c.ZONE_ID_2_EN.get(self.name, self.name)

    @property
    def region_name_cn(self) -> str:
        return c.EN_2_CN.get(self.region_name_en, self.region_name_en)

    def optimize(self, opt: optim.BaseOpt | None = None):
        if opt is not None:
            opt.optimizing_rooms(self.rooms, self.connections)

    def plot(self, ax: plt.Axes):
        for room in self.rooms:
            room.plot(ax)

        for conn in self.connections:
            conn.plot(ax)

        self.plot_box(ax)

    def plot_box(self, ax: plt.Axes):
        font_size = 15
        title = [
            (
                f"{self.region_name_cn} ({self.name})",
                "#000000",
                font_size,
            ),
            (
                f"{self.region_name_en}",
                "#000000",
                font_size,
            ),
        ]
        if len(self.sub_regions) > 1:
            for i in self.sub_regions:
                title.append(
                    (f"{c.en_2_cn(i)} ({i})", utils.color_hash(i), font_size * 0.66)
                )

        box = self.region_box
        utils.draw_multiline_text_centered(
            ax=ax,
            lines=title,
            posi=box.position + box.size * np.array([0, 1]),
            alpha=0.5,
        )

        rect = Rectangle(
            box.position,
            box.size[0],
            box.size[1],
            linewidth=4,
            edgecolor="black",
            facecolor="none",
        )
        ax.add_patch(rect)


class RegionTeleportConnection:
    @classmethod
    def from_name(
        cls,
        name1: str,
        name2: str,
        regions: list[Region],
        r1_posi: np.ndarray = None,
        r2_posi: np.ndarray = None,
    ):
        region_map = {r.name: r for r in regions}

        regs = []
        reg_positions = []
        rooms: list[Room] = []

        for name, posi in zip((name1, name2), (r1_posi, r2_posi)):
            name = name.upper()
            reg_name = name.split("_")[0]
            region = region_map[reg_name]
            room = region.get_room(name)
            if room is None:
                logger.warning(name)
                continue

            if posi is None:
                room_posi = room.box_position + room.box_size / 2
            else:
                room_posi = room.box_position + posi
            regs.append(region)
            reg_positions.append(room_posi - region.region_box.position)
            rooms.append(room)

        return cls(*regs, *reg_positions)

    def __init__(
        self,
        region1: Region,
        region2: Region,
        reg1_posi: np.ndarray,
        reg2_posi: np.ndarray,
    ):
        self.region1: Region = region1
        self.region2: Region = region2
        self.reg1_posi: np.ndarray = reg1_posi
        self.reg2_posi: np.ndarray = reg2_posi

    def plot(self, ax: plt.Axes):
        conn_r1 = self.region1.region_box.position + self.reg1_posi
        conn_r2 = self.region2.region_box.position + self.reg2_posi
        linewidth = 4
        alpha = 0.2
        # color = "#FF0000"

        # ax.annotate(
        #     f"",
        #     xy=(conn_r2[0], conn_r2[1]),
        #     xytext=(conn_r1[0], conn_r1[1]),
        #     fontsize=6,
        #     arrowprops=dict(arrowstyle="->", color=color, lw=linewidth, alpha=alpha),
        # )

        # # 创建一个大的箭头
        # long_arrow = FancyArrowPatch(conn_r1, conn_r2, mutation_scale=20, color=color, arrowstyle='->', lw=linewidth, alpha=alpha)
        # ax.add_patch(long_arrow)

        # 在箭头的中间添加多个小箭头
        conn_vec = conn_r2 - conn_r1
        conn_vec_norm = np.linalg.norm(conn_vec)
        conn_vec_unit = conn_vec / conn_vec_norm  # ! DIV 0
        SPACE = 50
        num_arrows = int(conn_vec_norm / SPACE)
        for i in range(1, num_arrows + 1):
            t = i / (num_arrows + 1)  # 计算每个小箭头的比例位置
            p0 = conn_r1 + t * conn_vec
            p1 = p0 + conn_vec_unit * SPACE / 2
            color = plt.cm.cool(t)

            small_arrow = FancyArrowPatch(
                p0,
                p1,
                mutation_scale=8,
                color=color,
                arrowstyle="->",
                lw=linewidth,
                alpha=alpha,
            )
            ax.add_patch(small_arrow)


def plot_region(world_path, output, name="ward", opt: optim.BaseOpt | None = None):
    region = Region.from_maptxt(world_path, name)
    if region is None:
        return
    fig, ax = plt.subplots(facecolor="white")
    ax: plt.Axes

    region.optimize(opt)
    region.position = np.array([0, 0])
    region.plot(ax)

    big_box = region.region_box

    x0, y0 = big_box.left_down
    x1, y1 = big_box.right_top

    delta_x = big_box.size[0]
    delta_y = big_box.size[1]
    ax.set_xlim(x0 - 0.1 * delta_x, x1 + 0.1 * delta_x)
    ax.set_ylim(y0 - 0.1 * delta_y, y1 + 0.1 * delta_y)

    ax.set_aspect(1)
    ax.axis("off")

    # size_ratio = delta_x / delta_y
    # fig.set_size_inches(16 * math.sqrt(size_ratio), 16 / math.sqrt(size_ratio))
    fig.set_size_inches(max(delta_x / 50, 5), max(delta_y / 50, 5), forward=True)
    # plt.tight_layout()
    plt.savefig(
        output, dpi=400, transparent=False, bbox_inches="tight", facecolor="white"
    )
    plt.close()


def yield_regions(world_path: Path) -> Generator[str, None, None]:
    bar = tqdm(
        list(
            filter(
                lambda x: x.name != "gates" and not x.name.endswith("-rooms"),
                world_path.iterdir(),
            )
        )
    )
    for i in bar:
        # if i.name != "warf":
        #     continue
        name = i.name
        title = f"{c.zone_id_2_cn(name)} ({name.upper()})"
        bar.desc = title
        yield name


def plot_all_map(
    world_path: Path = c.WORLD_PATH,
    output: Path = c.OUTPUT_PATH,
    opt: optim.BaseOpt | None = None,
):
    for name in yield_regions(world_path):
        title = f"{c.zone_id_2_cn(name)} ({name.upper()})"
        plot_region(
            c.WORLD_PATH,
            output=output / f"{title}.png",
            name=name,
            opt=opt,
        )


def plot_big_map(
    world_path: Path = c.WORLD_PATH,
    output: Path = c.OUTPUT_PATH / "union_map.pdf",
    opt: optim.BaseOpt | None = None,
):
    # mpl.use("Agg")
    regions: list[Region] = []
    for name in yield_regions(world_path):
        region = Region.from_maptxt(world_path, name)
        regions.append(region)
        region.optimize(opt)
        # region.position = np.array([0, 0])

    rt_conns: list[RegionTeleportConnection] = []
    for x, y, r1, r2, _ in c.__TELEPORTS:
        rt_conns.append(
            RegionTeleportConnection.from_name(
                r1, r2, regions, r1_posi=np.array([x, y]) / 20
            )
        )

    optim.AlignOpt("bfs").optimizing_regions(regions, rt_conns)

    # ===============================
    fig, ax = plt.subplots(facecolor="white")
    ax: plt.Axes

    big_box = optim.Box.combined_big_box([i.region_box for i in regions])

    x0, y0 = big_box.left_down
    x1, y1 = big_box.right_top

    delta_x = big_box.size[0]
    delta_y = big_box.size[1]
    ax.set_xlim(x0 - 0.2 * delta_x, x1 + 0.2 * delta_x)
    ax.set_ylim(y0 - 0.2 * delta_y, y1 + 0.2 * delta_y)

    ax.set_aspect(1)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_position([0, 0, 1, 1])

    # size_ratio = delta_x / delta_y
    # fig.set_size_inches(16 * math.sqrt(size_ratio), 16 / math.sqrt(size_ratio))
    FACTOR = 50
    fig.set_size_inches(delta_x / FACTOR, delta_y / FACTOR, forward=True)
    MAX_EDGE_PIXEL = 100000
    max_dpi = MAX_EDGE_PIXEL // max(delta_x / FACTOR, delta_y / FACTOR)
    # plt.tight_layout()

    for reg in regions:
        reg.plot(ax)

    for conn in rt_conns:
        if conn.region1.name == "WRSA" or conn.region2.name == "WRSA":
            continue
        conn.plot(ax)

    logger.info("Saving figure...")
    fig.savefig(output, dpi=max_dpi, transparent=False, facecolor="white")
    plt.close()


def test_load(
    world_path: Path = c.WORLD_PATH,
):
    fig, ax = plt.subplots(facecolor="white")
    # mpl.use("Agg")
    regions: list[Region] = []
    for name in yield_regions(world_path):
        region = Region.from_maptxt(world_path, name)
        regions.append(region)
        region.plot(ax)


if __name__ == "__main__":
    # test_load()
    # pprint(TEST_CATCHER)

    plot_all_map(opt=optim.AlignOpt())
    plot_big_map(opt=optim.AlignOpt())
    logger.info("Done!")
