from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import constants as c
from typing import TypedDict
from matplotlib.patches import Rectangle
from loguru import logger
from tqdm import tqdm
import math
import utils
import optroom

plt.rcParams["font.sans-serif"] = ["MicroSoft YaHei"]


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
        ax.imshow(self.water_mask * utils.rgba_pixel("#007AAE"))
        ax.imshow(self.wall_im * utils.rgba_pixel("#000000"))
        ax.imshow(self.background_im * utils.rgba_pixel("#000000", 0.25))
        ax.imshow(self.pipe_im * utils.rgba_pixel("#ffff00", 0.25))
        ax.imshow(self.entrance_im * utils.rgba_pixel("#ffff00", 0.5))
        ax.imshow(self.bar_im * utils.rgba_pixel("#880015", 1))
        ax.imshow(self.map_matrix[:, :, 5:6] * utils.rgba_pixel("#A349A4", 1))
        ax.imshow(self.map_matrix[:, :, 8:9] * utils.rgba_pixel("#3F48CC", 1))
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
        ax.imshow(self.rootxt.water_mask * utils.rgba_pixel("#007AAE"), extent=extent)
        ax.imshow(self.rootxt.wall_im * utils.rgba_pixel("#000000"), extent=extent)
        ax.imshow(
            self.rootxt.background_im * utils.rgba_pixel("#000000", 0.25), extent=extent
        )
        ax.imshow(
            self.rootxt.pipe_im * utils.rgba_pixel("#ffff00", 0.25), extent=extent
        )
        ax.imshow(
            self.rootxt.entrance_im * utils.rgba_pixel("#ffff00", 0.5), extent=extent
        )
        ax.imshow(self.rootxt.bar_im * utils.rgba_pixel("#880015", 1), extent=extent)
        ax.imshow(
            self.rootxt.map_matrix[:, :, 5:6] * utils.rgba_pixel("#A349A4", 1),
            extent=extent,
        )
        ax.imshow(
            self.rootxt.map_matrix[:, :, 8:9] * utils.rgba_pixel("#3F48CC", 1),
            extent=extent,
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
                    elif target_map != "NULL":
                        comments = f"({target_map}|{target_room})"
                    elif self.name.upper() == "WARA_P09":
                        comments = f"(上古城市|WAUA_E01|需要满级业力)"
                    elif self.name.upper() == "WAUA_TOYS":
                        comments = f"(古人线结局)"
                    elif self.name.upper()[:4] in {"WSUR", "WHIR", "WDSR", "WGWR"}:
                        comments = f"(外缘|随机房间)"
                    else:
                        comments = f"(恶魔|WRSA_L01|需要满级业力)"
                elif name == "PrinceBulb":
                    fontsize *= 2
                    comments = "王子"
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


def plot_map(world_path, output, name="ward", opt: optroom.BaseOpt | None = None):
    try:
        room_map, connections = load_maptxt(world_path=world_path, name=name)
    except Exception as e:
        return
    rooms = list(room_map.values())
    fig, ax = plt.subplots(facecolor="white")
    ax: plt.Axes

    if opt is not None:
        optroom.BaseOpt.use_opt(opt, rooms, connections)

    for room in rooms:
        room.plot(ax)

    for conn in connections:
        conn.plot(ax)

    big_box = optroom.Box.combined_big_box(map(optroom.Box.from_room, rooms))

    x0, y0 = big_box.left_down
    x1, y1 = big_box.right_top

    delta_x = big_box.size[0]
    delta_y = big_box.size[1]
    ax.set_xlim(x0 - 0.1 * delta_x, x1 + 0.1 * delta_x)
    ax.set_ylim(y0 - 0.1 * delta_y, y1 + 0.1 * delta_y)

    ax.set_aspect(1)
    ax.axis("off")
    fig.suptitle(f"{c.zone_id_2_cn(name)} ({name.upper()})", fontsize=30)

    # size_ratio = delta_x / delta_y
    # fig.set_size_inches(16 * math.sqrt(size_ratio), 16 / math.sqrt(size_ratio))
    fig.set_size_inches(delta_x / 50, delta_y / 50, forward=True)
    plt.tight_layout()
    plt.savefig(
        output, dpi=400, transparent=False, bbox_inches="tight", facecolor="white"
    )
    plt.close()


def plot_all_map(
    world_path: Path = c.WORLD_PATH,
    output: Path = c.OUTPUT_PATH,
    opt: optroom.BaseOpt | None = None,
):
    bar = tqdm(list(world_path.iterdir()))
    for i in bar:
        if i.name == "gates" or i.name.endswith("-rooms"):
            continue
        # if i.name != "whir":
        #     continue
        name = i.name
        title = f"{c.zone_id_2_cn(name)} ({name.upper()})"
        bar.desc = title
        plot_map(
            c.WORLD_PATH,
            output=output / f"{title}.png",
            name=name,
            opt=opt,
        )


if __name__ == "__main__":
    plot_all_map(opt=optroom.InitOpt())
