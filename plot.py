from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

from assets import RegionInfo, RoomInfo, RoomTxt
from colls import Box, EndPoint, Edge
from utils import CachedProperty
import utils
import constants as cons
import optim

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]


class Room(Box):
    COLORS = {
        "water": utils.rgba_pixel("#007AAE"),
        "wall": utils.rgba_pixel("#000000"),
        "background": utils.rgba_pixel("#000000", 0.25),
        "pipe": utils.rgba_pixel("#ffff00", 0.25),
        "entrance": utils.rgba_pixel("#ffff00", 0.5),
        "bar": utils.rgba_pixel("#880015", 1),
        "map5": utils.rgba_pixel("#A349A4", 1),
        "map8": utils.rgba_pixel("#3F48CC", 1),
        "sand": utils.rgba_pixel("#BF8F16", 1),
    }

    @dataclass
    class SpecialParams:
        special_room_type: str = ""
        subregion_name: str = ""

    def __init__(
        self,
        position,
        room_info: RoomInfo,
        *,
        size: np.ndarray = None,
        edges: list[Edge] = None,
        special_params: SpecialParams = SpecialParams(),
    ):
        size = (
            size if size is not None else np.array([room_info.width, room_info.height])
        )
        super().__init__(position, size, edges)
        self.info: RoomInfo = room_info
        self.special_params: Room.SpecialParams = special_params

    @property
    def name(self):
        return self.info.name

    def _render_terrain_handle(self) -> np.ndarray:
        width = self.width
        height = self.height
        im = np.zeros((height, width, 1), dtype=np.bool_)
        SCALE = 20
        SAMPLE = 20

        roomsettingtxt = self.info.roomsettingtxt

        if roomsettingtxt is None:
            return im
        terrain = [i for i in roomsettingtxt.placed_objects if i[0] == "TerrainHandle"]
        if len(terrain) < 2:
            return im
        terrain.sort(key=lambda x: x[1])

        terrain = list(map(list, terrain))
        for i in terrain:
            i[-1] = list(map(float, i[-1]))

        curve = np.empty((0, 2))
        for t1, t2 in zip(terrain[:-1], terrain[1:]):
            n1, x1, y1, (xl1, yl1, xr1, yr1, ukn1) = t1
            n2, x2, y2, (xl2, yl2, xr2, yr2, ukn2) = t2

            p1 = np.array((x1, y1))
            p2 = np.array((x2, y2))
            pr1 = np.array((xr1, yr1)) / SCALE
            pl2 = np.array((xl2, yl2)) / SCALE

            num = int((x2 - x1) * SAMPLE)

            curve = np.concatenate(
                [curve, utils.bezier_curve(p1, p1 + pr1, p2 + pl2, p2, num=num)], axis=0
            )

        x = curve[:, 0].astype(int)
        y = curve[:, 1].astype(int)

        mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x = x[mask]
        y = y[mask]

        im[height - 1 - y, x] = 1

        # if (
        #     0 <= terrain[0][1] < width
        #     and 0 <= (height - 1 - int(terrain[0][2])) < height - 1
        # ):
        #     im[height - 1 - int(terrain[0][2]), : int(terrain[0][1])] = 1
        # if (
        #     0 <= terrain[-1][1] < width
        #     and 0 <= (height - 1 - int(terrain[-1][2])) < height - 1
        # ):
        #     im[height - 1 - int(terrain[-1][2]), int(terrain[-1][1]) :] = 1

        im = np.maximum.accumulate(im, axis=0)
        # # plt.scatter(curve[:, 0], curve[:, 1])
        # # plt.scatter([i[1] for i in terrain], [-i[2] for i in terrain])
        # # plt.scatter(
        # #     [i[1] + i[-1][0] /SCALE for i in terrain], [i[2] + i[-1][1] /SCALE for i in terrain]
        # # )
        # # plt.scatter(
        # #     [i[1] + i[-1][2] /SCALE for i in terrain], [i[2] + i[-1][3] /SCALE for i in terrain]
        # # )
        # plt.scatter(x, height-y)
        # plt.imshow(im)
        # plt.show()
        return im

    def _render_map(self, color=RoomTxt.MapImColor()):
        room_info = self.info
        water_info = room_info.water_info
        height = room_info.height
        wmin, wmax = map(
            lambda x: int(min(max(0, x), height)),
            (water_info.water_flux_min_level, water_info.water_flux_max_level),
        )

        shape = (room_info.height, room_info.width, 4)
        if water_info.lethal_water:
            water = np.full(shape, color.water_acid, dtype=np.uint8)
        else:
            water = np.full(shape, color.water, dtype=np.uint8)

        # alpha
        water[: height - wmax, :, :] = 0
        water[height - wmax : height - wmin, :, :] //= 2

        canvas: np.ndarray = water

        map_im = room_info.roomtxt.map_im

        for im, c in zip(
            [
                map_im.background,
                self._render_terrain_handle(),
                map_im.wall,
                map_im.pipe,
                map_im.entrance,
                map_im.bar,
                map_im.im5,
                map_im.im8,
            ],
            [
                color.background,
                color.sand,
                color.wall,
                color.pipe,
                color.entrance,
                color.bar,
                color.im5,
                color.im8,
            ],
        ):

            canvas = utils.alpha_blend(im * np.full(shape, c, dtype=np.uint8), canvas)
        return canvas
        #     canvas_rgb = canvas[..., :3]
        #     canvas_alpha = canvas[..., 3:]

        #     bg_rgb = color[..., :3]
        #     bg_alpha = im * color[0][0][3] / 255

        #     out_alpha = canvas_alpha + bg_alpha * (1 - canvas_alpha)
        #     out_rgb = (
        #         canvas_rgb * canvas_alpha + bg_rgb * bg_alpha * (1 - canvas_alpha)
        #     ) / np.clip(out_alpha, 1e-6, 1.0)

        #     canvas = np.concatenate([out_rgb, out_alpha], axis=-1)

        # return (canvas * 255).astype(np.uint8)

    @dataclass
    class _WarpPointInfo:
        obj_type: str = "NULL"
        obj_type_cn: str = "NULL"
        from_room: str = "NULL"
        to_room: str = "NULL"
        from_coord: list[float] = field(default_factory=lambda: [0, 0])
        to_coord: list[float] = field(default_factory=lambda: [0, 0])
        comments: str = ""
        raw: list[str] = field(default_factory=list)

        @property
        def from_region(self):
            return self.from_room.split("_", maxsplit=1)[0]

        @property
        def to_region(self):
            return self.to_room.split("_", maxsplit=1)[0]

    @staticmethod
    def _is_float(num: str) -> bool:
        try:
            float(num)
            return True
        except ValueError:
            return False

    def _get_warppoint_info(
        self, obj: tuple[str, float, float, list[str]]
    ) -> _WarpPointInfo:
        wpi = self._WarpPointInfo()
        wpi.obj_type, x, y, property_ = obj
        wpi.from_coord = [x, y]
        assert wpi.obj_type in {"SpinningTopSpot", "WarpPoint"}
        wpi.obj_type_cn = cons.PLACE_OBJECT_NAME[wpi.obj_type]
        wpi.from_room = self.name.upper()
        wpi.raw = list(obj)

        # ====

        flag = "start"
        for i in property_:
            if i == "NULL" and flag in {
                "find_to_room",
                "find_to_coord_x",
                "find_to_coord_y",
            }:
                break

            if flag == "start" and i == "Watcher":
                flag = "find_to_room"
                continue
            elif flag == "find_to_room" and i != "Watcher":
                if "_" in i:
                    flag = "find_to_coord_x"
                    wpi.to_room = i.upper()
                continue
            elif flag == "find_to_coord_x" and self._is_float(i):
                wpi.to_coord = [float(i) / 20]
                flag = "find_to_coord_y"
                continue
            elif flag == "find_to_coord_y":
                wpi.to_coord.append(float(i) / 20)
                break

        # ====

        if wpi.from_room == "WAUA_BATH":
            wpi.comments = "古人线结局, 一次性传送"
            wpi.to_room = "WAUA_TOYS"
            # wpi.to_coord = [22.5, 61.5]
        elif wpi.from_room == "WARA_P09":
            wpi.comments = f"位于涟漪空间"
            wpi.to_room = "WAUA_E01"
            wpi.to_coord = [38, 16]  # 手动设置坐标，不精确
        elif wpi.from_room == "WAUA_TOYS":
            wpi.comments = f"古人线结局"
            wpi.to_room = "WAUA_TOYS"
            wpi.to_coord = [22.5, 61.5]
        # elif wpi.from_room == "WORA_STARCATCHER07":
        #     wpi.to_coord = "WORA_STARCATCHER02"
        #     wpi.to
        elif wpi.from_room[:4] in {
            "WSUR",
            "WHIR",
            "WDSR",
            "WGWR",
            "WSSR",
        }:
            wpi.comments = f"到达房间随机"
            wpi.to_room = "WORA_START"
            wpi.to_coord = [24, 96]  #  手动设置坐标，不精确
        elif wpi.to_room != "NULL":
            pass
        else:
            wpi.to_room = "WRSA_L01"
            wpi.comments = f"位于涟漪空间"
            wpi.to_coord = [76, 184.82]

        return wpi

    def _plot_object(self, ax: plt.Axes):
        room_setting = self.info.roomsettingtxt
        if room_setting is None:
            return
        for obj in room_setting.placed_objects:
            name, x, y = obj[:3]
            property_: list[str] = obj[-1]

            comments = ""
            fontsize = 3
            color = "#ffffff"
            if name in {"SpinningTopSpot", "WarpPoint"}:
                fontsize *= 2
                wp_info = self._get_warppoint_info(obj)
                name = wp_info.obj_type_cn

                to_reg_name_cn = (
                    cons.translate(
                        cons.REGION_DISPLAYNAME.get(
                            wp_info.to_region, wp_info.to_region
                        )
                    )
                )
                comments = f"({to_reg_name_cn}|{wp_info.to_room})" + (
                    f"\n({wp_info.comments})" if wp_info.comments else ""
                )


            elif name == "PrinceBulb":
                fontsize *= 2
                name = "王子"
            elif name in cons.PLACE_OBJECT_NAME:
                name = cons.PLACE_OBJECT_NAME[name]
            elif name == "CorruptionTube":
                x = self.position[0] + x
                y = self.position[1] + y
                x2 = x + float(property_[0]) / 20
                y2 = y + float(property_[1]) / 20
                ax.plot([x, x2], [y, y2], color="purple", linestyle=":", linewidth=0.5)
                continue
            else:
                continue

            x = self.position[0] + x
            y = self.position[1] + y

            ax.text(
                x,
                y,
                f"{name}{comments}",
                fontsize=fontsize,
                c=color,
                bbox=dict(
                    facecolor="#FF7F27aa",
                    edgecolor="#00000000",
                    boxstyle="square,pad=0",
                ),
            )

    def plot(self, ax: plt.Axes):
        extent = np.array(
            [
                self.position[0],
                self.position[0] + self.size[0],
                self.position[1],
                self.position[1] + self.size[1],
            ]
        )
        ax.imshow(
            self._render_map(),
            extent=extent,
            clip_on=False,
        )

        edgecolor = utils.color_hash(self.special_params.subregion_name)
        rect = Rectangle(
            self.position,
            self.width,
            self.height,
            linewidth=2,
            edgecolor=edgecolor,
            facecolor="none",
            linestyle="-" if self.info.roomtxt.border_type == "Solid" else "-.",
        )
        ax.add_patch(rect)

        text = self.info.name
        fontsize = 6
        color = "white"
        bbox = dict(
            facecolor="#000000ff",
            edgecolor="#00000000",
            boxstyle="square,pad=0",
        )

        if self.special_params.special_room_type.upper() in cons.SPECIAL_ROOM_TYPE_2_CN:
            sp_type = self.special_params.special_room_type.upper()
            sp_name = cons.SPECIAL_ROOM_TYPE_2_CN.get(sp_type, sp_type)
            color = "white"
            fontsize *= 1
            text += f"({sp_name})"
            bbox = dict(
                facecolor="#ff0000ff",
                edgecolor="#00000000",
                boxstyle="square,pad=0",
            )
        ax.text(
            *self.position,
            text,
            c=color,
            alpha=1,
            fontsize=fontsize,
            bbox=bbox,
            va="top",
        )

        self._plot_object(ax)


class Connection(Edge):
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
        end_point_1 = EndPoint(room1, room1_posi)
        end_point_2 = EndPoint(room2, room2_posi)
        super().__init__(end_point_1, end_point_2)

        self.room1_direct = room1_direct
        self.room2_direct = room2_direct

    @property
    def room1(self) -> Room:
        return self.end_point_1.box

    @property
    def room2(self) -> Room:
        return self.end_point_2.box

    @property
    def room1_posi(self):
        return self.end_point_1.box.position

    @property
    def room2_posi(self):
        return self.end_point_2.box.position

    @property
    def norm(self):
        ep1, ep2 = self.end_points
        start = ep1.glo_posi
        end = ep2.glo_posi
        return np.linalg.norm(start - end)

    def plot(self, ax: plt.Axes):
        ep1, ep2 = self.end_points
        conn_r1 = ep1.glo_posi
        conn_r2 = ep2.glo_posi

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

        linewidth = 1
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


class Region(Box, CachedProperty.Father):
    @staticmethod
    def combined_big_box(boxes) -> Box:
        return Box.combined_big_box(boxes)

    def __init__(self, region_info: RegionInfo):
        self.info: RegionInfo = region_info
        self.edges: list[Edge] = []
        CachedProperty.Father.__init__(self)

    def _build_region(self):
        region = self.info
        rooms: list[Room] = []

        room_positions = {
            i.name.upper(): np.array([i.canon_pos_x, i.canon_pos_y])
            for i in region.map_txt.rooms
        }
        special_room_types = {}

        for i in region.world_txt.data.get("ROOMS", []):
            room_name = i[0]
            room_type = i[-1]
            if room_type in cons.SPECIAL_ROOM_TYPE_2_CN:
                special_room_types[room_name.upper()] = room_type.upper()

        subregion_name = {}
        for i in region.map_txt.rooms:
            subregion_name[i.name.upper()] = i.subregion_name

        # ===

        for rinfo in region.roominfo_list:
            rnu = rinfo.name.upper()
            room = Room(
                room_positions[rnu],
                rinfo,
                special_params=Room.SpecialParams(
                    special_room_type=special_room_types.get(rnu, ""),
                    subregion_name=subregion_name.get(rnu, ""),
                ),
            )
            rooms.append(room)

        rooms_map = {i.name: i for i in rooms}

        connections: list[Connection] = []
        for i in region.map_txt.connections:
            name1, name2 = map(lambda x: x.upper(), (i.name_1, i.name_2))
            if name1 not in rooms_map:
                logger.debug(f"Endpoint {name1} not in rooms.")
                continue
            if name2 not in rooms_map:
                logger.debug(f"Endpoint {name2} not in rooms.")
                continue
            conn = Connection(
                rooms_map[name1],
                rooms_map[name2],
                np.array([i.position_1_x, i.position_1_y]),
                np.array([i.position_2_x, i.position_2_y]),
                i.direction_1,
                i.direction_2,
            )
            rooms_map[name1].append_edge(conn)
            rooms_map[name2].append_edge(conn)
            connections.append(conn)

        # ===

        rooms = list({i.name.upper(): i for i in rooms}.values())
        connections = list(
            {
                tuple(sorted((i.room1.name.upper(), i.room2.name.upper()))): i
                for i in connections
            }.values()
        )

        # ===

        self.subregion_names = set(
            map(lambda x: x.special_params.subregion_name, rooms)
        )
        if "" in self.subregion_names:
            self.subregion_names.remove("")

        self.rooms = rooms
        self.connections = connections

    @CachedProperty
    def room_map(self) -> dict[str, Room]:
        return {i.name.upper(): i for i in self.rooms}

    @CachedProperty
    def rooms(self) -> list[Room]:
        self._build_region()
        return self.rooms

    @CachedProperty
    def connections(self) -> list[Room]:
        self._build_region()
        return self.connections

    @CachedProperty
    def subregion_names(self) -> set[str]:
        self._build_region()
        return self.subregion_names

    @property
    def position(self):
        return self.combined_big_box(self.rooms).position

    @position.setter
    def position(self, value: np.ndarray):
        old_posi = self.combined_big_box(self.rooms).position
        delta = value - old_posi
        for r in self.rooms:
            r.position += delta

    @property
    def size(self):
        return self.combined_big_box(self.rooms).size

    @property
    def name(self):
        return self.info.name

    def opt(self, opt: optim.BaseOpt = optim.ForceOpt()):
        opt(self.rooms, self.connections)

    def _plot_box(self, ax: plt.Axes, *, box_size_factor=1):
        font_size = 30
        size = self.size * box_size_factor
        position = self.position - self.size * (box_size_factor - 1) / 2

        rect = Rectangle(
            position,
            size[0],
            size[1],
            linewidth=4,
            edgecolor=utils.color_hash(self.info.displayname),
            facecolor="none",
        )
        ax.add_patch(rect)

        title = [
            (
                f"{self.info.displayname_trans()} ({self.name})",
                "#ffffff",
                font_size,
            ),
            (
                f"{self.info.displayname}",
                "#ffffff",
                font_size,
            ),
        ]
        if len(self.subregion_names) > 1:
            for i in self.subregion_names:
                title.append(
                    (
                        f"{cons.translate(i)} ({i})",
                        utils.color_hash(i),
                        font_size * 0.66,
                    )
                )

        utils.draw_multiline_text_centered(
            ax=ax,
            lines=title,
            posi=position + size * np.array([0, 0]),
            alpha=1,
            bbox=dict(
                facecolor="#000000ff",
                edgecolor="#00000000",
                boxstyle="square,pad=0",
            ),
        )

    def plot(self, ax: plt.Axes, *, box_size_factor=1):
        rooms = self.rooms
        connections = self.connections

        self._plot_box(ax, box_size_factor=box_size_factor)

        for r in rooms:
            r.plot(ax)

        for c in connections:
            c.plot(ax)


class Teleport(Edge):

    class Warppoint(EndPoint):
        @property
        def rel_posi(self):
            return (self._room.position - self.box.position) + self._posi

        @property
        def box(self):
            return self.region

        def __init__(self, region: Region, room: Room, posi: np.ndarray):
            self.region = region
            self._room = room
            self._posi = posi

    def __init__(
        self, region_1, region_2, room_1, room_2, posi_1, posi_2, *, type_: str = ""
    ):
        from_point: Teleport.Warppoint = Teleport.Warppoint(region_1, room_1, posi_1)
        to_point: Teleport.Warppoint = Teleport.Warppoint(region_2, room_2, posi_2)
        super().__init__(from_point, to_point)
        self.end_point_1: Teleport.Warppoint
        self.end_point_2: Teleport.Warppoint

        self.type_: str = type_

    def plot(self, ax: plt.Axes):
        if self.end_point_2._room.name.upper() == "WRSA_L01":
            return
        conn_r1 = self.end_point_1.glo_posi
        conn_r2 = self.end_point_2.glo_posi
        linewidth = 8
        alpha = 0.2

        conn_vec = conn_r2 - conn_r1
        conn_vec_norm = np.linalg.norm(conn_vec)
        conn_vec_unit = conn_vec / conn_vec_norm  # ! DIV 0
        SPACE = 50
        num_arrows = int(conn_vec_norm / SPACE)
        for i in range(1, num_arrows + 1):
            t = i / (num_arrows + 1)  # 计算每个小箭头的比例位置
            p0 = conn_r1 + t * conn_vec
            p1 = p0 + conn_vec_unit * SPACE / 2
            if self.type_ == "SpinningTopSpot":
                color = plt.cm.Wistia(t)
            elif self.type_ == "WarpPoint":
                color = plt.cm.cool(t)
            else:
                color = "black"

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
