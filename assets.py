import winreg
import re
from pathlib import Path
import typing
import base64
import functools
from dataclasses import dataclass, field

import numpy as np

from loguru import logger
import utils
from utils import CachedProperty


class RainWorldPath:
    @staticmethod
    def locate_rain_world():
        # === Get Steam Path
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\\WOW6432Node\\Valve\\Steam"
        ) as key:
            steam_path, _ = winreg.QueryValueEx(key, "InstallPath")

        steam_path = Path(steam_path)
        # === Get Library Folders
        libraries_path = steam_path / "steamapps" / "libraryfolders.vdf"

        maybe_game_folders = [steam_path]

        if libraries_path.exists():
            for match_ in re.finditer(
                r"\t*\"path\"\t+\"(?P<path>.*)\"", libraries_path.read_text()
            ):
                match_: re.Match
                maybe_game_folders.append(Path(match_.group("path")))

        # === Find Rain World
        for path in maybe_game_folders:
            rainworld_path = path / "steamapps" / "common" / "Rain World"
            if rainworld_path.exists():
                return rainworld_path.absolute()

        raise FileNotFoundError("Locate Rain World Failed.")

    def __init__(self, rain_world_path: Path | None = None):
        self.rain_world_path = (
            rain_world_path if rain_world_path is not None else self.locate_rain_world()
        )

    @property
    def streaming_assets_path(self):
        path = self.rain_world_path / "RainWorld_Data\\StreamingAssets"
        assert path.exists()
        return path

    @property
    def mod_path(self):
        path = self.streaming_assets_path / "mods"
        assert path.exists()
        return path

    @property
    def world_path(self):
        path = self.streaming_assets_path / "world"
        assert path.exists()
        return path

    @property
    def mergedmods_path(self):
        path = self.streaming_assets_path / "mergedmods"
        assert path.exists()
        return path

    @functools.cache
    def get_translation(self, language="chi") -> dict[str, str]:
        language = language.lower()
        lines: list[str] = []

        text_path = self.mergedmods_path / "text" / f"text_{language}" / "strings.txt"
        text = text_path.read_text()[1:]
        lines.extend(text.splitlines())

        text_path = (
            self.streaming_assets_path / "text" / f"text_{language}" / "strings.txt"
        )
        text = text_path.read_text()[1:]
        lines.extend(text.splitlines())

        translation = {}
        for line in lines:
            en, trans = line.split("|")
            translation[en] = trans.replace("<LINE>", "\n")
        return translation


RAIN_WORLD_PATH = RainWorldPath()


class RegionPath(CachedProperty.Father):
    def __init__(
        self,
        region_name: str,
        *,
        mod_name: str | None,
        slugcat_name: str | None,
        rain_world_path: RainWorldPath = RAIN_WORLD_PATH,
    ):
        super().__init__()
        self.rain_world_path = rain_world_path
        self.mod_name: str | None = mod_name
        self.slugcat_name: str | None = slugcat_name
        self.region_name: str = region_name

    @property
    def vanilla_folder(self):
        return self.rain_world_path.world_path / self.region_name.lower()

    @property
    def vanilla_rooms_folder(self):
        return self.rain_world_path.world_path / f"{self.region_name.lower()}-rooms"

    @property
    def mod_path(self):
        return self.rain_world_path.mod_path / self.mod_name.lower()

    @property
    def mod_world_path(self):
        return self.mod_path / "world"

    @property
    def world_folders(self) -> list[Path]:
        """
        Notes
        ---
        查找顺序 : mod文件夹, 主文件夹, 依赖mod文件夹 (基于`mod_info.json`的priorities, BFS顺序)
        """
        mod_path = self.rain_world_path.mod_path

        mod_name_list = [self.mod_name] if self.mod_name is not None else []
        # use BFS
        for mod_name in mod_name_list:
            mod_info_path = mod_path / mod_name.lower() / "modinfo.json"
            if mod_info_path.exists():
                data: dict[str, typing.Any] = utils.JsonFile.load(mod_info_path)
                mod_name_list.extend(data.get("priorities", []))

        world_folders = [mod_path / i.lower() / "world" for i in mod_name_list]
        world_folders.insert(1, self.rain_world_path.world_path)
        return world_folders

    @property
    def region_folders(self) -> list[Path]:
        """
        Note
        ---
        - not ensure path exists
        """

        return [i / self.region_name.lower() for i in self.world_folders]

    @property
    def region_rooms_folders(self) -> list[Path]:
        """
        Note
        ---
        - not ensure path exists
        """

        return [i / f"{self.region_name.lower()}-rooms" for i in self.world_folders]

    @property
    def mod_modify_region_folder(self) -> Path:
        return self.mod_path / "modify" / "world" / self.region_name.lower()

    def locate_file(
        self,
        parents: list[Path],
        file_stem: str,
        file_suffix: str,
    ) -> Path | None:
        """
        Notes
        ---
        file_suffix
        - Example : ".txt"
        """
        file_names = [file_stem + file_suffix]
        if self.slugcat_name is not None:
            file_names.insert(
                0, file_stem + "-" + self.slugcat_name.lower() + file_suffix
            )

        for parent in parents:
            for file_name in file_names:
                path = parent / file_name
                if path.exists():
                    # logger.debug(f"Located {path}.")
                    return path

        logger.debug(f"Fail to locate {file_stem}{file_suffix}.")


class _BaseTxt(CachedProperty.Father):
    @classmethod
    def from_file(cls, path: str | Path):
        return cls(Path(path).read_text())

    @property
    def text(
        self,
    ) -> str:
        key = "text"
        return self._cache[key]

    @text.setter
    def text(self, text: str):
        self._cache.clear()  # *Note: Clean Cache
        key = "text"
        self._cache[key] = text

    @CachedProperty
    def lines(self) -> list[str]:
        return self.text.splitlines()

    def __init__(self, text: str):
        super().__init__()
        self.text = text


class WorldTxt(_BaseTxt):
    _WorldDataType = dict[str, list[list[str]]]

    @staticmethod
    def resolve_world_lines(lines: list[str]) -> _WorldDataType:
        world_data: WorldTxt._WorldDataType = {}
        key = None
        for line in lines:
            if not line or line.startswith("//"):
                continue
            if key is None:
                key = line
                world_data.setdefault(key, [])
                continue
            if line == f"END {key}":
                key = None
                continue
            world_data[key].append(line.split(" : "))

        return world_data

    @CachedProperty
    def data(self) -> _WorldDataType:
        """
        Possible Keys
        ---
        - ROOMS
        - CREATURES
        - BAT MIGRATION BLOCKAGES
        """
        return self.resolve_world_lines(self.lines)


class MergeWorldTxt(_BaseTxt):
    @CachedProperty
    def merge(self) -> WorldTxt._WorldDataType:
        merge_lines: list[str] = []
        start = False
        for line in self.lines:
            if not line or line.startswith("//"):
                continue
            if line == "[MERGE]":
                start = True
                continue
            if not start:
                continue
            if line == "[ENDMERGE]":
                break
            merge_lines.append(line)

        return WorldTxt.resolve_world_lines(merge_lines)


class MergeMapTxt(_BaseTxt):
    @CachedProperty
    def data(self):
        LINE_KEY = ["[REFERENCE]", "[IMAGE]", "[FIND]", "[REPLACE]"]

        res = {}
        stack = []
        for line in self.lines:
            if not line or line.startswith("//"):
                continue

            if line.startswith("[FILEDESTINATION]"):
                assert not stack
                file_destination = line.removeprefix("[FILEDESTINATION]")
                stack.append(file_destination)
                res.setdefault(file_destination, {})
                continue
            if line == "[ENDFILEDESTINATION]":
                assert len(stack) == 1
                stack.pop()
                continue
            if line == "[MERGE]":
                assert len(stack) == 1
                stack.append("[MERGE]")
                res[stack[0]].setdefault("[MERGE]", {})
                continue
            if line == "[ENDMERGE]":
                assert len(stack) == 2
                stack.pop()
                continue

            cursur = res
            for i in stack:
                cursur = cursur[i]

            for lk in LINE_KEY:
                if not line.startswith(lk):
                    continue
                cursur.setdefault(lk, [])
                cursur[lk].append(line.removeprefix(lk))
                break
            else:
                cursur.setdefault("", [])
                cursur[""].append(line)

        return res

    def merge(
        self, file_destination: str
    ) -> dict[typing.Literal["", "[REFERENCE]", "[IMAGE]"], list[str]]:
        return self.data.get(file_destination, {}).get("[MERGE]", {})

    def merge_lines(self, file_destination: str) -> list[str]:
        return self.data.get(file_destination, {}).get("[MERGE]", {}).get("", [])

    def find(self, file_destination: str) -> list[str]:
        return self.data.get(file_destination, {}).get("[FIND]", [])

    def replace(self, file_destination: str) -> list[str]:
        return self.data.get(file_destination, {}).get("[REPLACE]", [])


class MapTxt(_BaseTxt):
    _CONNECTION_ANNOTATION = (
        ("name_1", str),
        ("name_2", str),
        ("position_1_x", int),
        ("position_1_y", int),
        ("position_2_x", int),
        ("position_2_y", int),
        ("direction_1", int),
        ("direction_2", int),
    )
    _ROOM_ANNOTATION = (
        ("name", str),
        ("canon_pos_x", float),
        ("canon_pos_y", float),
        ("dev_pos_x", float),
        ("dev_pos_y", float),
        ("layer", int),
        ("subregion_name", str),
        ("room_width", int),
        ("room_height", int),
    )

    @dataclass
    class _ConnectionType:
        name_1: str = ""
        name_2: str = ""
        position_1_x: int = 0
        position_1_y: int = 0
        position_2_x: int = 0
        position_2_y: int = 0
        direction_1: int = 0
        direction_2: int = 0

    @dataclass
    class _RoomType:
        name: str = ""
        canon_pos_x: float = 0
        canon_pos_y: float = 0
        dev_pos_x: float = 0
        dev_pos_y: float = 0
        layer: int = 0
        subregion_name: str = ""
        room_width: int = 0
        room_height: int = 0

    @classmethod
    def _get_connection(cls, content: str) -> _ConnectionType:
        return cls._ConnectionType(
            **{
                key: type_(value)
                for (key, type_), value in zip(
                    cls._CONNECTION_ANNOTATION,
                    content.split(","),
                )
            }
        )

    @classmethod
    def _get_room(cls, head: str, content: str) -> _RoomType:
        return cls._RoomType(
            **{
                key: type_(value)
                for (key, type_), value in zip(
                    cls._ROOM_ANNOTATION,
                    [head] + content.split("><"),
                )
            }
        )

    @staticmethod
    def _sorted_tuple(iter: typing.Iterable):
        return tuple(sorted(iter))

    def _reset_rooms_and_connections(self) -> None:
        rooms: list[dict[str, typing.Any]] = []
        connections: list[dict[str, typing.Any]] = []
        for line in self.lines:
            if not line:
                continue
            head, content = line.split(": ", maxsplit=1)
            if head == "Connection":
                connections.append(self._get_connection(content))
            else:
                rooms.append(self._get_room(head, content))

        self.rooms = rooms
        self.connections = connections

    @CachedProperty
    def rooms(self) -> list[_RoomType]:
        self._reset_rooms_and_connections()
        return self.rooms

    @CachedProperty
    def connections(self) -> list[_ConnectionType]:
        self._reset_rooms_and_connections()
        return self.connections

    def merge_world_txt(self, merge_world_txt: MergeWorldTxt, slugcat="Watcher"):
        merge_data = merge_world_txt.merge

        hideroom = set()
        disconnected: set[tuple[str]] = set()
        for line in merge_data.get("CONDITIONAL LINKS", []):
            if slugcat not in line[0].split(","):
                continue
            if line[-1] == "DISCONNECTED":
                disconnected.add(self._sorted_tuple(line[1:3]))
            if line[1] == "HIDEROOM":
                hideroom.add(line[2])

        self.rooms = [i for i in self.rooms if i.name not in hideroom]
        self.connections = [
            i
            for i in self.connections
            if self._sorted_tuple((i.name_1, i.name_2)) not in disconnected
        ]

    def merge_map_txt(self, merge_map_txt: MergeMapTxt, file_destination: str):
        """
        Examples
        ---
        file_destination="map_xx-watcher.txt"
        """
        if file_destination not in merge_map_txt.data:
            logger.debug(f"f{file_destination} not in {merge_map_txt.data.keys()}.")
            return
        replace = {
            k: v
            for k, v in zip(
                merge_map_txt.find(file_destination),
                merge_map_txt.replace(file_destination),
            )
        }
        merge = merge_map_txt.merge_lines(file_destination)

        for i in self.rooms:
            i.name = replace.get(i.name, i.name)

        for i in self.connections:
            i.name_1 = replace.get(i.name_1, i.name_1)
            i.name_2 = replace.get(i.name_2, i.name_2)

        for line in merge:
            if not line:
                continue
            head, content = line.split(": ", maxsplit=1)
            if head == "Connection":
                self.connections.append(self._get_connection(content))
            else:
                self.rooms.append(self._get_room(head, content))


class RoomTxt(_BaseTxt):
    @dataclass
    class MapImColor:
        water: np.ndarray = field(default_factory=lambda: utils.rgba_pixel("#007AAE"))
        water_acid: np.ndarray = field(
            default_factory=lambda: utils.rgba_pixel("#7AE732")
        )
        wall: np.ndarray = field(default_factory=lambda: utils.rgba_pixel("#000000"))
        background: np.ndarray = field(
            default_factory=lambda: utils.rgba_pixel("#000000", 0.25)
        )
        entrance: np.ndarray = field(
            default_factory=lambda: utils.rgba_pixel("#ffff00", 0.5)
        )
        pipe: np.ndarray = field(
            default_factory=lambda: utils.rgba_pixel("#ffff00", 0.25)
        )
        bar: np.ndarray = field(default_factory=lambda: utils.rgba_pixel("#880015", 1))
        im5: np.ndarray = field(default_factory=lambda: utils.rgba_pixel("#A349A4", 1))
        im8: np.ndarray = field(default_factory=lambda: utils.rgba_pixel("#3F48CC", 1))
        sand: np.ndarray = field(default_factory=lambda: utils.rgba_pixel("#BF8F16", 1))

    class MapIm:
        def __init__(self, map_matrix: np.ndarray):
            self.map_matrix = map_matrix

        @property
        def wall(self):
            return self.map_matrix[:, :, 1:2]

        @property
        def background(self):
            return self.map_matrix[:, :, 6:7]

        @property
        def entrance(self):
            return self.map_matrix[:, :, 4:5]

        @property
        def pipe(self):
            return self.map_matrix[:, :, 3:4]

        @property
        def bar(self):
            return self.wall & self.background | self.map_matrix[:, :, 2:3]

        @property
        def im5(self):
            return self.map_matrix[:, :, 5:6]

        @property
        def im8(self):
            return self.map_matrix[:, :, 8:9]

    CHANNEL = 8 + 1

    @CachedProperty
    def name(self):
        return self.lines[0].upper()

    @CachedProperty
    def line10_base64_items(self) -> list[np.ndarray]:
        return [
            np.frombuffer(base64.b64decode(j), np.uint8)
            for i in self.lines[10].split("<<DIV - A>>")
            for j in i.split("<<DIV - B>>")
        ]

        # for i, im in enumerate(rt.line10):
        #     if im.shape[0] != 43904:
        #         continue
        #     # if i != 18:
        #     #     continue
        #     arr = np.transpose(np.reshape(im, (98, 112, 4)), (1,0,2))[::-1,:,:]
        #     plt.imshow(arr)
        #     plt.title(i)
        #     plt.show()
        #     plt.close()

    @CachedProperty
    def map_im(self) -> MapIm:
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
        map_str = self.lines[11].split("|")[:-1]
        map_num = [tuple(map(int, i.split(","))) for i in map_str]
        map_layer = np.array(
            [list(map(lambda x: x in i, range(self.CHANNEL))) for i in map_num],
            dtype=np.uint8,
        )
        return self.MapIm(
            np.reshape(
                map_layer,
                (self.width, self.height, self.CHANNEL),
            ).transpose((1, 0, 2))
        )

    @CachedProperty
    def border_type(self) -> typing.Literal["Solid", "Passable"]:
        border = self.lines[4].split(": ")[1]
        assert border in {"Solid", "Passable"}
        return border

    @CachedProperty
    def width(self):
        self._load_line1()
        return self.width

    @CachedProperty
    def height(self):
        self._load_line1()
        return self.height

    @CachedProperty
    def water_level(self):
        self._load_line1()
        return self.water_level

    def _load_line1(self):
        line1 = self.lines[1].split("|")
        size = line1[0]
        if len(line1) >= 2:
            water_level = line1[1]
        else:
            water_level = -1
        self.width, self.height = map(int, size.split("*"))
        self.water_level = int(water_level)

    def __init__(self, text):
        super().__init__(text)

    def test_show_all(self):
        from matplotlib import pyplot as plt
        import utils

        if True:
            fig, axs = plt.subplots(
                nrows=self.CHANNEL // 4 + 1,
                ncols=4,
                figsize=(16, 16),
                constrained_layout=True,
            )
            ax_list: list[plt.Axes] = axs.flatten().tolist()

            for i, ax in zip(range(self.CHANNEL), ax_list):
                ax.imshow(self.map_im[:, :, i], cmap="gray", clip_on=False)
                ax.set_title(str(i))
                ax.axis("off")

            ax = ax_list[-1]

        else:
            fig, ax = plt.subplots()
            ax: plt.Axes
            ax.imshow(self.water_im * utils.rgba_pixel("#007AAE"), clip_on=False)
            ax.imshow(self.wall_im * utils.rgba_pixel("#000000"), clip_on=False)
            ax.imshow(
                self.background_im * utils.rgba_pixel("#000000", 0.25), clip_on=False
            )
            ax.imshow(self.pipe_im * utils.rgba_pixel("#ffff00", 0.25), clip_on=False)
            ax.imshow(
                self.entrance_im * utils.rgba_pixel("#ffff00", 0.5), clip_on=False
            )
            ax.imshow(self.bar_im * utils.rgba_pixel("#880015", 1), clip_on=False)
            ax.imshow(
                self.map_im[:, :, 5:6] * utils.rgba_pixel("#A349A4", 1),
                clip_on=False,
            )
            ax.imshow(
                self.map_im[:, :, 8:9] * utils.rgba_pixel("#3F48CC", 1),
                clip_on=False,
            )

        plt.show()


class RoomSettingTxt(_BaseTxt):

    @CachedProperty
    def name(self):
        return self.lines[0].upper()

    @CachedProperty
    def data(self) -> dict[str, str]:
        data_: dict[str, str] = {}
        for line in self.lines:
            if line == "":
                continue
            key, value = line.split(": ")
            data_[key] = value
        return data_

    @CachedProperty
    def placed_objects(self) -> list[tuple[str, float, float, list[str]]]:
        """
        Notes
        ---
        x, y经过处理, 还原为相对坐标 (除以20)

        但是other的数据并未经过处理, 记得除以20
        """
        res = []

        if "PlacedObjects" not in self.data:
            return res
        for obj in self.data["PlacedObjects"].split(", "):
            if obj == "":
                continue
            name, x, y, other = obj.split("><")
            x, y = map(lambda x: float(x) / 20, (x, y))  # ! div by 20
            other = other.split("~")
            res.append((name, x, y, other))

        return res

    @CachedProperty
    def effects(self) -> list[tuple[str, str, float, float]]:
        res: list[tuple[str, str, float, float]] = []
        if "Effects" in self.data:
            for eff in self.data["Effects"].split(", "):
                eff: str
                if not eff:
                    continue
                name, ukn, x, y = eff.split("-")[:4]
                x, y = map(float, (x, y))

                res.append((name, ukn, x, y))

        return res

    def __init__(self, text):
        super().__init__(text)


class RoomInfo(CachedProperty.Father):
    @dataclass
    class _WaterInfo:
        water_flux_max_level: int
        water_flux_min_level: int
        lethal_water: bool = False

    def __init__(self, roomtxt: RoomTxt, roomsettingtxt: RoomSettingTxt | None):
        super().__init__()
        self.roomtxt: RoomTxt = roomtxt
        self.roomsettingtxt: RoomSettingTxt | None = roomsettingtxt

    @property
    def name(self):
        return self.roomtxt.name

    @CachedProperty
    def water_info(self) -> _WaterInfo:
        res = self._WaterInfo(
            water_flux_min_level=self.roomtxt.water_level,
            water_flux_max_level=self.roomtxt.water_level,
        )
        if self.roomsettingtxt is not None:
            for eff in self.roomsettingtxt.effects:
                eff: list[tuple[str, str, float, float]]
                name, ukn, x, y = eff
                if name == "LethalWater":
                    res.lethal_water = True
                elif name == "WaterFluxMaxLevel":
                    res.water_flux_max_level = y
                elif name == "WaterFluxMinLevel":
                    res.water_flux_min_level = y

        return res

    @property
    def width(self):
        return self.roomtxt.width

    @property
    def height(self):
        return self.roomtxt.height

    def __str__(self):
        return f"<RoomInfo: {self.name} (Setting: {self.roomsettingtxt is not None})>"


class RegionInfo(CachedProperty.Father):
    def __init__(self, region_path: RegionPath):
        super().__init__()
        self.region_path: RegionPath = region_path

    @property
    def name(self):
        return self.region_path.region_name

    @CachedProperty
    def displayname(self):
        for i in self.region_path.region_folders:
            path = i / "displayname.txt"
            if path.exists():
                return path.read_text()
        return self.name

    @functools.cache
    def displayname_trans(self, language: str = "chi") -> str:
        trans = self.region_path.rain_world_path.get_translation(language)
        return trans.get(self.displayname, self.displayname)

    @CachedProperty
    def merge_world_txt(self) -> MergeWorldTxt | None:

        if self.region_path.mod_name is None:
            return None

        reg_name = self.region_path.region_name.lower()
        path = self.region_path.mod_modify_region_folder / f"world_{reg_name}.txt"
        if not path.exists():
            return None
        return MergeWorldTxt.from_file(path)

    @CachedProperty
    def merge_map_txt(self) -> MergeMapTxt | None:

        if self.region_path.mod_name is None:
            return None

        reg_name = self.region_path.region_name.lower()
        path = self.region_path.mod_modify_region_folder / f"map_{reg_name}.txt"
        if not path.exists():
            return None
        return MergeMapTxt.from_file(path)

    @CachedProperty
    def map_txt(self) -> MapTxt:
        """
        Notes
        ---
        Automatic merge with `self.merge_txt`.
        """
        reg_name = self.region_path.region_name.lower()
        path = self.region_path.locate_file(
            self.region_path.region_folders, f"map_{reg_name}", ".txt"
        )
        assert path is not None

        map_txt = MapTxt.from_file(path)

        self.map_txt = map_txt

        # MERGE: 必须先执行merge_map_txt
        if self.region_path.mod_name is not None:
            ideal_map_txt_name = (
                f"map_{self.name.lower()}-{self.region_path.slugcat_name.lower()}.txt"
            )
            merge_map_txt = self.merge_map_txt
            if path.name != ideal_map_txt_name and merge_map_txt is not None:
                self.map_txt.merge_map_txt(merge_map_txt, ideal_map_txt_name)

        merge_world_txt = self.merge_world_txt
        if merge_world_txt is not None and self.map_txt is not None:
            self.map_txt: MapTxt
            self.map_txt.merge_world_txt(merge_world_txt)

        return map_txt

    @CachedProperty
    def world_txt(self) -> WorldTxt:
        reg_name = self.region_path.region_name.lower()
        path = self.region_path.locate_file(
            self.region_path.region_folders, f"world_{reg_name}", ".txt"
        )
        assert path is not None

        world_txt = WorldTxt.from_file(path)

        self.world_txt = world_txt

        return world_txt

    @CachedProperty
    def roominfo_list(self) -> list[RoomInfo]:
        res = []
        for r in self.map_txt.rooms:
            if r.name.lower().startswith("gate_"):
                room_txt_path = self.region_path.locate_file(
                    [i / "gates" for i in self.region_path.world_folders],
                    f"{r.name.lower()}",
                    ".txt",
                )
                room_txt_settings_path = self.region_path.locate_file(
                    [i / "gates" for i in self.region_path.world_folders],
                    f"{r.name.lower()}_settings",
                    ".txt",
                )
            else:
                room_txt_path = self.region_path.locate_file(
                    self.region_path.region_rooms_folders, f"{r.name.lower()}", ".txt"
                )
                room_txt_settings_path = self.region_path.locate_file(
                    self.region_path.region_rooms_folders,
                    f"{r.name.lower()}_settings",
                    ".txt",
                )

            if room_txt_path is None:
                continue

            room_txt = RoomTxt.from_file(room_txt_path)
            room_txt.name = r.name.upper()  # room txt may not have right room name

            room_txt_settings = (
                RoomSettingTxt.from_file(room_txt_settings_path)
                if room_txt_settings_path is not None
                else None
            )

            res.append(RoomInfo(roomtxt=room_txt, roomsettingtxt=room_txt_settings))

        return res
