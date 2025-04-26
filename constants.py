from pathlib import Path
import re
from typing import Literal, Any, Callable
import json
import functools
import enum

import numpy as np
from PIL import Image


class JsonFile:
    @staticmethod
    def load(json_path="data.json", encoding="utf-8"):
        """读取Json文件"""
        with open(json_path, "r", encoding=encoding) as f:
            return json.load(f)

    @staticmethod
    def write(item, json_path="data.json", encoding="utf-8", ensure_ascii=False):
        """写入Json文件"""
        with open(json_path, "w", encoding=encoding) as f:
            json.dump(item, f, ensure_ascii=ensure_ascii)


ROOT = Path(__file__).parent.absolute()
OUTPUT_PATH = ROOT / "output"
TMP_OUTPUT_PATH = OUTPUT_PATH / "temp"
CONSTANTS_FILES_PATH = ROOT / "constants_files"
SPECIAL_OBJECT_ICON_FOLDER = CONSTANTS_FILES_PATH / "special_object_icons"
KARMA_ICON_PATH = CONSTANTS_FILES_PATH / "karma"
TMP_OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
OBJECT_ICONS_PATH = CONSTANTS_FILES_PATH / "object_icons"


def load_constant_file(file_stem: str):
    """
    Returns
    ---
    default = {}
    """
    path_ = CONSTANTS_FILES_PATH / (file_stem + ".json")
    if not path_.exists():
        return {}
    return JsonFile.load(path_)


def save_constant_file(file_stem: str, item: dict | list | Any):
    path_ = CONSTANTS_FILES_PATH / (file_stem + ".json")
    return JsonFile.write(item, path_)


def _copy_deco(func: Callable[..., np.ndarray]):
    def warp(*args, **kwargs):
        return func(*args, **kwargs).copy()

    return warp


@_copy_deco
@functools.cache
def load_img(name: str, suffix: str = ".png", root=SPECIAL_OBJECT_ICON_FOLDER) -> np.ndarray:
    img = Image.open(root / (name + suffix)).convert("RGBA")
    img_array = np.array(img, np.uint8)
    return img_array


OBJECT_ICONS = {
    i.stem: load_img(i.stem, root=OBJECT_ICONS_PATH)
    for i in OBJECT_ICONS_PATH.iterdir()
}

SLUGCAT_REGIONS: dict[str, list[str]] = load_constant_file("slugcat_regions")
REGION_DISPLAYNAME: dict[str, str] = load_constant_file("region_displayname")
TRANSLATIONS: dict[str, dict[str, str]] = load_constant_file("translations")

_PLACE_OBJECT_NAME_CONSTANT_FILE_NAME = "place_object_list"


def translate(text: str, language="chi") -> str:
    return TRANSLATIONS[language].get(text, text)


def _region_displayname():
    from assets import RAIN_WORLD_PATH

    trans = {}
    for world in [RAIN_WORLD_PATH.world_path] + [
        i / "world" for i in RAIN_WORLD_PATH.mod_path.iterdir()
    ]:
        if not world.exists():
            continue
        for region in world.iterdir():
            if not region.exists():
                continue
            name = region.name.upper()
            displayname_path = region / "displayname.txt"
            if not displayname_path.exists():
                continue
            trans[name] = displayname_path.read_text()
    JsonFile.write(trans, CONSTANTS_FILES_PATH / "region_displayname.json")


def _translations():
    from assets import RAIN_WORLD_PATH

    trans = {}
    for lan in ["chi", "eng", "fre", "ger", "ita", "jap", "kor", "por", "rus", "spa"]:
        trans[lan] = RAIN_WORLD_PATH.get_translation(lan)

    JsonFile.write(trans, CONSTANTS_FILES_PATH / "translations.json")


class ObjectType(str, enum.Enum):
    DECORATION = "DECORATION"
    ITEM = "ITEM"
    UNDEFINED = ""


def _obj_list():
    from assets import RAIN_WORLD_PATH, RoomSettingTxt

    data = load_constant_file(_PLACE_OBJECT_NAME_CONSTANT_FILE_NAME)

    all_settings_files = list(RAIN_WORLD_PATH.rain_world_path.rglob("*_settings.txt"))
    for f in all_settings_files:
        if "modify" in f.parts:
            continue
        rst = RoomSettingTxt.from_file(f)
        for obj in rst.placed_objects:
            data.setdefault(obj[0], ObjectType.UNDEFINED)

    for i in OBJECT_ICONS_PATH.iterdir():
        if i.stem not in data:
            continue
        data.setdefault(i.stem, ObjectType.ITEM)

    save_constant_file(_PLACE_OBJECT_NAME_CONSTANT_FILE_NAME, data)


def _update_constants():
    _region_displayname()
    _translations()
    _obj_list()


OBJECT_TYPE: dict[str, ObjectType] = load_constant_file(
    _PLACE_OBJECT_NAME_CONSTANT_FILE_NAME
)

PLACE_OBJECT_NAME = {
    "KarmaFlower": "业力花",
    "SpinningTopSpot": "回响",
    "GhostSpot": "回响",
    "WarpPoint": "裂隙",
    "DataPearl": "珍珠",
    "DynamicWarpTarget": "动态路径终点",
    "ScavengerTreasury": "拾荒者宝库",
}
SPECIAL_ROOM_TYPE_2_CN = {
    "SWARMROOM": "蝠蝇巢室",
    "PERF_HEAVY": "大型房间",
    "SCAVTRADER": "拾荒者商人",
    "SHELTER": "庇护所",
    "SCAVOUTPOST": "拾荒者前哨",
    "ANCIENTSHELTER": "远古庇护所",
    "GATE": "业力门",
}

if __name__ == "__main__":
    pass
    # _update_constants()
