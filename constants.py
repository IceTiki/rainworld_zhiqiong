from pathlib import Path
import re
from typing import Literal, Any
import json


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


def load_constant_file(file_stem: str):
    path_ = CONSTANTS_FILES_PATH / (file_stem + ".json")
    if not path_.exists():
        return {}
    return JsonFile.load(path_)


def save_constant_file(file_stem: str, item: dict | list | Any):
    path_ = CONSTANTS_FILES_PATH / (file_stem + ".json")
    return JsonFile.write(item, path_)


SLUGCAT_REGIONS: dict[str, list[str]] = load_constant_file("slugcat_regions")
REGION_DISPLAYNAME: dict[str, str] = load_constant_file("region_displayname")
TRANSLATIONS: dict[str, dict[str, str]] = load_constant_file("translations")


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


def _update_constants():
    _region_displayname()
    _translations()


PLACE_OBJECT_NAME = {
    "KarmaFlower": "业力花",
    "SpinningTopSpot": "回响",
    "GhostSpot": "回响",
    "WarpPoint": "裂隙",
    "DataPearl": "珍珠",
}
SPECIAL_ROOM_TYPE_2_CN = {
    "SWARMROOM": "蝠蝇巢室",
    "PERF_HEAVY": "大型房间",
    "SCAVTRADER": "拾荒者商人",
    "SHELTER": "庇护所",
    "SCAVOUTPOST": "拾荒者前哨",
    "ANCIENTSHELTER": "古代庇护所",
    "GATE": "业力门",
}

if __name__ == "__main__":
    pass
    # _update_constants()
