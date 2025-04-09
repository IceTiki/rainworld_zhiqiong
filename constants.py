from pathlib import Path
import re
from typing import Literal
import json

ROOT = Path(__file__).parent.absolute()
RESOUCE_PATH = ROOT / "resource"
WORLD_PATH = RESOUCE_PATH / "world"
OUTPUT_PATH = ROOT / "output"

ZONE_ID_2_EN = {}
for i in WORLD_PATH.iterdir():
    displayname_path = i / "displayname.txt"
    if displayname_path.is_file():
        ZONE_ID_2_EN[i.name.upper()] = displayname_path.read_text()


EN_2_CN = {}
for i in (RESOUCE_PATH / "strings.txt").read_text()[2:].splitlines():
    en, cn = i.split("|")
    EN_2_CN[en] = cn


def find_special_object(path):
    directory = Path(path)
    spinning_top_spot = {}  # 回响
    warp_point = {}  # 传送点

    for file_path in directory.rglob("*.txt"):
        if not file_path.name.endswith("_settings.txt"):
            continue
        room_name = file_path.name.removesuffix("_settings.txt").upper()

        with file_path.open("r", encoding="utf-8") as file:
            content = file.read()

        patt1 = r"SpinningTopSpot.*Watcher~(?:\w+)~(?P<target>\w+)"
        for i in re.finditer(patt1, content):
            spinning_top_spot[room_name] = i.group("target").upper()

        patt2 = r"WarpPoint.*Watcher~(?:\w+)~(?P<target>\w+)"
        for i in re.finditer(patt2, content):
            warp_point[room_name] = i.group("target").upper()

    return {"spinning_top_spot": spinning_top_spot, "warp_point": warp_point}


def zone_id_2_cn(id_: str):
    id_ = id_.upper()
    return EN_2_CN.get(ZONE_ID_2_EN.get(id_, id_), ZONE_ID_2_EN.get(id_, id_))


SPECIAL_OBJECT: dict[Literal["spinning_top_spot", "warp_point"], dict[str, str]] = (
    find_special_object(WORLD_PATH)
)

PLACE_OBJECT_NAME = {
    "KarmaFlower": "业力花",
    "SpinningTopSpot": "回响",
    "WarpPoint": "裂隙",
}

ROOM_RECOMMAND_POSITION = {}
CORNIMAP_PATH = RESOUCE_PATH / "地图/源文件(用Cornifer打开)"
for cornimap in CORNIMAP_PATH.iterdir():
    if not cornimap.suffix == ".cornimap":
        continue
    with open(cornimap, "r", encoding="utf-8") as f:
        data = json.load(f)
    for obj in data["objects"]:
        ROOM_RECOMMAND_POSITION[obj["name"].upper()] = [obj["pos"]["x"], obj["pos"]["y"]]

