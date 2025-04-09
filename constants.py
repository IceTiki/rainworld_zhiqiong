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
    "DataPearl": "珍珠",
}
SPECIAL_ROOM_TYPE_2_CN = {
    "SWARMROOM": "蝠蝇巢室",
    "PERF_HEAVY": "大型房间",
    "SCAVTRADER": "拾荒者商人",
    "SHELTER": "庇护所",
    "SCAVOUTPOST": "拾荒者前哨",
    "ANCIENTSHELTER": "古代庇护所",
}

ROOM_RECOMMAND_POSITION = {}
CORNIMAP_PATH = RESOUCE_PATH / "地图/源文件(用Cornifer打开)"
for cornimap in CORNIMAP_PATH.iterdir():
    if not cornimap.suffix == ".cornimap":
        continue
    with open(cornimap, "r", encoding="utf-8") as f:
        data = json.load(f)
    for obj in data["objects"]:
        ROOM_RECOMMAND_POSITION[obj["name"].upper()] = [
            obj["pos"]["x"],
            obj["pos"]["y"],
        ]


def find_special_room(path):
    directory = Path(path)
    special_rooms = {}

    for file_path in directory.rglob("*.txt"):
        if not file_path.name.startswith("world_"):
            continue

        in_room = False
        for line in file_path.read_text().splitlines():
            if line == "ROOMS":
                in_room = True
            elif line == "END ROOMS":
                break
            if not in_room:
                continue
            items = line.split(" : ")
            if len(items) > 2:
                special_rooms[items[0].upper()] = items[2]

    return special_rooms


SPECIAL_ROOMS = find_special_room(WORLD_PATH)

