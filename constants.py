from pathlib import Path
import re
from typing import Literal
import json

ROOT = Path(__file__).parent.absolute()
RESOUCE_PATH = ROOT / "resource"
WORLD_PATH = RESOUCE_PATH / "comb_world"
OUTPUT_PATH = ROOT / "output"

ZONE_ID_2_EN = {}
for i in WORLD_PATH.iterdir():
    displayname_path = i / "displayname.txt"
    if displayname_path.is_file():
        ZONE_ID_2_EN[i.name.upper()] = displayname_path.read_text()


EN_2_CN = {}
for txt in (RESOUCE_PATH / "en2cn").iterdir():
    for line in txt.read_text()[2:].splitlines():
        en, cn = line.split("|")
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


def en_2_cn(en: str):
    return EN_2_CN.get(en, en)


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


SPECIAL_ROOMS: dict[str, str] = find_special_room(WORLD_PATH)
SPECIAL_ROOMS.update({"HI_WS01": "SHELTER"})

# 坐标记得除20
TELEPORTS = [
    [591.3841, 1316.087, "LF_B01W", "WRFA_SK04", "(珊瑚洞穴|WRFA_SK04)"],
    [1530.319, 1163.53, "WARA_P09", "WAUA_E01", "(上古城市|WAUA_E01|需要满级业力)"],
    [654.3783, 380.9694, "WARB_F18", "WARC_B12", "(腐臭幽谷|WARC_B12)"],
    [1698.804, 1205.208, "WARB_J01", "WARA_P05", "(破碎露台|WARA_P05)"],
    [463.3199, 429.7054, "WARB_J08", "WRSA_L01", "(恶魔|WRSA_L01|需要满级业力)"],
    [1531.017, 207.0246, "WARC_B12", "WARB_F18", "(盐化区|WARB_F18)"],
    [357.6498, 254.8015, "WARC_F01", "WARA_E08", "(破碎露台|WARA_E08)"],
    [254.4166, 1922.818, "WARD_R02", "WARB_F01", "(盐化区|WARB_F01)"],
    [467.4592, 284.1814, "WARD_R10", "WSSR_CRAMPED", "(不幸演化|WSSR_CRAMPED)"],
    [341.05, 370.2433, "WARD_B41", "WSKD_B38", "(雾罩海岸|WSKD_B38)"],
    [655.6271, 537.6578, "WARD_E01", "WARG_D06_FUTURE", "(表面|WARG_D06_FUTURE)"],
    [421.5186, 295.6393, "WARD_E09", "WRSA_L01", "(恶魔|WRSA_L01|需要满级业力)"],
    [1451.573, 435.2671, "WARD_E12", "WSKC_A10", "(风暴海岸|WSKC_A10)"],
    [463.8557, 1078.388, "WARD_E33", "WBLA_B08", "(不毛之地|WBLA_B08)"],
    [622.9644, 367.931, "WARE_I14", "WARB_H13", "(盐化区|WARB_H13)"],
    [289.6384, 506.9496, "WARE_G15", "WRSA_L01", "(恶魔|WRSA_L01|需要满级业力)"],
    [1467.321, 157.4426, "WARE_H05", "WSKC_A03", "(风暴海岸|WSKC_A03)"],
    [1332.679, 627.4624, "WARE_H16", "WTDA_Z01", "(炎热沙漠|WTDA_Z01)"],
    [309.0695, 1027.797, "WARF_A06", "WSKA_D13", "(暴雨铁路|WSKA_D13)"],
    [1609.447, 1276.307, "WARF_B11", "WRFA_F01", "(珊瑚洞穴|WRFA_F01)"],
    [1276.55, 998.1927, "WARF_B14", "WSKB_C18", "(普照港|WSKB_C18)"],
    [315.3611, 326.0649, "WARF_B33", "WTDA_B12", "(炎热沙漠|WTDA_B12)"],
    [1342.546, 1169.312, "WARF_D06", "WSKD_B33", "(雾罩海岸|WSKD_B33)"],
    [480.2742, 478.2177, "WARF_D15", "WRSA_L01", "(恶魔|WRSA_L01|需要满级业力)"],
    [309.4246, 1473.448, "WARG_B31", "WTDA_A13", "(炎热沙漠|WTDA_A13)"],
    [490.1779, 408.4085, "WARG_H20", "WRSA_L01", "(恶魔|WRSA_L01|需要满级业力)"],
    [597.1206, 198.6596, "WARG_W11", "WSKD_B12", "(雾罩海岸|WSKD_B12)"],
    [528.422, 333.6021, "WARG_W12", "WRSA_L01", "(恶魔|WRSA_L01|需要满级业力)"],
    [1664.141, 1126.082, "WARG_A06_FUTURE", "WTDB_A04", "(凄凉地带|WTDB_A04)"],
    [431.5437, 1231.817, "WARG_D06_FUTURE", "WARD_E01", "(冷库|WARD_E01)"],
    # [377.1898, 1866.425, "WAUA_TOYS", "NULL", "(古人线结局)"],
    [
        1506.179,
        223.313,
        "WAUA_BATH",
        "WAUA_TOYS",
        "(古人线结局, 一次性传送)\n(上古城市|WAUA_TOYS)",
    ],  # SB_D07
    [698.7733, 402.3932, "WBLA_B08", "WARD_E33", "(冷库|WARD_E33)"],
    [294.1339, 1034.489, "WBLA_D03", "WSKD_B01", "(雾罩海岸|WSKD_B01)"],
    [699.9332, 350.1698, "WBLA_E02", "WSSR_CRAMPED", "(不幸演化|WSSR_CRAMPED)"],
    [4689.768, 1240.903, "WBLA_J01", "WRSA_L01", "(恶魔|WRSA_L01|需要满级业力)"],
    [559.5618, 486.6902, "WDSR_A25", "WORA_START", "(外缘)"],
    [775.4473, 1195.533, "WGWR_DISPOSAL", "WORA_START", "(外缘)"],
    [1656.884, 886.6998, "WGWR_C09", "WORA_START", "(外缘)"],
    [660.2367, 1454.95, "WHIR_B13", "WORA_START", "(外缘)"],
    [660.05, 440.7469, "WHIR_A22", "WORA_START", "(外缘)"],
    [560.1949, 470.2051, "WHIR_A06", "WORA_START", "(外缘)"],
    [476.5552, 374.4548, "WORA_DESERT6", "WRSA_L01", "(恶魔|WRSA_L01|需要满级业力)"],
    [447.4414, 181.183, "WORA_STARCATCHER03", "WRSA_C01", "(恶魔|WRSA_C01)"],
    [
        656.6814,
        2014.8,
        "WORA_STARCATCHER07",
        "WORA_STARCATCHER02",
        "(外缘|WORA_STARCATCHER02)",
    ],
    [573.2645, 914.3567, "WPTA_C05", "WRSA_L01", "(恶魔|WRSA_L01|需要满级业力)"],
    [687.5314, 387.975, "WPTA_C07", "WVWA_H01", "(翠绿水道|WVWA_H01)"],
    [470.4816, 4230.22, "WPTA_F03", "WARA_P08", "(破碎露台|WARA_P08)"],
    [1591.35, 1070.618, "WRFA_F01", "WARF_B11", "(以太山脊|WARF_B11)"],
    [296.6536, 364.47, "WRFA_A12", "WSKA_D15", "(暴雨铁路|WSKA_D15)"],
    [479.1837, 556.9419, "WRFA_A21", "WRFB_A11", "(涡流泵|WRFB_A11)"],
    [572.8152, 293.5591, "WRFA_B09", "WRSA_L01", "(恶魔|WRSA_L01|需要满级业力)"],
    [391.2469, 470.8, "WRFA_D08", "WRRA_B01", "(锈蚀残骸|WRRA_B01)"],
    [286.0731, 1961.768, "WRFB_C07", "WRSA_L01", "(恶魔|WRSA_L01|需要满级业力)"],
    [318.1143, 347.7879, "WRFB_A22", "WARE_I01X", "(供热管道|WARE_I01X)"],
    [498.2588, 1218.085, "WRFB_B12", "WVWA_E01", "(翠绿水道|WVWA_E01)"],
    [1602.552, 530.9075, "WRRA_B01", "WRFA_D08", "(珊瑚洞穴|WRFA_D08)"],
    [351.6757, 473.8065, "WRRA_A07", "WSKB_C07", "(普照港|WSKB_C07)"],
    [658.6959, 1921.339, "WRRA_L01", "WRSA_L01", "(恶魔|WRSA_L01|需要满级业力)"],
    [609.2027, 381.5981, "WRRA_A26", "WTDB_A19", "(凄凉地带|WTDB_A19)"],
    [493.7858, 1231.547, "WRSA_D01", "WARA_P17", "(破碎露台|WARA_P17)"],
    [278.3431, 2045.426, "WSKA_D07", "WRSA_L01", "(恶魔|WRSA_L01|需要满级业力)"],
    [652.0967, 250.9022, "WSKA_D13", "WARF_A06", "(以太山脊|WARF_A06)"],
    [1675.7, 551.2532, "WSKA_D15", "WRFA_A12", "(珊瑚洞穴|WRFA_A12)"],
    [641.5557, 153.4376, "WSKB_C18", "WARF_B14", "(以太山脊|WARF_B14)"],
    [406.2, 295.7475, "WSKB_C07", "WRRA_A07", "(锈蚀残骸|WRRA_A07)"],
    [1036.085, 463.1304, "WSKC_A10", "WARD_E12", "(冷库|WARD_E12)"],
    [1488.462, 2892.485, "WSKC_A23", "WPTA_B10", "(信号尖塔|WPTA_B10)"],
    [416.5931, 352.9247, "WSKC_A25", "WRSA_L01", "(恶魔|WRSA_L01|需要满级业力)"],
    [1444.198, 474.0925, "WSKD_B12", "WARG_W11", "(表面|WARG_W11)"],
    [3412.011, 325.1455, "WSKD_B33", "WARF_D06", "(以太山脊|WARF_D06)"],
    [505.8369, 1132.554, "WSKD_B34", "WRSA_L01", "(恶魔|WRSA_L01|需要满级业力)"],
    [557.3563, 289.1472, "WSKD_B38", "WARD_B41", "(冷库|WARD_B41)"],  # WSKD_B38DRY
    [545.6909, 1246.809, "WSKD_B40", "WARD_R15", "(冷库|WARD_R15)"],
    [482.0687, 368.4155, "WSSR_LAB6", "WORA_START", "(外缘|NULL)"],
    [520.6984, 1220.099, "WSUR_B09", "WORA_START", "(外缘)"],
    [443.7992, 262.6651, "WTDA_A13", "WARG_B31", "(表面|WARG_B31)"],
    [2449.251, 549.05, "WTDA_Z01", "WARE_H16", "(供热管道|WARE_H16)"],
    [1816.8, 307.9868, "WTDA_Z07", "WRSA_L01", "(恶魔|WRSA_L01|需要满级业力)"],
    [666.8002, 510.7575, "WTDA_Z14", "WBLA_C01", "(不毛之地|WBLA_C01)"],
    [409.4899, 331.6025, "WTDB_A03", "WRSA_L01", "(恶魔|WRSA_L01|需要满级业力)"],
    [393.9413, 403.5609, "WTDB_A04", "WARG_A06_FUTURE", "(表面|WARG_A06_FUTURE)"],
    [482.251, 198.6747, "WTDB_A19", "WRRA_A26", "(锈蚀残骸|WRRA_A26)"],
    [1208.719, 446.0524, "WTDB_A26", "WRFB_D09", "(涡流泵|WRFB_D09)"],
    [488.2585, 4930.332, "WVWA_H01", "WPTA_C07", "(信号尖塔|WPTA_C07)"],
    [482.251, 546.1553, "WVWA_A09", "WRSA_L01", "(恶魔|WRSA_L01|需要满级业力)"],
    [4736.164, 330.2081, "WVWA_E01", "WRFB_B12", "(涡流泵|WRFB_B12)"],
    [305.8491, 1297.115, "WVWA_F03", "WARC_E03", "(腐臭幽谷|WARC_E03)"],
]
