from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from loguru import logger
import numpy as np
import re
from pprint import pprint
from typing import Literal
from tqdm import tqdm

plt.rcParams["font.sans-serif"] = ["MicroSoft YaHei"]

ROOT = Path(__file__).parent.absolute()
RESOUCE_PATH = ROOT / "resource"
WORLD_PATH = RESOUCE_PATH / "world"
OUTPUT_PATH = ROOT / "output"

ZONE_ID_2_EN = {}
for i in WORLD_PATH.iterdir():
    displayname_path = i / "displayname.txt"
    if displayname_path.is_file():
        ZONE_ID_2_EN[i.name.lower()] = displayname_path.read_text()


EN_2_CN = {}
for i in Path("strings.txt").read_text()[2:].splitlines():
    en, cn = i.split("|")
    EN_2_CN[en] = cn


def find_special_object(path):
    directory = Path(path)
    spinning_top_spot = {}  # 回响
    warp_point = {}  # 传送点

    for file_path in directory.rglob("*.txt"):
        if not file_path.name.endswith("_settings.txt"):
            continue
        room_name = file_path.name.removesuffix("_settings.txt").lower()

        with file_path.open("r", encoding="utf-8") as file:
            content = file.read()

        patt1 = r"SpinningTopSpot.*Watcher~(?:\w+)~(?P<target>\w+)"
        for i in re.finditer(patt1, content):
            spinning_top_spot[room_name] = i.group("target").lower()

        patt2 = r"WarpPoint.*Watcher~(?:\w+)~(?P<target>\w+)"
        for i in re.finditer(patt2, content):
            warp_point[room_name] = i.group("target").lower()

    return {"spinning_top_spot": spinning_top_spot, "warp_point": warp_point}


SPECIAL_OBJECT: dict[Literal["spinning_top_spot", "warp_point"], dict[str, str]] = (
    find_special_object(WORLD_PATH)
)
pprint(SPECIAL_OBJECT)


def read_map_image_warp_txt(content: str):
    result = {}
    for i in content.splitlines():
        name, cord = i.split(": ")
        x, y, w, h = map(int, cord.split(","))
        result[name] = (x, y, w, h)
    return result


def plot_map(map_path: Path, output: Path):
    map_name = map_path.name
    img_path = map_path / f"map_{map_name}.png"
    if map_name == "wara":
        img_path = map_path / f"map_wara_modify.png"
    if map_name == "wora":
        img_path = map_path / f"map_wora_modify.png"
    map_image_path = map_path / f"map_image_{map_name}.txt"
    map_warb_path = map_path / f"map_{map_name}.txt"

    for i in (img_path, map_image_path, map_warb_path):
        if not i.is_file():
            return

    img = mpimg.imread(img_path)
    # 绿替换为白
    mask = img == np.array([0, 1, 0, 1])
    mask = np.all(mask, axis=2)
    img[mask] = [1, 1, 1, 1]
    # 红替换为白
    mask = img == np.array([1, 0, 0, 1])
    mask = np.all(mask, axis=2)
    img[mask] = [221 / 255, 177 / 255, 177 / 255, 1]

    warp_txt = map_image_path.read_text()

    fig, ax = plt.subplots(figsize=(16, 16))

    img_h, img_w, _ = img.shape
    # if map_name == "wora":
    #     img_h += 93

    # wora shift 185
    # wora shift 93
    # wara shift 372
    ax: plt.Axes
    # 显示图像
    ax.imshow(img)
    name_cord_map = read_map_image_warp_txt(warp_txt)
    # 连接处
    for line in map_warb_path.read_text().splitlines():
        if not line.startswith("Connection: "):
            continue
        _, content = line.split(": ")
        r1, r2, r1_cx, r1_cy, r2_cx, r2_cy, _, _ = content.split(",")
        r1_cx, r1_cy, r2_cx, r2_cy = map(int, (r1_cx, r1_cy, r2_cx, r2_cy))

        if r1 not in name_cord_map or r2 not in name_cord_map:
            continue
        r1_x1, r1_y1, _, _ = name_cord_map[r1]
        r2_x1, r2_y1, _, _ = name_cord_map[r2]

        plt.plot(
            [r1_x1 + r1_cx, r2_x1 + r2_cx],
            [(img_h - (r1_y1 + r1_cy)), (img_h - (r2_y1 + r2_cy))],
            c="#54C1F0",
            linestyle="-",
            linewidth=1,
            alpha=0.5,
        )

    # 地图框和地图名
    for name, cord in name_cord_map.items():
        name: str
        x1, y1, w, h = cord
        y1 = img_h - y1
        x2 = x1 + w
        y2 = y1 - h
        rect = Rectangle(
            (x1, y1),
            w,
            -h,
            linewidth=1,
            edgecolor="#138535",
            facecolor="none",
        )
        ax.add_patch(rect)
        text = name
        fontsize = 3
        if name.lower() in SPECIAL_OBJECT["spinning_top_spot"]:
            target = SPECIAL_OBJECT["spinning_top_spot"][name.lower()]
            if "_" in target:
                target = target.split("_")[0]
                target = EN_2_CN.get(ZONE_ID_2_EN.get(target, target), target)
            text += f" (回响->{target}) "
            fontsize = 6
        if name.lower() in SPECIAL_OBJECT["warp_point"]:
            target = SPECIAL_OBJECT["warp_point"][name.lower()]
            if "_" in target:
                target = target.split("_")[0]
                target = EN_2_CN.get(ZONE_ID_2_EN.get(target, target), target)
            text += f" (裂隙->{target}) "
            fontsize = 6
        ax.text(x1, y1, text, c="yellow", alpha=1, fontsize=fontsize)

    map_cn_name = EN_2_CN.get(ZONE_ID_2_EN.get(map_name, map_name), map_name)
    ax.axis('off')
    fig.suptitle(f"{map_cn_name}", fontsize=30)
    plt.tight_layout()
    plt.savefig(output, dpi=400)
    plt.close()


if __name__ == "__main__":
    for i in tqdm(list(WORLD_PATH.iterdir())):
        output = OUTPUT_PATH / f"{EN_2_CN.get(ZONE_ID_2_EN.get(i.name, i.name), i.name)}.png"
        plot_map(i, output)
