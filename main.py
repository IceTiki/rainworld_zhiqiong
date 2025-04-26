import os
from pathlib import Path
from typing import Literal

from pprint import pprint
from matplotlib import pyplot as plt
from loguru import logger
from tqdm import tqdm
import numpy as np

import constants as cons
from plot import Region, Teleport, Edge, EndPoint
from assets import RegionInfo, RegionPath
import optim
from colls import Box
import utils


def plot_watcher_big_map():
    optim.DEBUG = True

    regions = [
        Region.from_slugcat_info(i, mod_name="Watcher", slugcat_name="Watcher")
        for i in cons.SLUGCAT_REGIONS["Watcher"]
    ]

    reg_map: dict[str, Region] = {i.name.upper(): i for i in regions}

    watcher_warppoints: list[dict[str, str | list[float]]] = cons.load_constant_file(
        "watcher_warppoints"
    )
    watcher_posi_cache: dict[str, dict] = cons.load_constant_file("watcher_posi_cache")
    for regname, posidata in watcher_posi_cache.items():
        reg_map[regname].position = posidata["posi"]
        for room_name, room_posi in posidata["room_posi"].items():
            reg_map[regname].room_map[room_name].position = np.array(room_posi) * 1.1

    teleports: list[Teleport] = []

    opt_boxes_map = {i: Box(i.position, i.size) for i in regions}
    opt_edges = []

    for i in watcher_warppoints:
        type_ = i["obj_type"]
        from_ = i["from_room"]
        to_ = i["to_room"]
        from_coord = i["from_coord"]
        to_coord = i["to_coord"]

        regname1 = from_.split("_", maxsplit=1)[0].upper()
        regname2 = to_.split("_", maxsplit=1)[0].upper()

        tp = Teleport(
            reg_map[regname1],
            reg_map[regname2],
            reg_map[regname1].room_map[from_],
            reg_map[regname2].room_map[to_],
            np.array(from_coord),
            np.array(to_coord),
            type_=type_,
        )
        teleports.append(tp)

        if regname2.upper() == "WRSA":  # 恶魔 Daemon
            continue

        if regname1.upper() in {"SU", "HI", "CC", "SH", "LF"}:
            continue

        box1, box2 = opt_boxes_map[reg_map[regname1]], opt_boxes_map[reg_map[regname2]]
        edge = Edge(
            EndPoint(box1, tp.end_point_1.rel_posi),
            EndPoint(box2, tp.end_point_2.rel_posi),
        )
        box1.append_edge(edge)
        box2.append_edge(edge)
        opt_edges.append(
            Edge(
                EndPoint(box1, tp.end_point_1.rel_posi),
                EndPoint(box2, tp.end_point_2.rel_posi),
            )
        )

    gate_rooms = set(
        j.name.upper()
        for i in regions
        for j in i.rooms
        if j.name.upper().startswith("GATE_")
    )
    for i in gate_rooms:
        _, r1, r2 = i.split("_")
        if r1 not in reg_map or r2 not in reg_map:
            continue

        reg1, reg2 = reg_map[r1], reg_map[r2]

        if i not in reg1.room_map or i not in reg2.room_map:
            continue

        for flag in range(2):
            if flag == 1:
                reg1, reg2 = reg2, reg1

            tp = Teleport(
                reg1,
                reg2,
                reg1.room_map[i],
                reg2.room_map[i],
                reg1.room_map[i].size / 2,
                reg2.room_map[i].size / 2,
                type_="Gate",
            )
            teleports.append(tp)

            box1, box2 = opt_boxes_map[reg1], opt_boxes_map[reg2]
            edge = Edge(
                EndPoint(box1, tp.end_point_1.rel_posi),
                EndPoint(box2, tp.end_point_2.rel_posi),
            )
            box1.append_edge(edge)
            box2.append_edge(edge)
            opt_edges.append(
                Edge(
                    EndPoint(box1, tp.end_point_1.rel_posi),
                    EndPoint(box2, tp.end_point_2.rel_posi),
                )
            )

    opt_boxes = list(opt_boxes_map.values())
    for opt in [
        optim.AlignOpt(),
        optim.ForceOpt(repel_factor=100),
        optim.AvoidOverlap(step=10),
        optim.NonConnGraphLayout(),
    ]:
        opt(opt_boxes, opt_edges)

    for reg, box in opt_boxes_map.items():
        reg.position = box.position * 1.1

    # ===========

    fig, ax = plt.subplots(facecolor="white")
    ax: plt.Axes

    big_box = Box.combined_big_box(regions)

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
    fig.tight_layout()
    MAX_EDGE_PIXEL = 100000
    max_dpi = MAX_EDGE_PIXEL // max(delta_x / FACTOR, delta_y / FACTOR)
    # plt.tight_layout()

    for reg in tqdm(regions, desc="plot regions"):
        reg.plot(ax)

    for tp in teleports:
        tp.plot(ax)

    output = cons.OUTPUT_PATH / "Watcher Big Map.pdf"
    logger.info("Saving figure...")
    fig.savefig(output, dpi=max_dpi, transparent=False, facecolor="white")
    plt.close()
    logger.info("Done!")


def plot_watcher_regions():
    # optim.DEBUG = True  # ! DEBUG
    regs = [
        Region(RegionInfo(RegionPath(i, mod_name="Watcher", slugcat_name="Watcher")))
        for i in cons.SLUGCAT_REGIONS["Watcher"]
    ]

    bar = tqdm(regs)

    posi_cache_name = "watcher_posi_cache"
    posi_cache_data: dict[
        str, dict[Literal["posi", "room_posi"], list[float] | dict[str, list[float]]]
    ] = cons.load_constant_file(posi_cache_name)

    for reg in bar:
        bar.desc = reg.info.displayname_trans("chi")
        # if reg.name.upper() != "WRRA":
        #     continue
        # if reg.name.upper() != "WARB":
        #     continue

        if reg.name.upper() in posi_cache_data:
            for room_name, posi in posi_cache_data[reg.name.upper()][
                "room_posi"
            ].items():
                reg.room_map[room_name].position = np.array(posi)
        else:
            reg.opt(optim.AlignOpt())
            reg.opt(optim.ForceOpt())
            reg.opt(optim.AvoidOverlap())
            reg.opt(optim.NonConnGraphLayout())

            posi_cache_data[reg.name] = {
                "posi": list(reg.position),
                "room_posi": {i.name: list(i.position) for i in reg.rooms},
            }

        fig, ax = plt.subplots(facecolor="white")
        ax: plt.Axes

        for r in reg.rooms:
            r.position *= 1.1

        reg.plot(ax, box_size_factor=1.1)

        x0, y0 = reg.left_down
        x1, y1 = reg.right_top

        delta_x = reg.size[0]
        delta_y = reg.size[1]
        ax.set_xlim(x0 - 0.1 * delta_x, x1 + 0.1 * delta_x)
        ax.set_ylim(y0 - 0.1 * delta_y, y1 + 0.1 * delta_y)

        ax.set_aspect(1)
        ax.axis("off")

        fig.set_size_inches(max(delta_x / 50, 7), max(delta_y / 50, 7), forward=True)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0.2)
        # fig.tight_layout()
        fig.savefig(
            cons.OUTPUT_PATH
            / f"{reg.info.displayname} {reg.info.displayname_trans("chi")} ({reg.name.upper()}).png",
            dpi=300,
        )
        plt.close()

    # cons.save_constant_file(posi_cache_name, posi_cache_data)


def test_load_watcher():
    objs = {}
    regs = [
        Region(RegionInfo(RegionPath(i, mod_name="Watcher", slugcat_name="Watcher")))
        for i in cons.SLUGCAT_REGIONS["Watcher"]
    ]
    for reg in regs:
        for r in reg.rooms:
            rst = r.info.roomsettingtxt
            if rst is None:
                continue
            for obj in rst.placed_objects:
                name, x, y, prop = obj
                objs[name] = None

    cons.save_constant_file("place_object_list", objs)


if __name__ == "__main__":

    for i in cons.TMP_OUTPUT_PATH.iterdir():
        if not i.suffix == ".png":
            continue
        os.remove(i)

    plot_watcher_regions()
    plot_watcher_big_map()
    # test_load_watcher()
