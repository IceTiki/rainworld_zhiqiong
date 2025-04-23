import constants as c
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import networkx as nx
import numpy as np
import assets
from pprint import pprint

FLAG_CN = False

plt.rcParams["font.sans-serif"] = ["MicroSoft YaHei"]
tele = {
    "WARA_P09": "WAUA_E01",
    "WARB_F18": "WARC_B12",
    "WARB_J01": "WARA_P05",
    "WARB_J08": "WRSA_L01",
    "WARC_B12": "WARB_F18",
    "WARC_F01": "WARA_E08",
    "WARD_R02": "WARB_F01",
    "WARD_R10": "WSSR_CRAMPED",
    "WARD_B41": "WSKD_B38",
    "WARD_E01": "WARG_D06_FUTURE",
    "WARD_E09": "WRSA_L01",
    "WARD_E12": "WSKC_A10",
    "WARD_E33": "WBLA_B08",
    "WARE_I14": "WARB_H13",
    "WARE_G15": "WRSA_L01",
    "WARE_H05": "WSKC_A03",
    "WARE_H16": "WTDA_Z01",
    "WARF_A06": "WSKA_D13",
    "WARF_B11": "WRFA_F01",
    "WARF_B14": "WSKB_C18",
    "WARF_B33": "WTDA_B12",
    "WARF_D06": "WSKD_B33",
    "WARF_D15": "WRSA_L01",
    "WARG_B31": "WTDA_A13",
    "WARG_H20": "WRSA_L01",
    "WARG_W11": "WSKD_B12",
    "WARG_W12": "WRSA_L01",
    "WARG_A06_FUTURE": "WTDB_A04",
    "WARG_D06_FUTURE": "WARD_E01",
    # "WAUA_BATH": "WAUA_TOYS",
    "WBLA_B08": "WARD_E33",
    "WBLA_D03": "WSKD_B01",
    "WBLA_E02": "WSSR_CRAMPED",
    "WBLA_J01": "WRSA_L01",
    "WDSR_A25": "WORA",
    "WGWR_DISPOSAL": "WORA",
    "WGWR_C09": "WORA",
    "WHIR_B13": "WORA",
    "WHIR_A22": "WORA",
    "WHIR_A06": "WORA",
    "WORA_DESERT6": "WRSA_L01",
    "WORA_STARCATCHER03": "WRSA_C01",
    # "WORA_STARCATCHER07": "WORA_STARCATCHER02",
    "WPTA_C05": "WRSA_L01",
    "WPTA_C07": "WVWA_H01",
    "WPTA_F03": "WARA_P08",
    "WRFA_F01": "WARF_B11",
    "WRFA_A12": "WSKA_D15",
    "WRFA_A21": "WRFB_A11",
    "WRFA_B09": "WRSA_L01",
    "WRFA_D08": "WRRA_B01",
    "WRFB_C07": "WRSA_L01",
    "WRFB_A22": "WARE_I01X",
    "WRFB_B12": "WVWA_E01",
    "WRRA_B01": "WRFA_D08",
    "WRRA_A07": "WSKB_C07",
    "WRRA_L01": "WRSA_L01",
    "WRRA_A26": "WTDB_A19",
    "WRSA_D01": "WARA_P17",
    "WSKA_D07": "WRSA_L01",
    "WSKA_D13": "WARF_A06",
    "WSKA_D15": "WRFA_A12",
    "WSKB_C18": "WARF_B14",
    "WSKB_C07": "WRRA_A07",
    "WSKC_A10": "WARD_E12",
    "WSKC_A23": "WPTA_B10",
    "WSKC_A25": "WRSA_L01",
    "WSKD_B12": "WARG_W11",
    "WSKD_B33": "WARF_D06",
    "WSKD_B34": "WRSA_L01",
    "WSKD_B38DRY": "WARD_B41",
    "WSKD_B40": "WARD_R15",
    # "WSSR_LAB6": "WRSA_L01",
    "WSSR_LAB6": "WORA_START",
    "WSUR_B09": "WORA",
    "WTDA_A13": "WARG_B31",
    "WTDA_Z01": "WARE_H16",
    "WTDA_Z07": "WRSA_L01",
    "WTDA_Z14": "WBLA_C01",
    "WTDB_A03": "WRSA_L01",
    "WTDB_A04": "WARG_A06_FUTURE",
    "WTDB_A19": "WRRA_A26",
    "WTDB_A26": "WRFB_D09",
    "WVWA_H01": "WPTA_C07",
    "WVWA_A09": "WRSA_L01",
    "WVWA_E01": "WRFB_B12",
    "WVWA_F03": "WARC_E03",
    "CC_C12": "WSKB_C17",
    "SH_A08": "WSKA_D02",
    "LF_B01W": "WRFA_SK04",
}

echos = {
    ("LF", "WRFA"),
    ("WARA", "WAUA"),
    ("WARB", "WARA"),
    ("WARC", "WARA"),
    ("WARD", "WARB"),
    ("WARE", "WSKC"),
    ("WARF", "WTDA"),
    ("WAUA", "NULL"),
    ("WAUA", "SB"),
    ("WBLA", "WSKD"),
    ("WPTA", "WARA"),
    ("WRFB", "WARE"),
    ("WSKC", "WPTA"),
    ("WSKD", "WARD"),
    ("WTDA", "WBLA"),
    ("WTDB", "WRFB"),
    ("WVWA", "WARC"),
    ("CC", "WSKB"),
    ("SH", "WSKA"),
    ("LF", "WRFA"),
}

best_posi = {
    "WARA": [830.68, 185.77],
    "WARB": [614.81, 156.8],
    "WARC": [730.01, 44.123999999999995],
    "WARD": [468.08, 271.59],
    "WARE": [489.71, 157.93],
    "WARF": [177.43, 205.76],
    "WARG": [389.52, 226.48],
    "WAUA": [968.8, 71.871],
    "WBLA": [367.74, 356.75],
    "WDSR": [764.9250000000001, 371.001],
    "WGWR": [847.713, 273.933],
    "WHIR": [942.213, 359.313],
    "WORA": [853.59, 362.19000000000005],
    "WPTA": [703.35, 137.4],
    "WRFA": [217.47, 29.773],
    "WRFB": [404.0, 43.948],
    "WRRA": [131.28, 77.797],
    "WRSA": [650.0, 450.0],
    "WSKA": [33.68000000000001, 38.0],
    "WSKB": [38.044, 135.8],
    "WSKC": [584.48, 231.3],
    "WSKD": [273.08, 314.0],
    "WSSR": [560.0, 390.0],
    "WSUR": [859.476, 450.129],
    "WTDA": [267.45, 188.89],
    "WTDB": [280.12, 103.92],
    "WVWA": [599.92, 37.693],
    "CC": [-70.0, 135.0],
    "SH": [-70.0, 37.0],
    "LF": [-70.0, 291],
}

tele_id = {
    k.split("_", maxsplit=1)[0]: v.split("_", maxsplit=1)[0] for k, v in tele.items()
}

# 构建有向图
G = nx.DiGraph()

# 添加节点
nodes = list(
    sorted(set([i for i in tele_id.values()]).union(set([i for i in tele_id.keys()])))
)

if FLAG_CN:
    id2cn = {
        i: f"{c.translate(c.REGION_DISPLAYNAME[i])}\n{c.REGION_DISPLAYNAME[i]}\n({i})"
        for i in nodes
    }
else:
    id2cn = {i: f"{c.REGION_DISPLAYNAME[i]}\n({i})" for i in nodes}

G.add_nodes_from(nodes)


for k, v in tele.items():
    k, v = map(lambda x: x.split("_", maxsplit=1)[0], (k, v))
    if v == "WRSA":
        continue
    G.add_edge(k, v)

# 使用 circular_layout 来生成圆形布局
# pos = nx.spring_layout(G, k=1, scale=0.4, iterations=50, seed=2)
# pos = nx.spring_layout(G, k=0.5, scale=0.4, iterations=100, seed=7)
# pos = nx.circular_layout(
#     G,
#     scale=0.4,
# )
# pos = nx.kamada_kawai_layout(G,scale=0.4)
pos = nx.nx_pydot.pydot_layout(G, prog="sfdp")
pos["WARD"] += np.array([0, -30])
pos["WARB"] += np.array([-30, -30])
pos["WSKA"] += np.array([-70, 20])
pos["WSKD"] += np.array([-20, 20])
pos["WTDA"] += np.array([-60, -20])
pos["WARG"] += np.array([40, -20])
pos["WARC"] += np.array([0, -30])
pos["WARA"] += np.array([0, 50])
pos["WARF"] += np.array([0, 50])
pos["WRSA"] = np.array([650.0, 450.0])
# pos["WBLA"] = np.array([125, 425.0])

for i in ("WDSR", "WGWR", "WHIR", "WSUR"):
    pos[i] = 0.3 * np.array(pos[i]) + 0.7 * np.array(pos["WORA"])

for i in ("WDSR", "WGWR", "WHIR", "WSUR", "WORA"):
    pos[i] += np.array([70, -250.0])


for k, v in best_posi.items():
    pos[k] = v

# 绘制图形
fig, ax = plt.subplots(figsize=(16, 16))
ax: plt.Axes

fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# 为每个节点放置图片
for node in nodes:
    img_path = (
        assets.RAIN_WORLD_PATH.streaming_assets_path
        / "mods\\watcher\\illustrations"
        / f"warp-{node.lower()}.png"
    )

    img = mpimg.imread(img_path)
    x, y = pos[node]
    img_size = 30  # 控制缩放程度
    ax.imshow(
        img,
        extent=(x - img_size, x + img_size, y - img_size, y + img_size),
    )

to_wrsa_zone = set()
for k, v in tele_id.items():
    if v == "WRSA":
        to_wrsa_zone.add(k)


for i in to_wrsa_zone:
    pos0 = pos[i]
    pos1 = pos["WRSA"]
    ax.plot(
        [pos0[0], pos1[0]], [pos0[1], pos1[1]], "purple", linestyle="--", linewidth=2
    )


# for k, v in tele_id.items():
#     pos0 = pos[k]
#     pos1 = pos[v]
#     ax.plot(
#         [pos0[0], pos1[0]], [pos0[1], pos1[1]], "black", linestyle="--", linewidth=1
#     )
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import LineCollection
from scipy.interpolate import splprep, splev
import numpy as np

# 使用三次贝塞尔曲线生成平滑曲线
for u, v in G.edges():
    r1, r2 = u, v
    start = np.array(pos[u])
    end = np.array(pos[v])
    vec = end - start
    vec_unit = vec / np.linalg.norm(vec)  # !DIV 0
    normal = np.array([-vec[1], vec[0]])

    start += vec_unit * 25
    end -= vec_unit * 25

    mid = (start + end) / 2
    control = mid + 0.05 * normal

    # 构造三点路径：起点-控制点-终点
    x = [start[0], control[0], end[0]]
    y = [start[1], control[1], end[1]]

    # 使用 B-spline 插值平滑曲线
    tck, u = splprep([x, y], k=2, s=0)
    unew = np.linspace(0, 1, 100)
    out = splev(unew, tck)
    verts = np.stack(out, axis=1)
    segments = np.array([verts[:-1], verts[1:]]).transpose(1, 0, 2)

    # 渐变线段
    colors = np.linspace(0, 1, len(segments))
    cmap = "coolwarm"
    color = plt.cm.coolwarm(1.0)
    if (r1, r2) in echos:
        cmap = "Wistia"
        color = plt.cm.Wistia(1.0)
    # if (r1, r2) in {("WRSA", "WARA")}:
    #     cmap = "Purples"
    #     color = plt.cm.Purples(1.0)

    if (r1, r2) in {("WARA", "WAUA"), ("WRSA", "WARA")}:
        ax.text(
            *mid,
            "需涟漪空间 (Need Ripplespace)" if FLAG_CN else "Need Ripplespace",
            ha="center",
            va="center",
            color="purple",
            fontweight="bold",
        )

    lc = LineCollection(segments, cmap=cmap, array=colors, linewidth=3, alpha=0.9)
    ax.add_collection(lc)

    # 添加箭头头部
    arrow = FancyArrowPatch(
        posA=verts[-2],
        posB=verts[-1],
        arrowstyle="-|>",
        mutation_scale=30,
        color=color,
        lw=0,
    )
    ax.add_patch(arrow)
# nx.draw_networkx_edges(
#     G,
#     pos,
#     ax=ax,
#     arrows=True,
#     width=2,
#     edge_color="#000000",
#     node_size=1000,
#     arrowsize=20,
#     connectionstyle="arc3,rad=0.2",
# )
# nx.draw_networkx_edges(
#     G,
#     pos,
#     ax=ax,
#     arrows=True,
#     width=1,
#     style=":",
#     edge_color="#FFF200",
#     node_size=1000,
#     arrowsize=20,
#     connectionstyle="arc3,rad=0.2",
# )

# 标签（带文字背景框）
nx.draw_networkx_labels(
    G,
    pos,
    ax=ax,
    labels=id2cn,
    font_size=10,
    font_weight="bold",
    bbox=dict(
        facecolor="#ffffffaa",  # 半透明白色
        edgecolor="#00000000",  # 无边框
        boxstyle="square,pad=0",  # 方形背景框
    ),
)

from matplotlib.lines import Line2D

handles = [
    Line2D([0], [0], color="#FFB919", lw=2, label="回响 (Echo)" if FLAG_CN else "Echo"),
    Line2D([0], [0], color="#B095D4", lw=2, label="裂隙 (Warppoint)" if FLAG_CN else "Warppoint"),
    Line2D(
        [0],
        [0],
        color="purple",
        linestyle="--",
        lw=2,
        label="涟漪空间裂隙 (Ripplespace Warppoint)" if FLAG_CN else "Ripplespace Warppoint",
    ),
]
# legend_line =
# FFB919
ax.legend(handles=handles, loc="upper left")

ax.axis("off")
plt.suptitle(
    "观望者传送总览\nWatcher Map Teleportation Overview" if FLAG_CN else "Watcher Map Teleportation Overview", fontsize=30, fontweight="bold"
)
plt.show()
