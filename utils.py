import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import hashlib
from pathlib import Path
from loguru import logger
import json
from typing import TypeVar, Generic, Callable, Optional, Any

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., T])


class CachedProperty(property, Generic[T]):
    """
    for `CachedObject`
    """

    class Father:
        def __init__(self):
            self._cache: dict = {}

    def __init__(self, func: F):
        self.func: F = func
        self.name: str = func.__name__
        self.func_setter: Optional[Callable[[Any, T], None]] = None

    def __get__(self, instance: Father, owner: type) -> T:
        if instance is None:
            return self
        # if not hasattr(instance, "_cache"):
        #     instance._cache = {}

        key = f"{owner.__name__}.{self.name}"
        if key not in instance._cache:
            instance._cache[key] = self.func(instance)
        return instance._cache[key]

    def __set__(self, instance: Father, value: T):
        if self.func_setter is not None:
            self.func_setter(instance, value)
        else:
            key = f"{instance.__class__.__name__}.{self.name}"
            instance._cache[key] = value

    def setter(self, fset: Callable[[Any, T], None]) -> "CachedProperty[T]":
        self.func_setter = fset
        return self


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


def bezier_curve(
    p0: np.ndarray, p0r: np.ndarray, p1l: np.ndarray, p1: np.ndarray, num=100
):
    t = np.linspace(0, 1, num)
    curve = (
        (1 - t)[:, None] ** 3 * p0
        + 3 * (1 - t)[:, None] ** 2 * t[:, None] * p0r
        + 3 * (1 - t)[:, None] * t[:, None] ** 2 * p1l
        + t[:, None] ** 3 * p1
    )
    return curve


def alpha_blend(
    fg: np.ndarray,
    bg: np.ndarray,
):
    fg = fg.astype(np.float32) / 255.0
    bg = bg.astype(np.float32) / 255.0

    fg_rgb = fg[..., :3]
    fg_alpha = fg[..., 3:]

    bg_rgb = bg[..., :3]
    bg_alpha = bg[..., 3:]

    out_alpha = fg_alpha + bg_alpha * (1 - fg_alpha)
    out_rgb = (fg_rgb * fg_alpha + bg_rgb * bg_alpha * (1 - fg_alpha)) / np.clip(
        out_alpha, 1e-6, 1.0
    )
    out_image = np.concatenate([out_rgb, out_alpha], axis=-1)
    return (out_image * 255).astype(np.uint8)


def uniform(arr: np.ndarray):
    norm = np.linalg.norm(arr)
    if norm != 0:
        return arr / np.linalg.norm(arr)
    return np.zeros_like(arr)


def rgba_pixel(color: str = "#ffffff", alpha: float = 1):
    carr = [int(color[i : i + 2], 16) for i in (1, 3, 5)] + [alpha * 255]
    return np.array([[carr]], np.uint8)


def calculate_cos_theta(a: np.ndarray, b: np.ndarray):
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cos_theta


def cubic_bezier_curve(
    p1: np.ndarray,
    p2: np.ndarray,
    vec1: np.ndarray,
    vec2: np.ndarray,
    ax: plt.Axes,
    plot_params={},
):
    """
    绘制一条三次贝塞尔曲线，给定起点、终点和两个控制向量。

    参数：
    p1: 起点 (x1, y1)
    p2: 终点 (x2, y2)
    vec1: 控制向量1，定义曲线方向 (cx1, cy1)
    vec2: 控制向量2，定义曲线方向 (cx2, cy2)
    ax: matplotlib 的 Axes 对象，用于绘图
    """
    # 计算控制点
    p0 = p1
    p3 = p2
    p1 = p0 + vec1
    p2 = p3 + vec2

    # 定义三次贝塞尔曲线的公式
    def cubic_bezier(t):
        return (
            (1 - t) ** 3 * p0
            + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t**2 * p2
            + t**3 * p3
        )

    # 生成t的值，用于绘制曲线
    t_values = np.linspace(0, 1, 100)
    bezier_curve = np.array([cubic_bezier(t) for t in t_values])

    # 绘制贝塞尔曲线
    ax.plot(bezier_curve[:, 0], bezier_curve[:, 1], **plot_params)


def pad_list(list_, target_length: int, padding_value=None) -> list:
    return list_ + [padding_value] * (target_length - len(list_))


def str_sha256(string: str) -> str:
    return hashlib.sha256(string.encode()).hexdigest()


def register_hex_colors(hex_colors):
    latex_defs = []
    hex_to_latex = {}

    for i, hex_code in enumerate(hex_colors):
        name = f"color{i}"
        clean = hex_code.lstrip("#")
        latex_defs.append(rf"\definecolor{{{name}}}{{HTML}}{{{clean}}}")
        hex_to_latex[hex_code] = name

    return latex_defs, hex_to_latex


def color_hash(string: str, register=False) -> str:
    salt = "salt"
    hex = str_sha256(f"{string}{salt}")[:6]
    # if register:
    #     name = f"color{hex}"
    #     mpl.rcParams["pgf.preamble"] += "\n" + rf"\definecolor{{{name}}}{{HTML}}{{{hex}}}"
    #     return name
    return f"#{hex}"


def draw_multiline_text_centered(
    ax: plt.Axes,
    lines: list[tuple[str, str, float]],
    posi: np.ndarray = np.array([0, 0]),
    fontname="Microsoft YaHei",
    rel_fontscale: float = 1,  # 用于从 fontsize 估算轴坐标高度的缩放系数,
    alpha=1,
    bbox=dict(
                facecolor="#ffffff88",
                edgecolor="#00000000",
                boxstyle="square,pad=0",
            )
):

    cx, cy = posi
    # 根据 fontsize 粗略估算每行占据的“相对高度”（用于垂直居中）
    line_heights = [fontsize * rel_fontscale for _, _, fontsize in lines]
    total_height = sum(line_heights)
    start_y = cy  # 从顶行开始往下画

    for i, (text, color, fontsize) in enumerate(lines):
        y = start_y - sum(line_heights[:i])  # 累积偏移
        ax.text(
            cx,
            y,
            text,
            ha="left",
            va="top",
            color=color,
            fontsize=fontsize,
            # transform=ax.transAxes,
            fontname=fontname,
            fontweight="bold",
            alpha=alpha,
            bbox=bbox,
        )


RAIN_WORLD_STREAMING_ASSETS = Path(
    r"D:\Environment\Application\Steam\steamapps\common\Rain World\RainWorld_Data\StreamingAssets"
)


def world_file_locator(
    file_rel_path: Path,
    world_paths: list[Path] = [
        RAIN_WORLD_STREAMING_ASSETS / r"mods\watcher\world",
        RAIN_WORLD_STREAMING_ASSETS / r"mods\moreslugcats\world",
        RAIN_WORLD_STREAMING_ASSETS / "world",
    ],
    mod_name="watcher",
) -> Path | None:
    for world in world_paths:
        for file_path in (
            file_rel_path.with_stem(file_rel_path.stem + f"-{mod_name}"),
            file_rel_path,
        ):
            file_full_path = world / file_path
            if not file_full_path.is_file():
                continue
            return file_full_path

    logger.warning(f"{file_rel_path} not found.")
    return None
