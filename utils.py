import numpy as np
from matplotlib import pyplot as plt

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

def cubic_bezier_curve(p1:np.ndarray, p2:np.ndarray, vec1:np.ndarray, vec2:np.ndarray, ax:plt.Axes, plot_params = {}):
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
        return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3

    # 生成t的值，用于绘制曲线
    t_values = np.linspace(0, 1, 100)
    bezier_curve = np.array([cubic_bezier(t) for t in t_values])

    # 绘制贝塞尔曲线
    ax.plot(bezier_curve[:, 0], bezier_curve[:, 1], **plot_params)