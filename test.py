import fitz

from tikilib import binary as tb

import constants as cons
from PIL import Image, ImageOps
from pathlib import Path

def gene_gif():
    # 获取所有 PNG 文件路径
    image_paths = [i for i in Path(cons.TMP_OUTPUT_PATH).iterdir() if i.suffix == ".png"]

    # 打开所有图片
    images = [Image.open(p) for p in image_paths]

    # 获取最大宽度和高度
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    # 对所有图片进行 padding 到相同大小
    padded_images = []

    images += [images[-1] for _ in range(50)]
    for img in images:
        delta_w = max_width - img.width
        delta_h = max_height - img.height
        padding = (
            delta_w // 2,
            delta_h // 2,
            delta_w - delta_w // 2,
            delta_h - delta_h // 2,
        )
        padded = ImageOps.expand(img, padding, fill=(0, 0, 0))  # 黑色背景
        padded_images.append(padded.convert("P"))  # 转为调色板模式以兼容 GIF

    # 保存为 GIF
    padded_images[0].save(
        "anima.gif",
        save_all=True,
        append_images=padded_images[1:],
        duration=100,  # 每帧 100ms
        loop=0,
    )
# ============================

match 2:
    case 1:
        gene_gif()
    case 2:
        tb.MuPdf.pdf2png(cons.OUTPUT_PATH / "Watcher Big Map.pdf", cons.OUTPUT_PATH, zoom=2)
print("done!")
