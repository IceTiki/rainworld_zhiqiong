import fitz

from tikilib import binary as tb

import constants as cons

tb.MuPdf.pdf2png(cons.OUTPUT_PATH / "Watcher Big Map.pdf", cons.OUTPUT_PATH, zoom=2)
print("done!")
