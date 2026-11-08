"""Generate the small layered-board toolbar icon (development only)."""
from pathlib import Path
from PIL import Image, ImageDraw

folder = Path(__file__).resolve().parents[1] / "resources"
folder.mkdir(exist_ok=True)
image = Image.new("RGBA", (192, 192))
draw = ImageDraw.Draw(image)
for y, color in ((126, "#527fc4"), (86, "#ce7d2c"), (46, "#6ed6c4")):
    draw.polygon([(16, y), (96, y-32), (176, y), (96, y+32)], fill=color)
    draw.line([(16, y), (96, y+32), (176, y)], fill="#ecf5fa", width=4)
draw.line([(56, 60), (56, 140)], fill="#eacb7e", width=6)
draw.line([(136, 60), (136, 140)], fill="#eacb7e", width=6)
for size in (24, 48):
    image.resize((size, size), Image.Resampling.LANCZOS).save(folder / f"layers_{size}.png")
