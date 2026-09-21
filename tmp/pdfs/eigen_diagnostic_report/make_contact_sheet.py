from pathlib import Path

from PIL import Image, ImageDraw


root = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/tmp/pdfs/eigen_diagnostic_report")
pages = sorted(root.glob("render-*.png"))
thumb_w = 350
thumb_h = 495
margin = 22
label_h = 28
cols = 3
rows = (len(pages) + cols - 1) // cols
sheet = Image.new("RGB", (margin + cols * (thumb_w + margin), margin + rows * (thumb_h + label_h + margin)), "#d8e0e5")
draw = ImageDraw.Draw(sheet)
for i, path in enumerate(pages):
    image = Image.open(path).convert("RGB")
    image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    x = margin + (i % cols) * (thumb_w + margin)
    y = margin + (i // cols) * (thumb_h + label_h + margin)
    sheet.paste(image, (x, y))
    draw.text((x, y + thumb_h + 5), f"page {i + 1}", fill="#15242e")
sheet.save(root / "contact_sheet.png")
