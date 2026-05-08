"""
Build the Track Record trophy wall — composite the 5 venue badges
on a single editorial canvas with subtle gold/black accent typography.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ASSETS = Path(__file__).resolve().parents[1]
BADGE_DIR = ASSETS / "badges"
OUT = Path(__file__).parent / "trophy_wall.png"

# Canvas
W, H = 2400, 1100
PAD_X = 80
COLS = 5
GAP = 60
BADGE_SIZE = (W - 2 * PAD_X - (COLS - 1) * GAP) // COLS

# Layout: badges centered vertically, with header above and meta below
TOP_HEADER_Y = 80
BADGE_TOP = 240
META_Y_OFFSET = 30  # below the badge

# Tier metadata
ORDER = [
    {
        "key": "fse",
        "label": "FSE 2026",
        "tier": "Diamond  Tier-3",
        "info": "ACM · CCF-A",
        "count": "5 papers",
    },
    {
        "key": "icml",
        "label": "ICML 2026",
        "tier": "Diamond  Tier-3",
        "info": "ML Flagship",
        "count": "1 paper",
    },
    {
        "key": "tosem",
        "label": "TOSEM 2026",
        "tier": "Diamond  Tier-3",
        "info": "ACM Trans.",
        "count": "2 articles",
    },
    {
        "key": "aei",
        "label": "AEI 2026",
        "tier": "Platinum  Tier-2",
        "info": "Elsevier · SCI Q1",
        "count": "1 minor rev.",
    },
    {
        "key": "icogb",
        "label": "ICoGB 2026",
        "tier": "Gold  Tier-1",
        "info": "Civil Eng.",
        "count": "1 paper",
    },
]


# Try to load nice fonts; fall back to default
def load_font(
    size: int, bold: bool = False, italic: bool = False
) -> ImageFont.FreeTypeFont:
    candidates = []
    if italic and bold:
        candidates += [
            "/Library/Fonts/Georgia Bold Italic.ttf",
            "/System/Library/Fonts/Supplemental/Georgia Bold Italic.ttf",
        ]
    if bold:
        candidates += [
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ]
    if italic:
        candidates += [
            "/Library/Fonts/Georgia Italic.ttf",
            "/System/Library/Fonts/Supplemental/Georgia Italic.ttf",
        ]
    candidates += [
        "/Library/Fonts/Georgia.ttf",
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for c in candidates:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size)
            except OSError:
                continue
    return ImageFont.load_default()


def text_w(draw: ImageDraw.ImageDraw, txt: str, font) -> int:
    bbox = draw.textbbox((0, 0), txt, font=font)
    return bbox[2] - bbox[0]


def main() -> None:
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # ---- header ----
    f_eyebrow = load_font(28)
    f_title = load_font(78, bold=True)
    f_title_i = load_font(78, italic=True)

    eyebrow = "VERIFIED TRACK RECORD · 2025–2026"
    eyebrow_w = text_w(draw, eyebrow, f_eyebrow)
    draw.text(
        ((W - eyebrow_w) // 2, TOP_HEADER_Y), eyebrow, fill="#6b6b72", font=f_eyebrow
    )

    # title: "A trophy case, quietly accumulating."
    line1 = "A trophy case,"
    line2 = "quietly accumulating."
    line1_w = text_w(draw, line1, f_title)
    line2_w = text_w(draw, line2, f_title_i)
    title_y = TOP_HEADER_Y + 60
    draw.text(((W - line1_w) // 2, title_y), line1, fill="#0a0a0b", font=f_title)
    draw.text(((W - line2_w) // 2, title_y + 90), line2, fill="#6b6b72", font=f_title_i)

    # ---- badges row ----
    f_meta_label = load_font(22, bold=True)
    f_meta_venue = load_font(40, bold=True)
    f_meta_info = load_font(22)
    f_meta_count = load_font(26, bold=True)

    badge_y = BADGE_TOP + 220  # shift down to leave room for title
    for i, item in enumerate(ORDER):
        x = PAD_X + i * (BADGE_SIZE + GAP)
        # load badge
        bpath = BADGE_DIR / f"{item['key']}.png"
        if not bpath.exists():
            print(f"  ! missing {bpath}")
            continue
        b = Image.open(bpath).convert("RGBA")
        b.thumbnail((BADGE_SIZE, BADGE_SIZE), Image.Resampling.LANCZOS)
        # paste centered in cell
        bx = x + (BADGE_SIZE - b.width) // 2
        by = badge_y
        canvas.paste(b, (bx, by), b)

        # ---- meta beneath ----
        meta_x_center = x + BADGE_SIZE // 2
        my = by + b.height + META_Y_OFFSET

        tier = item["tier"]
        first = tier.split()[0]
        tier_color = {
            "Diamond": "#1f3a8a",
            "Platinum": "#3a3a42",
            "Gold": "#8a6a1c",
        }.get(first, "#3a3a42")
        tw = text_w(draw, tier, f_meta_label)
        draw.text(
            (meta_x_center - tw // 2, my), tier, fill=tier_color, font=f_meta_label
        )

        my += 38
        venue = item["label"]
        vw = text_w(draw, venue, f_meta_venue)
        draw.text(
            (meta_x_center - vw // 2, my), venue, fill="#0a0a0b", font=f_meta_venue
        )

        my += 56
        info = item["info"]
        iw = text_w(draw, info, f_meta_info)
        draw.text((meta_x_center - iw // 2, my), info, fill="#6b6b72", font=f_meta_info)

        my += 36
        count = item["count"]
        cw = text_w(draw, count, f_meta_count)
        # tiny separator line above count
        draw.line(
            [(meta_x_center - 80, my - 6), (meta_x_center + 80, my - 6)],
            fill="#e8e8eb",
            width=1,
        )
        draw.text(
            (meta_x_center - cw // 2, my + 4), count, fill="#0a0a0b", font=f_meta_count
        )

    # ---- footer summary band ----
    band_y = H - 90
    draw.line([(PAD_X, band_y), (W - PAD_X, band_y)], fill="#e8e8eb", width=1)
    f_summary = load_font(26, italic=True)
    summary = "10 papers accepted · 5 top venues · 30+ under active review"
    sw = text_w(draw, summary, f_summary)
    draw.text(((W - sw) // 2, band_y + 26), summary, fill="#6b6b72", font=f_summary)

    canvas.convert("RGB").save(OUT, "PNG", optimize=True)
    print(f"  + {OUT}  ({W}x{H})")


if __name__ == "__main__":
    main()
