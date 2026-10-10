"""Abstract textless cover: JAX tensor blocks feeding a Gemma sparkle built on a 4-bit grid."""
import math, random, sys
from PIL import Image, ImageDraw, ImageFilter

S = 2
W, H = 1376 * S, 578 * S
BG = (20, 22, 30)
random.seed(7)

img = Image.new("RGB", (W, H), BG)

# soft radial glow behind the centre
glow = Image.new("RGB", (W, H), BG)
g = ImageDraw.Draw(glow)
cx, cy = W // 2, H // 2
for r in range(520 * S, 0, -8 * S):
    t = 1 - r / (520 * S)
    c = tuple(int(BG[i] + (v - BG[i]) * t * 0.35) for i, v in enumerate((70, 80, 190)))
    g.ellipse([cx - r * 1.6, cy - r, cx + r * 1.6, cy + r], fill=c)
img = Image.blend(img, glow.filter(ImageFilter.GaussianBlur(40 * S)), 1.0)
d = ImageDraw.Draw(img, "RGBA")

# faint circuit traces at the edges
for _ in range(70):
    x = random.choice([random.randint(0, 260 * S), random.randint(W - 260 * S, W)])
    y = random.randint(30 * S, H - 30 * S)
    pts = [(x, y)]
    for _ in range(3):
        x += random.choice([-1, 1]) * random.randint(20, 70) * S
        pts.append((x, y))
        y += random.choice([-1, 1]) * random.randint(10, 40) * S
        pts.append((x, y))
    d.line(pts, fill=(90, 110, 160, 45), width=2 * S)
    d.ellipse([x - 4 * S, y - 4 * S, x + 4 * S, y + 4 * S], fill=(110, 130, 190, 70))

def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

# Gemma-style four-point sparkle made of grid cells
BLUE, VIOLET, PINK = (66, 133, 244), (140, 90, 230), (200, 110, 220)
R = 230 * S
cell = 13 * S
def inside(x, y):
    # astroid-like four-point star: |x|^p + |y|^p <= R^p with p < 1
    p = 0.55
    return (abs(x) / R) ** p + (abs(y) / R) ** p <= 1
for gy in range(-R, R, cell):
    for gx in range(-R, R, cell):
        mx, my = gx + cell / 2, gy + cell / 2
        if not inside(mx, my):
            continue
        dist = math.hypot(mx, my) / R
        ang = (math.atan2(my, mx) + math.pi) / (2 * math.pi)
        col = lerp(lerp(BLUE, VIOLET, ang), PINK, max(0, 0.6 - dist) * 0.6)
        a = int(255 * (1 - dist * 0.55))
        pad = S * (1 if random.random() < 0.85 else 3)
        d.rectangle([cx + gx + pad, cy + gy + pad, cx + gx + cell - pad, cy + gy + cell - pad], fill=col + (a,))
# bright core
core = Image.new("L", (W, H), 0)
ImageDraw.Draw(core).ellipse([cx - 40 * S, cy - 40 * S, cx + 40 * S, cy + 40 * S], fill=255)
core = core.filter(ImageFilter.GaussianBlur(30 * S))
img.paste(Image.new("RGB", (W, H), (235, 230, 255)), (0, 0), core.point(lambda v: int(v * 0.8)))
d = ImageDraw.Draw(img, "RGBA")

# JAX-coloured isometric blocks, stacked like tensors, on both sides
TEAL, GREEN, PURPLE = (0, 166, 166), (106, 176, 76), (142, 68, 173)
def cube(x, y, s, col, alpha=230):
    top = [(x, y - s / 2), (x + s, y), (x, y + s / 2), (x - s, y)]
    left = [(x - s, y), (x, y + s / 2), (x, y + s * 1.5), (x - s, y + s)]
    right = [(x, y + s / 2), (x + s, y), (x + s, y + s), (x, y + s * 1.5)]
    d.polygon(left, fill=lerp(col, (0, 0, 0), 0.35) + (alpha,))
    d.polygon(right, fill=lerp(col, (0, 0, 0), 0.15) + (alpha,))
    d.polygon(top, fill=lerp(col, (255, 255, 255), 0.15) + (alpha,))
    for poly in (top, left, right):
        d.line(poly + [poly[0]], fill=(255, 255, 255, 40), width=S)

stacks = [(330, 250, TEAL), (250, 340, GREEN), (410, 350, PURPLE),
          (W // S - 330, 250, PURPLE), (W // S - 410, 350, TEAL), (W // S - 250, 340, GREEN)]
anchors = []
for x, y, col in stacks:
    for k in range(3):
        cube(x * S, (y - k * 34) * S, 34 * S, col, 150 + 35 * k)
    anchors.append((x * S, (y - 2 * 34) * S, col))

# data lines from blocks into the sparkle
for x, y, col in anchors:
    for j in range(4):
        off = (j - 1.5) * 9 * S
        pts = []
        for i in range(41):
            t = i / 40
            px = x + (cx - x) * t
            py = y + off + (cy - y - off) * (t ** 1.6) + math.sin(t * math.pi) * -30 * S
            pts.append((px, py))
        for i in range(40):
            t = i / 40
            a = int(150 * math.sin(t * math.pi))
            d.line([pts[i], pts[i + 1]], fill=col + (a,), width=2 * S)
        # a few bright packets
        for _ in range(2):
            i = random.randint(8, 32)
            px, py = pts[i]
            d.ellipse([px - 4 * S, py - 4 * S, px + 4 * S, py + 4 * S], fill=lerp(col, (255, 255, 255), 0.5) + (220,))

img = img.resize((W // S, H // S), Image.LANCZOS)
img.save(sys.argv[1], quality=92)
print("wrote", sys.argv[1], img.size)
