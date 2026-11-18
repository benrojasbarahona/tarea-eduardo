"""Agrega el sufijo -1 al código impreso en cada imagen de producto.

Uso:  python3 scripts/agregar_sufijo.py 16-11-2026 [índices...]
Lee  <fecha>/input-<fecha>/ y escribe <fecha>/output-<fecha>/CodigoProducto-1.jpg,
siempre de 800x800 exactos (la imagen se escala y se centra sobre fondo blanco).
La ubicación del código en cada imagen está en scripts/boxes-<fecha>.json.
"""
import os, sys, json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATE = sys.argv[1] if len(sys.argv) > 1 else '17-11-2026'
SRC = os.path.join(ROOT, DATE, f'input-{DATE}')
DST = os.path.join(ROOT, DATE, f'output-{DATE}')
CHK = os.path.join(HERE, 'check', DATE)  # recortes de revisión (no se versionan)
os.makedirs(DST, exist_ok=True); os.makedirs(CHK, exist_ok=True)
SIZE = 800  # tamaño final exacto (cuadrado)

BOXES = {int(k): v for k, v in json.load(open(os.path.join(HERE, f'boxes-{DATE}.json'))).items()}

SANS = '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
INTER = '/usr/share/fonts/opentype/inter/Inter-Regular.otf'
SERIF = '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf'

# Ajustes manuales por fecha
FONT, DONOR, LABEL_FIX, SHIFT = {}, {}, {}, {}
STEM_HYPHEN = False  # True: el guion se dibuja con el grosor del trazo del '1' de la imagen
if DATE == '17-11-2026':
    BOXES[48]['code'] = 'MAAS387'
    BOXES[20]['tight'] = [737, 1425, 1008, 1462]
    BOXES[38]['tight'] = [39, 1446, 491, 1496]
    FONT = {46: INTER, 47: INTER, 48: INTER, 21: SERIF}
    # códigos sin '1': se toma el '1' de una imagen gemela con la misma tipografía
    DONOR = {13: 12, 14: 15, 19: 17, 26: 27, 31: 32, 42: 43, 45: 44, 49: 11}
    # Kit Pistola (26) dice 'Cóg.': se reemplaza por 'Cód.' de la imagen gemela 27
    LABEL_FIX = {26: (27, (348, 428, 404, 455), (354, 427))}
elif DATE == '16-11-2026':
    BOXES[2]['tight'] = [5, 225, 70, 232]
    BOXES[21]['tight'] = [5, 220, 71, 228]
    BOXES[24]['tight'] = [174, 231, 231, 237]
    # Basurero 65 lt: el código toca el basurero negro; se baja el bloque de texto 5 px
    SHIFT = {6: [((4, 225, 66, 233), (0, 5)), ((4, 215, 24, 223), (0, 5))]}  # de abajo hacia arriba
    BOXES[6]['tight'] = [5, 231, 64, 237]
    DONOR = {0: 42, 2: 1, 3: 4, 6: 22, 8: 11, 14: 13, 15: 12, 17: 12, 19: 20, 28: 31, 29: 31,
             35: 36, 39: 44, 40: 44, 41: 44, 43: 44, 46: 4, 47: 4}
    STEM_HYPHEN = True
SS = 8  # supersampling for font rendering


def load(f):
    im = Image.open(os.path.join(SRC, f))
    im = im.convert('RGBA')
    bg = Image.new('RGBA', im.size, (255, 255, 255, 255)); bg.alpha_composite(im)
    return bg.convert('RGB')


def lum(a):
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def analyse(a, tb, code):
    x0, y0, x1, y1 = tb
    reg = a[y0:y1, x0:x1]; L = lum(reg)
    # local background: brightest pixels around the box
    ring = a[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4]
    bgl = np.percentile(lum(ring), 90)
    inkl = np.percentile(L, 2)
    thr = (bgl + inkl) / 2
    dark = L <= inkl + 0.2 * (bgl - inkl)
    color = reg[dark].mean(axis=0)
    comps = label((L < thr))
    segs = []
    for (cy0, cx0, cy1, cx1, m) in comps:
        segs.append((x0 + cx0, x0 + cx1, y0 + cy0, m, cy0, cx0))
    segs.sort(key=lambda t: t[0])
    chars = code.replace(' ', '')
    ok = len(segs) == len(chars)
    return dict(bgl=bgl, inkl=inkl, color=color, segs=segs, ok=ok, chars=chars)


def label(mask):
    """8-connected components; returns list of (y0,x0,y1,x1,submask) sorted by x, ignoring specks."""
    H, W = mask.shape; lab = np.zeros((H, W), int); n = 0; out = []
    for y in range(H):
        for x in range(W):
            if mask[y, x] and not lab[y, x]:
                n += 1; st = [(y, x)]; lab[y, x] = n; pts = []
                while st:
                    cy, cx = st.pop(); pts.append((cy, cx))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not lab[ny, nx]:
                                lab[ny, nx] = n; st.append((ny, nx))
                if len(pts) < 3: continue
                ys = [p[0] for p in pts]; xs = [p[1] for p in pts]
                out.append((min(ys), min(xs), max(ys) + 1, max(xs) + 1, lab == n))
    return out


def glyph_one(a, tb, info):
    """alpha mask of the last '1' glyph of the code line, cropped tightly in x, rows tb[1]-1..tb[3]+1"""
    chars = info['chars']
    if '1' not in chars: return None
    k = len(chars) - 1 - chars[::-1].index('1')
    if not info['ok']:
        # glyphs merged somewhere (e.g. touching letters); map from the right end instead
        k = len(info['segs']) - (len(chars) - k)
        if k < 0: return None
        c = info['segs'][k]; cm = c[3]
        gh = cm.shape[0] if False else (np.nonzero(cm.any(axis=1))[0])
        gw = c[1] - c[0]; capH = tb[3] - tb[1]
        if not (gw < 0.6 * capH and len(gh) >= 0.9 * capH): return None
    sx0, sx1, _, cm, _, _ = info['segs'][k]
    y0, y1 = tb[1] - 1, tb[3] + 1
    patch = lum(a[y0:y1, sx0 - 1:sx1 + 1])
    al = np.clip((info['bgl'] - patch) / (info['bgl'] - info['inkl']), 0, 1)
    # keep only this glyph's pixels (component mask dilated by 1px for the anti-aliased rim)
    full = np.zeros((a.shape[0], a.shape[1]), bool)
    full[tb[1]:tb[3], tb[0]:tb[2]] = cm
    sub = full[y0:y1, sx0 - 1:sx1 + 1]
    d = sub.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            d |= np.roll(np.roll(sub, dy, 0), dx, 1)
    al = al * d
    return dict(alpha=al, top=y0 - tb[1], left_pad=1)  # ink starts 1px inside patch


def ink_cols_rows(img_alpha):
    m = img_alpha > 0.5 * 255
    xs = np.nonzero(m.any(axis=0))[0]; ys = np.nonzero(m.any(axis=1))[0]
    return xs, ys


def render(text, font, W, H, ox, oy):
    im = Image.new('L', (W, H), 0)
    ImageDraw.Draw(im).text((ox, oy), text, font=font, fill=255)
    return np.asarray(im).astype(float)


def font_layout(code, capH, img_w, fontpath):
    """Return supersampled render of '-1' positioned relative to the code ink right edge/baseline."""
    f100 = ImageFont.truetype(fontpath, 100 * SS)
    W = int(f100.getlength(code + '-1') + 200 * SS); H = 300 * SS
    r = render(code, f100, W, H, 50 * SS, 50 * SS)
    xs, ys = ink_cols_rows(r)
    h100 = (ys[-1] + 1 - ys[0])
    size = 100 * SS * capH * SS / h100
    f = ImageFont.truetype(fontpath, max(1, int(round(size))))
    W = int(f.getlength(code + '-1') + 100 * SS); H = int(size * 2 + 40 * SS)
    rc = render(code, f, W, H, 20 * SS, 20 * SS)
    rh = render(code + '-', f, W, H, 20 * SS, 20 * SS)
    r1 = render(code + '-1', f, W, H, 20 * SS, 20 * SS)
    xs, ys = ink_cols_rows(rc)
    cl, cr, base = xs[0], xs[-1] + 1, ys[-1] + 1
    sx = (img_w * SS) / (cr - cl)  # horizontal scale to match the image's letter width/spacing
    hyph = np.clip(rh - rc, 0, 255)
    one = np.clip(r1 - rh, 0, 255)
    return dict(cr=cr, base=base, sx=sx, hyph=hyph, one=one, capH_ss=ys[-1] + 1 - ys[0])


def place_layer(layer, cr, base, sx, tx, tbase, rows_above):
    """Downsample a supersampled layer so that render point (cr, base) lands on image point (tx, tbase).
    Returns (alpha array in [0,1], x_off, y_off)."""
    xs, ys = ink_cols_rows(layer)
    if len(xs) == 0: return None
    # crop from cr (exact) to beyond ink, vertically aligned to baseline in multiples of SS
    top = base - rows_above * SS
    bot = base + 4 * SS
    x_end = xs[-1] + 2 * SS
    crop = layer[top:bot, cr:x_end]
    out_w = max(1, int(round(crop.shape[1] * sx / SS)))
    out_h = crop.shape[0] // SS
    im = Image.fromarray(crop.astype(np.uint8)).resize((out_w, out_h), Image.BOX)
    return np.asarray(im).astype(float) / 255, tx, tbase - rows_above


def composite(a, alpha, x, y, color):
    h, w = alpha.shape
    H, W = a.shape[:2]
    xa, ya = max(0, x), max(0, y)
    xb, yb = min(W, x + w), min(H, y + h)
    al = alpha[ya - y:yb - y, xa - x:xb - x][..., None]
    a[ya:yb, xa:xb] = a[ya:yb, xa:xb] * (1 - al) + color * al


def stem_hyphen(hy, one_alpha):
    """Redraw the hyphen as a bar as thick as the stem of the image's own '1' (keeps the
    font's hyphen length and vertical centre), so it matches the weight of tiny/bold text."""
    al, x, y = hy
    rows = np.nonzero((al > 0.25).any(axis=1))[0]; cols = np.nonzero((al > 0.25).any(axis=0))[0]
    if len(rows) == 0: return hy
    w_al = al.sum(axis=0); x0 = cols[0]; x1 = cols[-1] + 1
    cy = (al.sum(axis=1) * np.arange(al.shape[0])).sum() / al.sum() + 0.5
    h = one_alpha.shape[0]
    mid = one_alpha[int(h * 0.35):int(h * 0.85)]
    t = float(np.median(mid.sum(axis=1)))  # stem width in px
    t = max(t, 1.0)
    # length: keep the font's coverage-weighted width
    L = al.sum() / max(al.sum(axis=0).max(), 1e-6)
    L = max(L, x1 - x0 - 0.5)
    k = 16
    H, W = al.shape
    big = Image.new('L', (W * k, H * k), 0)
    cx = (x0 + x1) / 2
    ImageDraw.Draw(big).rectangle([round((cx - L / 2) * k), round((cy - t / 2) * k),
                                   round((cx + L / 2) * k) - 1, round((cy + t / 2) * k) - 1], fill=255)
    new = np.asarray(big.resize((W, H), Image.BOX)).astype(float) / 255
    return new, x, y


def fix_label(i, im):
    """Corrige una etiqueta mal escrita copiando la correcta desde una imagen gemela."""
    if i in LABEL_FIX:
        j, box, at = LABEL_FIX[i]
        im.paste(load(BOXES[j]['file']).crop(box), at)
    for box, (dx, dy) in SHIFT.get(i, []):
        part = im.crop(box)
        im.paste((255, 255, 255), box)
        im.paste(part, (box[0] + dx, box[1] + dy))
    return im


def process(i):
    v = BOXES[i]; f = v['file']; code = v['code']; tb = v['tight']
    im = fix_label(i, load(f)); a = np.asarray(im).astype(float); h, w = a.shape[:2]
    info = analyse(a, tb, code)
    capH = tb[3] - tb[1]
    fontpath = FONT.get(i, SANS)
    lay = font_layout(code, capH, tb[2] - tb[0], fontpath)
    rows_above = capH + 4
    hy = place_layer(lay['hyph'], lay['cr'], lay['base'], lay['sx'], tb[2], tb[3], rows_above)
    # position of the '1' ink-left, as predicted by font metrics
    one_r = lay['one']; oxs, oys = ink_cols_rows(one_r)
    one_left = tb[2] + (oxs[0] - lay['cr']) * lay['sx'] / SS
    one_src = 'copied'
    g = glyph_one(a, tb, info)
    if g is None and i in DONOR:
        d = BOXES[DONOR[i]]; da = np.asarray(load(d['file'])).astype(float)
        dinfo = analyse(da, d['tight'], d['code'])
        g = glyph_one(da, d['tight'], dinfo)
        if g is not None:
            dcap = d['tight'][3] - d['tight'][1]
            if dcap != capH:
                s = capH / dcap
                al = Image.fromarray((g['alpha'] * 255).astype(np.uint8))
                al = al.resize((max(1, round(al.width * s)), max(1, round(al.height * s))), Image.LANCZOS)
                g = dict(alpha=np.asarray(al).astype(float) / 255, top=round(g['top'] * s), left_pad=round(g['left_pad'] * s))
            one_src = f'donor {DONOR[i]}'
    if g is not None:
        one_alpha = g['alpha']; ox = int(round(one_left)) - g['left_pad']; oy = tb[1] + g['top']
        one_right = ox + one_alpha.shape[1] - g['left_pad']
    else:
        one_src = 'font'
        pl = place_layer(one_r, lay['cr'], lay['base'], lay['sx'], tb[2], tb[3], rows_above)
        one_alpha, ox, oy = pl
        cols = np.nonzero((one_alpha > 0.5).any(axis=0))[0]
        one_right = ox + cols[-1] + 1
    # does it fit? keep a small right margin; if not, extend the canvas to the right with white
    m = max(4, round(0.02 * max(w, h)))
    pad = max(0, one_right - (w - m))
    if pad:
        big = np.full((h, w + pad, 3), 255.0); big[:, :w] = a; a = big
    color = info['color']
    if STEM_HYPHEN and g is not None:
        hy = stem_hyphen(hy, g['alpha'])
    composite(a, hy[0], hy[1], hy[2], color)
    composite(a, one_alpha, ox, oy, color)
    img = Image.fromarray(np.clip(np.round(a), 0, 255).astype(np.uint8))
    # final: exactly SIZE x SIZE, image scaled to fit and centred on white
    W, H = img.size
    s = SIZE / max(W, H)
    nw, nh = min(SIZE, round(W * s)), min(SIZE, round(H * s))
    img = img.resize((nw, nh), Image.LANCZOS)
    out = Image.new('RGB', (SIZE, SIZE), (255, 255, 255))
    offx, offy = (SIZE - nw) // 2, (SIZE - nh) // 2
    out.paste(img, (offx, offy))
    name = code.replace(' ', '') + '-1.jpg'
    path = os.path.join(DST, name)
    q = 95
    while True:
        out.save(path, 'JPEG', quality=q, subsampling=0, optimize=True)
        if os.path.getsize(path) <= 3 * 1024 * 1024 or q <= 60: break
        q -= 5
    # check crop (in output coords)
    cx0 = offx + int((tb[0] - capH * 2) * s); cy0 = offy + int((tb[1] - capH * 2.2) * s)
    cx1 = offx + int((one_right + capH * 1.5) * s); cy1 = offy + int((tb[3] + capH * 1.2) * s)
    out.crop((max(0, cx0), max(0, cy0), min(SIZE, cx1), min(SIZE, cy1))).save(os.path.join(CHK, f'{i:02d}.png'))
    return dict(i=i, file=f, out=name, size=out.size, kb=os.path.getsize(path) // 1024, seg_ok=info['ok'],
                nseg=len(info['segs']), one=one_src, pad=pad, color=[int(c) for c in color], sx=round(lay['sx'], 3))


if __name__ == '__main__':
    ids = [int(x) for x in sys.argv[2:]] or sorted(BOXES)
    for i in ids:
        print(process(i))
