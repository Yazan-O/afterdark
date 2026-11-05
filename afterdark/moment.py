import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

W, H = 640, 480
VIEWS = ("events", "backing", "camera", "painting")
RASTER = (240, 320)


class MomentError(ValueError):
    pass


def png_bytes(image):
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


def crop_box(x, y, r):
    r = max(8, int(r))
    x, y = int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))
    return max(0, x - r), max(0, y - r), min(W, x + r), min(H, y + r)


def brighten(rgb):
    return (255 * (rgb.astype(np.float32) / 255) ** 0.45).astype(np.uint8)


class Moment:
    def __init__(self, root):
        self.root = Path(root)
        path = self.root / "meta.json"
        if not path.exists():
            raise MomentError(f"No moment at {self.root.name}")
        self.meta = json.loads(path.read_text(encoding="utf-8"))
        self.id = self.root.name
        self.n_paint = int(self.meta["n_paint"])
        self.seconds = float(self.meta["seconds"])
        self.paint_fps = int(self.meta["paint_fps"])
        self.bin_fps = int(self.meta["bin_fps"])
        self.counts = np.asarray(self.meta["counts"], np.int64)
        self.on = np.asarray(self.meta["on"], np.int64)
        film = self.root / "events_film.bin"
        if film.exists() and "film_index" in self.meta:
            self.event_path, self.event_index = film, np.asarray(self.meta["film_index"], np.int64)
        else:
            self.event_path, self.event_index = self.root / "events.bin", np.asarray(self.meta["index"], np.int64)
        self._events = None
        self.people = self._load_people()

    def _load_people(self):
        path = self.root / "people.json"
        if not path.exists():
            path = Path(__file__).resolve().parents[1] / "web" / "moments" / self.id / "people.json"
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        if "FIXTURE" in str(data.get("note", "")).upper():
            return {}
        return {int(f["k"]): f["people"] for f in data.get("frames", []) if f.get("people")}

    def check_k(self, k):
        k = int(k)
        if not 0 <= k < self.n_paint:
            raise MomentError(f"k must be 0 to {self.n_paint - 1}")
        return k

    def t_ms(self, k):
        return k * 1000 // self.paint_fps

    def painted(self):
        folder = self.root / "paint"
        if not folder.is_dir():
            return []
        return sorted(int(p.stem) for p in folder.glob("*.webp") if p.stem.isdigit() and int(p.stem) < self.n_paint)

    def words(self):
        if self._events is None:
            self._events = np.memmap(self.event_path, dtype="<u4", mode="r")
        return self._events

    def events(self, k, window_ms):
        window_ms = float(np.clip(window_ms, 1, 3000))
        centre = self.t_ms(k) / 1000
        lo, hi = centre - window_ms / 2000, centre + window_ms / 2000
        words = self.words()
        xs, ys, ps, weights = [], [], [], []
        bin_s = 1 / self.bin_fps
        first = max(0, int(lo / bin_s))
        last = min(len(self.counts) - 1, int(np.ceil(hi / bin_s)) - 1)
        for b in range(first, last + 1):
            chunk = np.asarray(words[self.event_index[b]:self.event_index[b + 1]])
            if not len(chunk):
                continue
            t = (b + ((chunk >> 20) & 4095) / 4095) * bin_s
            chunk = chunk[(t >= lo) & (t < hi)]
            if not len(chunk):
                continue
            stored = int(self.event_index[b + 1] - self.event_index[b])
            xs.append((chunk & 1023).astype(np.int32))
            ys.append(((chunk >> 10) & 511).astype(np.int32))
            ps.append(((chunk >> 19) & 1).astype(np.int8))
            weights.append(np.full(len(chunk), self.counts[b] / stored, np.float32))
        if not xs:
            empty = np.zeros(0, np.int32)
            return empty, empty, np.zeros(0, np.int8), np.zeros(0, np.float32)
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(ps), np.concatenate(weights)

    def event_map(self, k, window_ms):
        x, y, _, w = self.events(k, window_ms)
        grid = np.zeros((H, W), np.float32)
        keep = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        np.add.at(grid, (y[keep], x[keep]), w[keep])
        return grid

    def render_events(self, k, window_ms):
        grid = self.event_map(k, window_ms)
        top = max(float(np.percentile(grid, 99.5)), 1.0)
        v = 255 * np.clip(np.log1p(grid) / np.log1p(top), 0, 1)
        return Image.fromarray(v.astype(np.uint8)).convert("RGB")

    def _frame(self, folder, index):
        base = self.root / folder / f"{index:04d}"
        for ext in (".png", ".webp"):
            if base.with_suffix(ext).exists():
                return Image.open(base.with_suffix(ext))
        raise MomentError(f"No {folder} frame {index} in {self.id}")

    def backing(self, k):
        image = self._frame("backing", k).convert("L")
        if image.size != (W, H):
            image = image.resize((W, H), Image.BILINEAR)
        return np.asarray(image, np.float32)

    def camera_index(self, k):
        offsets = self.meta.get("camera_offsets_us")
        n_files = len(list((self.root / "cam").glob("*")))
        if offsets and n_files >= len(offsets) > self.n_paint:
            return int(np.argmin(np.abs(np.asarray(offsets, np.int64) - self.t_ms(k) * 1000)))
        return k

    def camera_raw(self, k):
        image = self._frame("cam", self.camera_index(k)).convert("RGB")
        if image.size != (W, H):
            image = image.resize((W, H), Image.BILINEAR)
        return np.asarray(image)

    def camera(self, k):
        return Image.fromarray(brighten(self.camera_raw(k)))

    def painting_path(self, k, version=None):
        folder = self.root / "paint"
        if version:
            path = folder / "repaint" / f"{k:04d}.{version}.webp"
        else:
            path = folder / f"{k:04d}.webp"
        if not path.exists():
            near = self.painted()
            hint = f"; painted frames include {near[:12]}" if near else ""
            raise MomentError(f"No painting at k={k}{hint}")
        return path

    def painting(self, k, version=None):
        return Image.open(self.painting_path(k, version)).convert("RGB").resize((W, H), Image.LANCZOS)

    def ev10_path(self, k):
        path = self.root / "ev10" / f"{k:04d}.png"
        if not path.exists():
            raise MomentError(f"No painter input at k={k}")
        return path

    def view_image(self, k, view, window_ms=100, version=None):
        if view == "events":
            return self.render_events(k, window_ms)
        if view == "backing":
            return Image.fromarray(self.backing(k).astype(np.uint8)).convert("RGB")
        if view == "camera":
            return self.camera(k)
        if view == "painting":
            return self.painting(k, version)
        raise MomentError(f"view must be one of {', '.join(VIEWS)}")

    def contact_sheet(self):
        seconds = max(1, int(self.seconds))
        cols, tile = 3, (212, 159)
        rows = int(np.ceil(seconds / cols))
        sheet = Image.new("RGB", (cols * tile[0], rows * tile[1]))
        draw = ImageDraw.Draw(sheet)
        for s in range(seconds):
            k = min(s * self.paint_fps + self.paint_fps // 2, self.n_paint - 1)
            x, y = (s % cols) * tile[0], (s // cols) * tile[1]
            sheet.paste(Image.fromarray(self.backing(k).astype(np.uint8)).convert("RGB").resize(tile), (x, y))
            draw.text((x + 4, y + 3), f"k={k}", fill=(255, 220, 90))
        return sheet

    def pair_sheet(self):
        seconds = max(1, int(self.seconds))
        tile = (213, 160)
        rows = int(np.ceil(seconds / 3))
        sheet = Image.new("RGB", (6 * tile[0], rows * tile[1]))
        draw = ImageDraw.Draw(sheet)
        for s in range(seconds):
            k = min(s * self.paint_fps + self.paint_fps // 2, self.n_paint - 1)
            x, y = (s % 3) * 2 * tile[0], (s // 3) * tile[1]
            sheet.paste(self.camera(k).resize(tile, Image.LANCZOS), (x, y))
            try:
                sheet.paste(self.painting(k).resize(tile, Image.LANCZOS), (x + tile[0], y))
            except MomentError:
                draw.text((x + tile[0] + 4, y + 20), "no painting", fill=(200, 200, 200))
            draw.text((x + 4, y + 3), f"k={k}", fill=(255, 220, 90))
        return sheet


def activity_grid(backing, cols=8, rows=6):
    out = []
    for r in range(rows):
        row = ""
        for c in range(cols):
            cell = backing[r * H // rows:(r + 1) * H // rows, c * W // cols:(c + 1) * W // cols]
            row += str(min(9, int(cell.mean() / 25.6)))
        out.append(row)
    return out


def people_mask(person):
    raster = np.zeros(160 * 120, bool)
    pos, value = 0, False
    for run in person.get("mask_rle", []):
        raster[pos:pos + int(run)] = value
        pos += int(run)
        value = not value
    return raster.reshape(120, 160)
