#!/usr/bin/env python3
"""Turn a local raster avatar into terminal-style SVG artwork.

The source image is a local-only build input. This script emits real vector
paths, plain ASCII text, or a low-resolution color-cell grid (never an <image>
element or base64 data), and can insert that artwork into the marked portrait
area in the profile SVGs.

Dependencies: Pillow and NumPy.
"""

from __future__ import annotations

import argparse
import html
import itertools
import math
import re
from collections import defaultdict, deque
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageFilter, ImageOps
except ImportError as error:  # pragma: no cover - dependency hint for local use
    raise SystemExit(
        "trace-image-to-svg.py requires Pillow and NumPy; "
        "install them with: python3 -m pip install Pillow numpy"
    ) from error


Point = tuple[int, int]
FloatPoint = tuple[float, float]
AsciiCell = tuple[str, str]
BlockCell = tuple[int, int, str]
CropBox = tuple[float, float, float, float]
AVATAR_START = "<!-- avatar:start -->"
AVATAR_END = "<!-- avatar:end -->"


def polygon_area(points: list[FloatPoint]) -> float:
    """Return the unsigned area of a closed polygon."""
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )
    ) / 2


def point_line_distance(point: FloatPoint, start: FloatPoint, end: FloatPoint) -> float:
    if start == end:
        return math.dist(point, start)
    x, y = point
    x1, y1 = start
    x2, y2 = end
    numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    return numerator / math.hypot(y2 - y1, x2 - x1)


def simplify_open(points: list[FloatPoint], tolerance: float) -> list[FloatPoint]:
    """Ramer-Douglas-Peucker simplification for one open polyline."""
    if len(points) <= 2:
        return points
    distances = [
        point_line_distance(point, points[0], points[-1]) for point in points[1:-1]
    ]
    if not distances:
        return [points[0], points[-1]]
    maximum = max(distances)
    if maximum <= tolerance:
        return [points[0], points[-1]]
    split = distances.index(maximum) + 1
    return simplify_open(points[: split + 1], tolerance)[:-1] + simplify_open(
        points[split:], tolerance
    )


def simplify_closed(points: list[FloatPoint], tolerance: float) -> list[FloatPoint]:
    """Simplify a ring without choosing an unstable adjacent start/end pair."""
    if len(points) < 5:
        return points
    anchor = min(range(len(points)), key=lambda index: (points[index][1], points[index][0]))
    opposite = max(
        range(len(points)), key=lambda index: math.dist(points[anchor], points[index])
    )
    if anchor > opposite:
        anchor, opposite = opposite, anchor
    first = simplify_open(points[anchor : opposite + 1], tolerance)
    second = simplify_open(points[opposite:] + points[: anchor + 1], tolerance)
    simplified = first[:-1] + second[:-1]
    return simplified if len(simplified) >= 3 else points


def mask_edges(mask: np.ndarray) -> dict[Point, list[Point]]:
    """Build clockwise pixel-boundary edges with the filled area on the right."""
    height, width = mask.shape
    edges: dict[Point, list[Point]] = defaultdict(list)
    for y, x in np.argwhere(mask):
        x = int(x)
        y = int(y)
        if y == 0 or not mask[y - 1, x]:
            edges[(x, y)].append((x + 1, y))
        if x == width - 1 or not mask[y, x + 1]:
            edges[(x + 1, y)].append((x + 1, y + 1))
        if y == height - 1 or not mask[y + 1, x]:
            edges[(x + 1, y + 1)].append((x, y + 1))
        if x == 0 or not mask[y, x - 1]:
            edges[(x, y + 1)].append((x, y))
    return edges


def direction(start: Point, end: Point) -> int:
    vector = (end[0] - start[0], end[1] - start[1])
    return {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}[vector]


def choose_next(previous: Point, current: Point, candidates: list[Point]) -> Point:
    """Resolve diagonal contacts by keeping the filled region on the right."""
    incoming = direction(previous, current)
    preferred = ((incoming + 1) % 4, incoming, (incoming - 1) % 4, (incoming + 2) % 4)
    by_direction = {direction(current, candidate): candidate for candidate in candidates}
    return next(by_direction[item] for item in preferred if item in by_direction)


def trace_mask(mask: np.ndarray, tolerance: float, min_area: float) -> list[list[FloatPoint]]:
    edges = mask_edges(mask)
    unused = {(start, end) for start, ends in edges.items() for end in ends}
    rings: list[list[FloatPoint]] = []

    while unused:
        start_edge = min(unused)
        start, current = start_edge
        previous = start
        ring: list[FloatPoint] = [(float(start[0]), float(start[1]))]
        unused.remove(start_edge)

        while current != start:
            ring.append((float(current[0]), float(current[1])))
            candidates = [end for end in edges[current] if (current, end) in unused]
            if not candidates:
                raise RuntimeError(f"open contour encountered at {current}")
            following = choose_next(previous, current, candidates)
            unused.remove((current, following))
            previous, current = current, following

        if polygon_area(ring) >= min_area:
            rings.append(simplify_closed(ring, tolerance))
    return rings


def format_number(value: float) -> str:
    rounded = round(value, 2)
    if rounded == 0:
        return "0"
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


def rings_to_path(rings: list[list[FloatPoint]]) -> str:
    parts: list[str] = []
    for ring in rings:
        coordinates = " ".join(f"{format_number(x)} {format_number(y)}" for x, y in ring)
        parts.append(f"M{coordinates}Z")
    return "".join(parts)


def palette_classes(palette: list[tuple[int, int, int]]) -> dict[int, str]:
    """Map source clusters to ordered theme tones instead of fixed colors."""
    luminances: list[tuple[float, int]] = []
    for index, (red, green, blue) in enumerate(palette):
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        luminances.append((luminance, index))
    return {
        index: f"avatar-tone-{rank}"
        for rank, (_, index) in enumerate(sorted(luminances))
    }


def vectorize(
    input_path: Path,
    size: int,
    colors: int,
    threshold: int,
    tolerance: float,
    min_area: float,
) -> tuple[list[str], list[tuple[int, int, int]]]:
    with Image.open(input_path) as source:
        source = ImageOps.exif_transpose(source).convert("RGB")
        width, height = source.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        source = source.crop((left, top, left + side, top + side))
        source = source.resize((size, size), Image.Resampling.LANCZOS)
        softened = source.filter(ImageFilter.GaussianBlur(radius=0.8))
        indexed = softened.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)

    labels = np.asarray(indexed, dtype=np.uint8)
    rgb = np.asarray(softened, dtype=np.int16)
    raw_palette = indexed.getpalette() or []
    palette = [tuple(raw_palette[index * 3 : index * 3 + 3]) for index in range(colors)]
    classes = palette_classes(palette)
    paths: list[str] = []

    # Paint large quantized regions first; later layers remain crisp vector shapes.
    for index in sorted(range(colors), key=lambda item: sum(palette[item])):
        rings = trace_mask(labels == index, tolerance, min_area)
        if not rings:
            continue
        path_data = rings_to_path(rings)
        paths.append(
            f'    <path class="{classes[index]}" d="{path_data}" '
            'fill-rule="evenodd" clip-rule="evenodd"/>'
        )

    # Small source-color masks preserve the blue ribbon and violet-pink eyes.
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    blue_mask = (blue - red > 14) & (blue - green > 10)
    blue_rings = trace_mask(blue_mask, tolerance * 0.7, min_area * 0.45)
    if blue_rings:
        paths.append(
            f'    <path class="avatar-blue" d="{rings_to_path(blue_rings)}" '
            'fill-rule="evenodd" clip-rule="evenodd"/>'
        )

    y_grid, x_grid = np.ogrid[:size, :size]
    eye_region = (
        (x_grid >= int(size * 0.16))
        & (x_grid <= int(size * 0.72))
        & (y_grid >= int(size * 0.43))
        & (y_grid <= int(size * 0.69))
    )
    magenta_mask = (red - green > 20) & (blue - green > 12) & eye_region
    magenta_rings = trace_mask(magenta_mask, tolerance * 0.62, min_area * 0.3)
    if magenta_rings:
        paths.append(
            f'    <path class="avatar-magenta" d="{rings_to_path(magenta_rings)}" '
            'fill-rule="evenodd" clip-rule="evenodd"/>'
        )

    # A thresholded ink pass restores eyes, lashes, hair strands, and terminal-like linework.
    grayscale = np.asarray(softened.convert("L"), dtype=np.uint8)
    ink_rings = trace_mask(grayscale < threshold, tolerance * 0.72, min_area * 0.55)
    if ink_rings:
        paths.append(
            f'    <path class="avatar-ink" d="{rings_to_path(ink_rings)}" '
            'fill-rule="evenodd" clip-rule="evenodd" opacity="0.86"/>'
        )
    return paths, palette


def make_ascii_rows(input_path: Path, columns: int, rows: int) -> list[list[AsciiCell]]:
    """Render sparse, direction-aware edges instead of a dense luminance ramp."""
    with Image.open(input_path) as source:
        source = ImageOps.exif_transpose(source).convert("RGB")
        width, height = source.size
        if width > height * 1.2:
            # The current source is a landscape illustration. This crop keeps
            # the central girl's ahoge, hair, face, shoulders, and ribbon while
            # dropping the sea, legs, and side mascots.
            source = source.crop(
                (
                    round(width * 0.15),
                    round(height * 0.024),
                    round(width * 0.617),
                    round(height * 0.697),
                )
            )
        else:
            side = min(width, height)
            left = (width - side) // 2
            top = (height - side) // 2
            source = source.crop((left, top, left + side, top + side))

        sampled = source.filter(ImageFilter.GaussianBlur(radius=1.05)).resize(
            (columns, rows), Image.Resampling.LANCZOS
        )
        color_grid = np.asarray(sampled, dtype=np.float32)
        gray_grid = np.asarray(ImageOps.autocontrast(sampled.convert("L"), cutoff=1), dtype=np.float32)

    y_grid, x_grid = np.mgrid[:rows, :columns]
    x_normalized = (x_grid + 0.5) / columns
    y_normalized = (y_grid + 0.5) / rows

    padded = np.pad(gray_grid, 1, mode="edge")
    gradient_x = (
        padded[:-2, 2:]
        + 2 * padded[1:-1, 2:]
        + padded[2:, 2:]
        - padded[:-2, :-2]
        - 2 * padded[1:-1, :-2]
        - padded[2:, :-2]
    )
    gradient_y = (
        padded[2:, :-2]
        + 2 * padded[2:, 1:-1]
        + padded[2:, 2:]
        - padded[:-2, :-2]
        - 2 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
    )
    edge_strength = np.hypot(gradient_x, gradient_y)

    red, green, blue = np.moveaxis(color_grid, 2, 0)
    # The blue sky is the only large backdrop in the crop. A warm/neutral color
    # gate plus a loose bust mask isolates the character without attempting a
    # brittle pixel-perfect cutout.
    foreground = (red > blue * 0.57) | (green > blue * 0.76)
    head_and_hair = (
        ((x_normalized - 0.5) / 0.51) ** 2
        + ((y_normalized - 0.4) / 0.45) ** 2
        <= 1
    )
    torso = (y_normalized >= 0.38) & (
        abs(x_normalized - 0.5) <= 0.28 + y_normalized * 0.25
    )
    silhouette = foreground & (head_and_hair | torso)
    candidates = edge_strength[silhouette]
    threshold = float(np.percentile(candidates, 66)) if candidates.size else math.inf

    output: list[list[AsciiCell]] = []
    for y in range(rows):
        output_row: list[AsciiCell] = []
        for x in range(columns):
            if not silhouette[y, x] or edge_strength[y, x] < threshold:
                output_row.append((" ", "ascii-blank"))
                continue

            tangent = (math.atan2(gradient_y[y, x], gradient_x[y, x]) + math.pi / 2) % math.pi
            if tangent < math.pi / 8 or tangent >= math.pi * 7 / 8:
                character = "-"
            elif tangent < math.pi * 3 / 8:
                character = "\\"
            elif tangent < math.pi * 5 / 8:
                character = "|"
            else:
                character = "/"

            cell_red, cell_green, cell_blue = color_grid[y, x]
            luminance = 0.2126 * cell_red + 0.7152 * cell_green + 0.0722 * cell_blue
            if cell_blue - cell_red > 14 and cell_blue - cell_green > 8:
                color_class = "ascii-blue"
            elif cell_red - cell_green > 20 and y_normalized[y, x] >= 0.48:
                color_class = "ascii-magenta"
            elif luminance < 92:
                color_class = "ascii-tone-0"
            elif luminance < 138:
                color_class = "ascii-tone-1"
            elif luminance < 182:
                color_class = "ascii-tone-2"
            elif luminance < 222:
                color_class = "ascii-tone-3"
            else:
                color_class = "ascii-tone-4"
            output_row.append((character, color_class))
        output.append(output_row)
    return output


def load_curated_ascii(input_path: Path) -> list[list[AsciiCell]]:
    """Load hand-curated ASCII and add restrained hair, face, and bow tones."""
    lines = input_path.read_text(encoding="utf-8").splitlines()
    if not lines or not any(line.strip() for line in lines):
        raise SystemExit(f"ASCII art file is empty: {input_path}")

    width = max(len(line) for line in lines)
    center = (width - 1) / 2
    output: list[list[AsciiCell]] = []
    for y, line in enumerate(lines):
        row: list[AsciiCell] = []
        for x, character in enumerate(line.ljust(width)):
            distance = abs(x - center)
            if character == " ":
                color_class = "ascii-blank"
            elif 19 <= y <= 24 and distance <= 10:
                color_class = "ascii-magenta"
            elif y <= 20:
                color_class = (
                    "ascii-tone-4"
                    if 8 <= y <= 16 and distance <= 10
                    else "ascii-blue"
                )
            else:
                color_class = "ascii-tone-2"
            row.append((character, color_class))
        output.append(row)
    return output


def crop_source(source: Image.Image, crop: CropBox | None) -> Image.Image:
    """Apply an optional normalized crop, otherwise return a centered square."""
    width, height = source.size
    if crop is not None:
        left, top, right, bottom = crop
        return source.crop(
            (
                round(width * left),
                round(height * top),
                round(width * right),
                round(height * bottom),
            )
        )
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return source.crop((left, top, left + side, top + side))


def parse_crop(value: str) -> CropBox:
    """Parse left,top,right,bottom as normalized coordinates."""
    try:
        crop = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("crop must contain four decimal numbers") from error
    if len(crop) != 4:
        raise argparse.ArgumentTypeError("crop must be left,top,right,bottom")
    left, top, right, bottom = crop
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise argparse.ArgumentTypeError("crop values must satisfy 0 <= left < right <= 1 and 0 <= top < bottom <= 1")
    return left, top, right, bottom


def make_block_cells(
    input_path: Path,
    columns: int,
    rows: int,
    crop: CropBox | None,
    palette_size: int,
) -> list[BlockCell]:
    """Turn the outlined head into a transparent, source-derived color grid."""
    with Image.open(input_path) as source:
        source = ImageOps.exif_transpose(source).convert("RGB")
        source = crop_source(source, crop)
        sampled = source.resize((columns, rows), Image.Resampling.NEAREST)
        colors = np.asarray(sampled, dtype=np.uint8)

        # The illustration has a strong dark contour around the character. At
        # 4x grid resolution, treat that contour as a wall and flood-fill from
        # the crop edges. Everything enclosed by the wall becomes foreground.
        mask_scale = 4
        mask_width = columns * mask_scale
        mask_height = rows * mask_scale
        mask_source = source.resize(
            (mask_width, mask_height), Image.Resampling.LANCZOS
        )
        mask_rgb = np.asarray(mask_source, dtype=np.float32)
        mask_red, mask_green, mask_blue = np.moveaxis(mask_rgb, 2, 0)
        mask_luminance = (
            0.2126 * mask_red + 0.7152 * mask_green + 0.0722 * mask_blue
        )

    barrier = mask_luminance < 82
    padded_barrier = np.pad(barrier, 1, mode="constant")
    closed_barrier = np.zeros_like(barrier)
    for offset_y in range(3):
        for offset_x in range(3):
            closed_barrier |= padded_barrier[
                offset_y : offset_y + mask_height,
                offset_x : offset_x + mask_width,
            ]

    background = np.zeros((mask_height, mask_width), dtype=bool)
    pending: deque[tuple[int, int]] = deque()
    for x in range(mask_width):
        pending.extend(((0, x), (mask_height - 1, x)))
    for y in range(mask_height):
        pending.extend(((y, 0), (y, mask_width - 1)))
    while pending:
        y, x = pending.popleft()
        if background[y, x] or closed_barrier[y, x]:
            continue
        background[y, x] = True
        if y > 0:
            pending.append((y - 1, x))
        if y + 1 < mask_height:
            pending.append((y + 1, x))
        if x > 0:
            pending.append((y, x - 1))
        if x + 1 < mask_width:
            pending.append((y, x + 1))

    foreground = ~background
    coverage = foreground.reshape(
        rows, mask_scale, columns, mask_scale
    ).mean(axis=(1, 3))
    y_grid, x_grid = np.mgrid[:rows, :columns]
    x_normalized = (x_grid + 0.5) / columns
    y_normalized = (y_grid + 0.5) / rows
    head_window = (
        ((x_normalized - 0.52) / 0.53) ** 2
        + ((y_normalized - 0.5) / 0.51) ** 2
        <= 1
    )
    # The red area above the black headband belongs to the illustration's
    # backdrop, not the hair. Follow the headband's shallow arch so the crop
    # removes that cap without clipping the side flower or hair below it.
    headband_top = 0.16 + 0.35 * (x_normalized - 0.5) ** 2
    background_cap = (
        (x_normalized > 0.14)
        & (x_normalized < 0.82)
        & (y_normalized < headband_top)
    )
    # End every part of the portrait on the same terminal row. The discarded
    # cells are stray collar/hair pixels below the otherwise flat baseline.
    bottom_trim = y_normalized >= 0.9
    visible = (coverage >= 0.28) & head_window & ~background_cap & ~bottom_trim

    # Keep the largest enclosed island plus sizeable accessories that touch it
    # diagonally at block resolution. This drops the mascot and patterned ears.
    remaining = {(int(y), int(x)) for y, x in np.argwhere(visible)}
    components: list[set[tuple[int, int]]] = []
    while remaining:
        component: set[tuple[int, int]] = set()
        pending = [remaining.pop()]
        while pending:
            current_y, current_x = pending.pop()
            component.add((current_y, current_x))
            for neighbor_y in range(max(0, current_y - 1), min(rows, current_y + 2)):
                for neighbor_x in range(max(0, current_x - 1), min(columns, current_x + 2)):
                    neighbor = (neighbor_y, neighbor_x)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        pending.append(neighbor)
        components.append(component)

    kept: set[tuple[int, int]] = max(components, key=len) if components else set()
    minimum_accessory_size = max(3, round(len(kept) * 0.025))
    for component in components:
        if len(component) >= minimum_accessory_size:
            kept |= component
    # Fill between the retained outer contour on each scanline. This restores
    # small internal islands such as the two eyes, lashes, and mouth without
    # bringing back objects outside the head silhouette.
    visible = np.zeros((rows, columns), dtype=bool)
    for y in range(rows):
        row_positions = [x for component_y, x in kept if component_y == y]
        if row_positions:
            visible[y, min(row_positions) : max(row_positions) + 1] = True
    visible &= head_window & ~background_cap & ~bottom_trim

    positions = np.argwhere(visible)
    if not len(positions):
        return []
    foreground_pixels = colors[visible]
    palette_source = Image.fromarray(
        foreground_pixels.reshape(1, len(foreground_pixels), 3), "RGB"
    )
    indexed = palette_source.quantize(
        colors=palette_size, method=Image.Quantize.MEDIANCUT
    )
    labels = np.asarray(indexed, dtype=np.uint8)[0]
    raw_palette = indexed.getpalette() or []

    cells: list[BlockCell] = []
    for (y, x), label in zip(positions, labels):
        offset = int(label) * 3
        red, green, blue = raw_palette[offset : offset + 3]
        cells.append((int(x), int(y), f"#{red:02x}{green:02x}{blue:02x}"))
    return cells


def render_blocks(
    cells: list[BlockCell],
    columns: int,
    rows: int,
    size: int,
    gap_ratio: float,
) -> list[str]:
    """Render each sampled terminal cell as a small SVG color block."""
    cell_width = size / columns
    cell_height = size / rows
    gap = min(cell_width, cell_height) * gap_ratio
    rendered: list[str] = []
    for x, y, paint in cells:
        paint_attribute = (
            f'fill="{paint}"' if paint.startswith("#") else f'class="{paint}"'
        )
        rendered.append(
            f'    <rect {paint_attribute} x="{format_number(x * cell_width + gap / 2)}" '
            f'y="{format_number(y * cell_height + gap / 2)}" '
            f'width="{format_number(cell_width - gap)}" height="{format_number(cell_height - gap)}" '
            f'rx="{format_number(gap * 0.42)}"/>'
        )
    return rendered


def render_ascii(rows: list[list[AsciiCell]], size: int) -> list[str]:
    if not rows:
        return []
    columns = max(len(row) for row in rows)
    horizontal_padding = 4
    cell_width = (size - horizontal_padding) / columns
    font_size = cell_width / 0.61
    line_height = size / len(rows)
    baseline = line_height * 0.82
    rendered: list[str] = []
    for row_index, row in enumerate(rows):
        spans: list[str] = []
        column = 0
        for color_class, cells in itertools.groupby(row, key=lambda cell: cell[1]):
            characters = "".join(cell[0] for cell in cells)
            if color_class != "ascii-blank":
                x = horizontal_padding / 2 + column * cell_width
                spans.append(
                    f'<tspan class="{color_class}" x="{format_number(x)}">'
                    f"{html.escape(characters)}</tspan>"
                )
            column += len(characters)
        rendered.append(
            f'    <text class="avatar-ascii" y="{format_number(baseline + row_index * line_height)}" '
            f'font-size="{format_number(font_size)}">{"".join(spans)}</text>'
        )
    return rendered


def render_standalone(
    paths: list[str],
    artwork: list[str],
    size: int,
    palette: list[tuple[int, int, int]],
    mode: str,
) -> str:
    source_colors = ", ".join(f"rgb{color}" for color in palette) or "semantic terminal palette"
    if mode == "blocks":
        title = "MorningKay terminal block portrait"
        description = "A low-resolution SVG color-cell portrait; no raster image is embedded."
    elif not paths:
        title = "MorningKay ASCII portrait"
        description = "A plain-text ASCII portrait for a terminal-style profile; no raster image is embedded."
    else:
        title = "Vectorized MorningKay portrait"
        description = "A color-quantized vector portrait traced from a local raster input."
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">
  <title>{title}</title>
  <desc>{description} Source clusters: {html.escape(source_colors)}.</desc>
  <style>
    .avatar-tone-0 {{ fill: #313244; }} .avatar-tone-1 {{ fill: #585b70; }}
    .avatar-tone-2 {{ fill: #8f849c; }} .avatar-tone-3 {{ fill: #cba6b7; }}
    .avatar-tone-4 {{ fill: #f2cdcd; }} .avatar-tone-5 {{ fill: #f5e0dc; }}
    .avatar-tone-6 {{ fill: #f9e2af; }} .avatar-blue {{ fill: #89b4fa; }}
    .avatar-magenta {{ fill: #f38ba8; }} .avatar-ink {{ fill: #1e1e2e; }}
    .avatar-ascii {{ font-family: monospace; font-weight: 600; white-space: pre; }}
    .ascii-tone-0 {{ fill: #6c7086; }} .ascii-tone-1 {{ fill: #9399b2; }}
    .ascii-tone-2 {{ fill: #cba6b7; }} .ascii-tone-3 {{ fill: #f2cdcd; }}
    .ascii-tone-4 {{ fill: #f5e0dc; }} .ascii-blue {{ fill: #89b4fa; }}
    .ascii-magenta {{ fill: #cba6f7; }}
    .block-ink {{ fill: #313244; }} .block-hair-shadow {{ fill: #6c7086; }}
    .block-hair {{ fill: #89b4fa; }} .block-hair-highlight {{ fill: #d9e0ee; }}
    .block-skin {{ fill: #f5c2c7; }} .block-skin-shadow {{ fill: #eba0ac; }}
    .block-pink {{ fill: #f38ba8; }} .block-pink-dark {{ fill: #b65d7a; }}
    .block-light {{ fill: #f5e0dc; }} .block-neutral {{ fill: #bac2de; }}
    .block-shadow {{ fill: #7f849c; }}
  </style>
  <rect width="{size}" height="{size}" rx="18" fill="#45475a"/>
  <g shape-rendering="{'crispEdges' if mode == 'blocks' else 'geometricPrecision'}">
{chr(10).join(paths)}
{chr(10).join(artwork)}
  </g>
</svg>
'''


def embed_paths(svg_path: Path, paths: list[str], artwork: list[str]) -> None:
    source = svg_path.read_text(encoding="utf-8")
    rendered = chr(10).join(paths + artwork)
    replacement = f"{AVATAR_START}\n{rendered}\n    {AVATAR_END}"
    pattern = rf"{re.escape(AVATAR_START)}.*?{re.escape(AVATAR_END)}"
    updated, count = re.subn(pattern, replacement, source, flags=re.DOTALL)
    if count != 1:
        raise SystemExit(f"expected one avatar marker pair in {svg_path}, found {count}")
    svg_path.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="local JPEG/PNG source")
    parser.add_argument("output", type=Path, help="standalone SVG preview")
    parser.add_argument("--threshold", type=int, default=112, choices=range(1, 255))
    parser.add_argument("--tolerance", type=float, default=1.15)
    parser.add_argument("--colors", type=int, default=7, choices=range(3, 13))
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--min-area", type=float, default=3.0)
    parser.add_argument(
        "--ascii-columns",
        type=int,
        default=44,
        help="ASCII columns overlaid on the vector portrait (default: 44)",
    )
    parser.add_argument(
        "--style",
        choices=("ascii", "hybrid", "blocks"),
        default="ascii",
        help="render pure ASCII, ASCII over vector paths, or SVG color blocks (default: ascii)",
    )
    parser.add_argument(
        "--ascii-file",
        type=Path,
        help="use a hand-curated ASCII text file instead of automatic sampling",
    )
    parser.add_argument(
        "--block-columns",
        type=int,
        default=48,
        help="number of columns in block mode (default: 48)",
    )
    parser.add_argument(
        "--block-rows",
        type=int,
        help="number of rows in block mode (default: same as columns)",
    )
    parser.add_argument(
        "--block-palette",
        type=int,
        default=16,
        choices=range(4, 25),
        help="source-derived palette size in block mode (default: 16)",
    )
    parser.add_argument(
        "--block-gap",
        type=float,
        default=0.015,
        help="gap ratio between color blocks, from 0 to 0.2 (default: 0.015)",
    )
    parser.add_argument(
        "--crop",
        type=parse_crop,
        help="normalized source crop as left,top,right,bottom",
    )
    parser.add_argument(
        "--embed",
        type=Path,
        nargs="*",
        default=[],
        metavar="SVG",
        help="also replace the avatar marker pair in one or more profile SVGs",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"input image does not exist: {args.input}")
    if args.ascii_file is not None and not args.ascii_file.is_file():
        raise SystemExit(f"ASCII art file does not exist: {args.ascii_file}")
    if (
        args.tolerance < 0
        or args.min_area < 0
        or args.size < 32
        or args.ascii_columns < 16
        or args.block_columns < 16
        or (args.block_rows is not None and args.block_rows < 16)
        or not 0 <= args.block_gap <= 0.2
    ):
        raise SystemExit(
            "tolerance/min-area must be non-negative, size must be at least 32, "
            "and ASCII/block columns must be at least 16"
        )

    if args.style == "hybrid":
        paths, palette = vectorize(
            args.input,
            args.size,
            args.colors,
            args.threshold,
            args.tolerance,
            args.min_area,
        )
    else:
        paths, palette = [], []
    if args.style == "blocks":
        block_rows = args.block_rows or args.block_columns
        block_cells = make_block_cells(
            args.input,
            args.block_columns,
            block_rows,
            args.crop,
            args.block_palette,
        )
        artwork = render_blocks(
            block_cells,
            args.block_columns,
            block_rows,
            args.size,
            args.block_gap,
        )
        generated_summary = f"{len(block_cells)} SVG color blocks"
    else:
        if args.ascii_file is not None:
            ascii_rows = load_curated_ascii(args.ascii_file)
        else:
            ascii_rows = make_ascii_rows(
                args.input,
                args.ascii_columns,
                max(8, round(args.ascii_columns * 0.7)),
            )
        artwork = render_ascii(ascii_rows, args.size)
        generated_summary = f"{len(ascii_rows)} ASCII rows"
    args.output.write_text(
        render_standalone(paths, artwork, args.size, palette, args.style),
        encoding="utf-8",
    )
    for svg_path in args.embed:
        embed_paths(svg_path, paths, artwork)
    print(
        f"generated {len(paths)} vector layers plus {generated_summary} "
        f"at {args.size}x{args.size} -> {args.output}"
        + (f"; embedded into {len(args.embed)} SVG(s)" if args.embed else "")
    )


if __name__ == "__main__":
    main()
