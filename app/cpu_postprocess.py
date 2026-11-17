"""Small, dependency-light polygon post-processing for the CPU backend."""

from shapely.geometry import Polygon


def valid_polygon(points):
    try:
        polygon = Polygon(points)
    except Exception:
        return None
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon if not polygon.is_empty and polygon.area > 0 else None


def _bounds(item):
    points = item.get("polygon", [])
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _overlap(first, second):
    intersection = first.intersection(second).area
    union = first.union(second).area
    return intersection / union if union else 0.0


def polygon_nms(items, threshold):
    ranked = sorted(items, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    kept = []
    shapes = []
    bounds = []
    removed = 0
    for item in ranked:
        shape = valid_polygon(item.get("polygon", []))
        if shape is None:
            removed += 1
            continue
        current_bounds = shape.bounds
        duplicate = any(
            not (current_bounds[2] <= other[0] or other[2] <= current_bounds[0]
                 or current_bounds[3] <= other[1] or other[3] <= current_bounds[1])
            and _overlap(shape, other_shape) >= threshold
            for other_shape, other in zip(shapes, bounds)
        )
        if duplicate:
            removed += 1
            continue
        kept.append(item)
        shapes.append(shape)
        bounds.append(current_bounds)
    return kept, removed


def boundary_ranked_items(items, margin=48.0, penalty=0.15):
    ranked = []
    for item in items:
        copy = dict(item)
        tile = copy.get("source_tile")
        if tile and copy.get("polygon"):
            x0, y0, x1, y1 = map(float, tile)
            left, top, right, bottom = _bounds(copy)
            distance = min(left - x0, top - y0, x1 - right, y1 - bottom)
            strength = max(0.0, min(1.0, (margin - distance) / max(margin, 1.0)))
            copy["_original_score"] = float(copy.get("score", 0.0))
            copy["score"] = copy["_original_score"] * (1.0 - penalty * strength)
        ranked.append(copy)
    return ranked


def restore_scores(items):
    restored = []
    for item in items:
        copy = dict(item)
        if "_original_score" in copy:
            copy["score"] = copy.pop("_original_score")
        restored.append(copy)
    return restored


def geometry_filter(items, max_dimension_per_char):
    kept = []
    removed = 0
    for item in items:
        points = item.get("polygon", [])
        text = str(item.get("text", ""))
        if not points:
            removed += 1
            continue
        left, top, right, bottom = _bounds(item)
        ratio = max(right - left, bottom - top) / max(1, len(text))
        if ratio > max_dimension_per_char:
            removed += 1
            continue
        kept.append(item)
    return kept, removed


def public_items(items):
    result = []
    for item in items:
        copy = dict(item)
        copy.pop("source_tile", None)
        for key in list(copy):
            if key.startswith("_"):
                copy.pop(key, None)
        result.append(copy)
    return result
