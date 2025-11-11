# ObjectSizeFilter.py
from typing import Dict, Any, List, Tuple, Optional

def compute_size_thresholds(
    bounding_box_start: Tuple[int, int],
    bounding_box_end: Tuple[int, int],
    precision_percent: Optional[int]
) -> Optional[Dict[str, float]]:
    """
    thresholds from pixel ROI and precision percent (e.g., 41 => ±41%).
    Returns None if inputs invalid, meaning 'no filtering'.
    """
    sx, sy = bounding_box_start
    ex, ey = bounding_box_end
    width = abs(float(ex) - float(sx))
    height = abs(float(ey) - float(sy))
    if width <= 0 or height <= 0 or precision_percent is None:
        return None

    pf = float(int(precision_percent)) * 0.01  # 41 -> 0.41
    min_w = int(width - width * pf); max_w = int(width + width * pf)
    min_h = int(height - height * pf); max_h = int(height + height * pf)
    if max_w <= 0 or max_h <= 0:
        return None

    return {
        "min_width": max(1, min_w),
        "max_width": max(1, max_w),
        "min_height": max(1, min_h),
        "max_height": max(1, max_h),
    }

def _bbox_xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float):
    return (x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1))

def _passes_size_filter_xyxy(bbox_xyxy: List[float], th: Dict[str, float]) -> bool:
    x1, y1, x2, y2 = bbox_xyxy
    _, _, w, h = _bbox_xyxy_to_xywh(x1, y1, x2, y2)
    return (th["min_width"] <= w <= th["max_width"]
            and th["min_height"] <= h <= th["max_height"])

def filter_detections_by_size(
    dets_for_stream: List[Tuple[Any, List[Dict[str, Any]]]],
    thresholds: Optional[Dict[str, float]],
    bbox_key: str = "bbox"
) -> List[Tuple[Any, List[Dict[str, Any]]]]:
    """
    dets_for_stream: List of (meta, [det_dict...]).
    Each det_dict must include bbox_key as [x1,y1,x2,y2] in pixels.
    Returns same shape with filtered dets. If thresholds is None: passthrough.
    """
    if thresholds is None:
        return dets_for_stream

    out: List[Tuple[Any, List[Dict[str, Any]]]] = []
    for meta, det_list in dets_for_stream:
        if not det_list:
            out.append((meta, det_list))
            continue
        kept = []
        for d in det_list:
            bbox = d.get(bbox_key)
            if bbox and len(bbox) == 4 and _passes_size_filter_xyxy(bbox, thresholds):
                kept.append(d)
        out.append((meta, kept))
    return out