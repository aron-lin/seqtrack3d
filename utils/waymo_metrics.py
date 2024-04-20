import numpy as np
import torch
import torchmetrics.utilities.data
from shapely.geometry import Polygon

def fromWaymoBoxToPoly(box):
    return Polygon(tuple(box.corners()[[0, 1]].T[[0, 1, 5, 4]]))

def estimateWaymoOverlap(box_a, box_b, dim=2):

    Poly_anno = fromWaymoBoxToPoly(box_a)
    Poly_subm = fromWaymoBoxToPoly(box_b)

    box_inter = Poly_anno.intersection(Poly_subm)
    box_union = Poly_anno.union(Poly_subm)
    if dim == 2:
        return box_inter.area / box_union.area
    else:
        zmax = min(box_a.center[2], box_b.center[2])
        zmin = max(box_a.center[2] - box_a.wlh[2],
                   box_b.center[2] - box_b.wlh[2])
        inter_vol = box_inter.area * max(0, zmax - zmin)
        anno_vol = box_a.wlh[0] * box_a.wlh[1] * box_a.wlh[2]
        subm_vol = box_b.wlh[0] * box_b.wlh[1] * box_b.wlh[2]
        overlap = inter_vol * 1.0 / (anno_vol + subm_vol - inter_vol)
    return overlap