import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.ndimage import convolve
from collections import deque
import os

BASE_DIR = os.path.dirname(__file__)
@st.cache_data
def load_empirical(kind: str):
    veg_path = os.path.join(BASE_DIR, "field-data", kind, "vegetation.asc")
    elev_path = os.path.join(BASE_DIR, "field-data", kind, "topography.asc")
    
    veg = np.loadtxt(veg_path)[1:-1, 1:-1] / 100
    elev = np.loadtxt(elev_path)[1:-1, 1:-1]
    plane = elev.mean(axis=1, keepdims=True)
    micro = elev - plane
    return veg, plane, micro

def get_figsize(matrix_shape, max_width=10, max_height=10, cell_size=0.06):
    n_rows, n_cols = matrix_shape
    # Calculate desired width and height based on cell size
    width, height = n_cols * cell_size, n_rows * cell_size
    # Clamp width and height to the maximum allowed
    if width > max_width:
        scale, width = max_width / width, max_width
        height = height * scale
    if height > max_height:
        scale, height = max_height / height, max_height
        width = width * scale
    return (width, height)

def generate_vegetation_matrix(rows, cols, ratio_v, pclust, v_source):
    '''
    Generate vegetation distribution on a matrix landscape.
    Inputs
        rows, cols: dimensions of landscape matrix (int)
        ratio_v: proportion of vegetated cells, float [0,1]
        pclust: clustering probability, float [0,1]
        v_source: dict of empirical vegetation density samples {key: density_value}
    Output
        v: 2D numpy array of vegetation densities, shape (rows, cols)
    '''
    v = np.zeros((rows, cols), dtype=float)
    # Set of all free cells as coordinate tuples
    free_cells_list = [(i, j) for i in range(rows) for j in range(cols)]
    free_cells_set = set(free_cells_list)
    nb_veg_final, nb_veg = int(rows * cols * ratio_v), 0
    v_sample = v_source[np.where(v_source)]
    # Start at a random free cell
    idx = random.randrange(len(free_cells_list))
    node = free_cells_list[idx]
    while nb_veg < nb_veg_final:
        # Assign vegetation sample
        v[node] = random.choice(v_sample)
        nb_veg += 1
        free_cells_set.remove(node)
        # Swap remove node from list to keep O(1) removal
        last = free_cells_list[-1]
        pos = idx
        free_cells_list[pos] = last
        free_cells_list.pop()
        # If list still has elements, update idx for next selection
        if nb_veg == nb_veg_final or not free_cells_list:
            break
        i, j = node
        neighbor_coords = [
            (i, j + 1), (i, j - 1),
            (i - 1, j), (i + 1, j),
            (i - 1, j + 1), (i + 1, j + 1),
            (i + 1, j - 1), (i - 1, j - 1)]
        free_neighbors = [n for n in neighbor_coords if 0 <= n[0] < rows and 0 <= n[1] < cols and n in free_cells_set]
        if random.random() < pclust and free_neighbors:
            node = random.choice(free_neighbors)
            # Update idx correspondingly
            idx = free_cells_list.index(node)  # O(n), but neighbors list is small (max 8), so tiny cost here
        else:
            idx = random.randrange(len(free_cells_list))
            node = free_cells_list[idx]
    return v

def compute_landscape_vegetation_score(vmat: np.ndarray) -> np.ndarray:
    K = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=float)
    return convolve(vmat, K, mode="nearest")

def generate_microtopography(v, interpolation, damping=0.2):
    score = compute_landscape_vegetation_score(v)
    eps = np.random.normal(0, interpolation['alpha_std']*damping, size=v.shape)
    return interpolation['alpha'] * score + interpolation['beta'] + eps

def d4_steepest_descent(tmat: np.ndarray) -> np.ndarray:
    """
    Return a direction matrix with codes:
      0 = sink/no successor
      1 = East, 2 = South, 3 = West, 4 = North
    Hillslope rules:
      - Left-right (W/E) neighbors are periodic (wrap across columns).
      - Top row cannot point North (no N candidate is considered).
      - Bottom row is forced to point South (code 2) for all columns: outflow layer.
      - Otherwise, choose the strictly lower neighbor (E,S,W,N) with the largest drop (t[i,j] - t[ni,nj]).
        If no strictly lower neighbor exists, choose the lowest neighbor among allowed directions to ensure a defined path.
        Ties are broken by the order [E, S, W, N].
    """
    R, C = tmat.shape
    out = np.zeros((R, C), dtype=np.uint8)

    for i in range(R):
        for j in range(C):
            # Bottom row: all point South
            if i == R - 1:
                out[i, j] = 2
                continue
            # Collect D4 neighbors with lateral periodic boundaries for E/W
            # East (wrap)
            ej = (j + 1) % C
            e_elev = tmat[i, ej]
            # West (wrap)
            wj = (j - 1) % C
            w_elev = tmat[i, wj]
            # South (in-bounds unless bottom row, handled above)
            s_elev = tmat[i + 1, j]
            # North (disallowed on top row)
            n_allowed = (i > 0)
            n_elev = tmat[i - 1, j] if n_allowed else np.inf
            center = tmat[i, j]
            # Compute drops (positive means downhill)
            drops = [
                (center - e_elev, 1, i, ej),           # E
                (center - s_elev, 2, i + 1, j),        # S
                (center - w_elev, 3, i, wj),           # W
                (center - n_elev, 4, i - 1, j)         # N (ignored if top row via inf)
            ]
            # Filter out N if top row (drop will be -inf or -large if n_allowed False)
            if not n_allowed:
                drops[3] = (-np.inf, 4, i, j)  # effectively disable N
            # First, prefer strictly downhill candidates
            downhill = [d for d in drops if d[0] > 0]
            if downhill:
                # Choose max drop (steepest), tie by the order [E,S,W,N]
                downhill.sort(key=lambda x: (-x[0], x[1]))
                out[i, j] = downhill[0][1]
            else:
                # No strictly lower neighbor: pick the lowest elevation among allowed directions
                # Construct elevation list matching the same order [E,S,W,N]
                elevs = [
                    (e_elev, 1),
                    (s_elev, 2),
                    (w_elev, 3),
                    (n_elev, 4),
                ]
                elevs.sort(key=lambda x: (x[0], x[1]))  # lowest elevation, tie by order
                choice_code = elevs[0][1]
                out[i, j] = choice_code
    return out

_DY = np.array([0,  1,  0, -1], dtype=int)   # E, S, W, N
_DX = np.array([1,  0, -1,  0], dtype=int)

def compute_SC(out: np.ndarray) -> np.ndarray:
    """
    Structural connectivity (upslope contributing cells) from a D4 direction matrix.
    Input:
      out: (R,C) uint8/int array with {0=sink, 1=E, 2=S, 3=W, 4=N}
           E/W are understood as periodic laterally; N disallowed on top row by construction;
           S on bottom row flows out of the domain (treated as sink).
    Output:
      sc: (R,C) int array with number of upslope cells draining to each cell (excludes the cell itself).
    """
    R, C = out.shape
    rr, cc = np.indices((R, C))
    # Map codes to neighbor index 0..3; 0 -> -1 for sink
    lut = np.full(256, -1, dtype=int)
    lut[[1, 2, 3, 4]] = np.arange(4)  # E,S,W,N -> 0..3
    ks = lut[out.astype(np.uint8)]    # (R,C)
    # Build successor indices with lateral wrap (E/W), row bounds for N/S
    has_succ = ks >= 0
    dy = np.zeros_like(ks, dtype=int)
    dx = np.zeros_like(ks, dtype=int)
    # Only fill where there is a direction
    m = has_succ
    dy[m] = _DY[ks[m]]
    dx[m] = _DX[ks[m]]
    nr = rr + dy
    nc = (cc + dx) % C  # periodic columns
    # If S from bottom row, treat as out-of-domain sink (no successor)
    s_from_bottom = (out == 2) & (rr == R - 1)
    has_succ[s_from_bottom] = False
    # Also guard any row-out-of-bounds (e.g., N from top if present) as sinks
    in_bounds = (nr >= 0) & (nr < R)
    has_succ &= in_bounds
    nr[~has_succ] = -1
    nc[~has_succ] = -1
    # Flatten successors
    u_flat = (rr * C + cc).ravel()
    next_flat = np.full(R * C, -1, dtype=int)
    mask = has_succ.ravel()
    nflat = (nr.ravel()[mask] * C + nc.ravel()[mask])
    next_flat[u_flat[mask]] = nflat
    # In-degree per node
    indeg = np.zeros(R * C, dtype=int)
    valid = next_flat >= 0
    np.add.at(indeg, next_flat[valid], 1)
    # Accumulation: start with 1 per cell and subtract 1 at end to exclude self
    acc = np.ones(R * C, dtype=int)
    q = deque(u_flat[indeg == 0].tolist())
    # Kahn-style topological propagation (loops are unlikely on a hillslope)
    while q:
        u = q.popleft()
        v = next_flat[u]
        if v >= 0:
            acc[v] += acc[u]
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    sc = acc.reshape(R, C) - 1
    sc[sc < 0] = 0
    return sc


