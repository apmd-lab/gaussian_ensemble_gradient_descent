import numpy as np
import networkx as nx
from scipy.ndimage import binary_opening, convolve, distance_transform_edt, label
from skimage.morphology import skeletonize
from collections import Counter

S8 = np.ones((3, 3), dtype=int)
NEIGHBOR_OFFSETS_HALF = [(0, 1), (1, -1), (1, 0), (1, 1)]

def pairwise_Hamming_distance(x_sample):
    N = x_sample.shape[0]
    p = np.mean(x_sample, axis=0)
    var = p * (1 - p)
    dist = (2 * N / (N - 1)) * np.mean(var, axis=0)
    return dist, var

def _count_connected_components(img, periodic):
    labeled, n_comp = label(img)
    if not periodic or n_comp <= 1:
        return n_comp

    parent = list(range(n_comp + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            return 1
        return 0

    # Merge components touching across top/bottom boundary (axis 0)
    top, bottom = labeled[0, :], labeled[-1, :]
    mask_tb = (top > 0) & (bottom > 0) & (top != bottom)
    for a, b in zip(top[mask_tb], bottom[mask_tb]):
        n_comp -= union(a, b)

    # Merge components touching across left/right boundary (axis 1)
    left, right = labeled[:, 0], labeled[:, -1]
    mask_lr = (left > 0) & (right > 0) & (left != right)
    for a, b in zip(left[mask_lr], right[mask_lr]):
        n_comp -= union(a, b)

    return n_comp

def Betti_numbers(x_sample, Nx, Ny, periodic):
    N = x_sample.shape[0]
    beta0 = np.zeros(N, dtype=int)
    beta1 = np.zeros(N, dtype=int)
    for i in range(N):
        img = x_sample[i,:].reshape(Nx, Ny)
        beta0[i] = _count_connected_components(img, periodic)      # connected solid islands
        beta1[i] = _count_connected_components(1 - img, periodic)  # connected void holes
    unique, counts = np.unique(np.column_stack((beta0, beta1)), axis=0, return_counts=True)
    n_unique = unique.shape[0]
    p = counts / N
    n_eff = 1.0 / np.sum(p**2)

    return n_unique, n_eff

def _label_8conn_periodic(mask, periodic):
    labeled, n_comp = label(mask, structure=S8)
    if not periodic or n_comp <= 1:
        return labeled, n_comp

    parent = list(range(n_comp + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Merge 8-connected components across top/bottom and left/right boundaries
    top, bottom = labeled[0, :], labeled[-1, :]
    for shift in (-1, 0, 1):
        b_shifted = np.roll(bottom, shift) if periodic else np.pad(bottom, 1)[1 + shift : 1 + shift + len(bottom)]
        mask_tb = (top > 0) & (b_shifted > 0) & (top != b_shifted)
        for a, b in zip(top[mask_tb], b_shifted[mask_tb]):
            union(a, b)

    left, right = labeled[:, 0], labeled[:, -1]
    for shift in (-1, 0, 1):
        r_shifted = np.roll(right, shift) if periodic else np.pad(right, 1)[1 + shift : 1 + shift + len(right)]
        mask_lr = (left > 0) & (r_shifted > 0) & (left != r_shifted)
        for a, b in zip(left[mask_lr], r_shifted[mask_lr]):
            union(a, b)

    # Relabel to contiguous IDs 1..n_merged
    roots = [find(i) for i in range(n_comp + 1)]
    unique_roots = sorted(set(roots[1:]))
    root_to_id = {r: idx + 1 for idx, r in enumerate(unique_roots)}
    remap = np.array([0] + [root_to_id[roots[i]] for i in range(1, n_comp + 1)], dtype=int)
    return remap[labeled], len(unique_roots)

def _skeleton_to_reduced_multigraph(img, periodic):
    mode = 'wrap' if periodic else 'constant'
    img_bool = img > 0

    if periodic:
        px, py = img.shape[0] // 2, img.shape[1] // 2
        padded = np.pad(img_bool, ((px, px), (py, py)), mode='wrap')
        skel = skeletonize(padded)[px : px + img.shape[0], py : py + img.shape[1]].astype(int)
    else:
        skel = skeletonize(img_bool).astype(int)

    # 8-connected neighbor count for each skeleton pixel (excluding itself)
    deg = (convolve(skel, S8, mode=mode, cval=0) - 1) * skel

    # 1. Vertices: merge adjacent junction pixels (deg >= 3), keep endpoints/isolated dots distinct
    junc_mask = (skel == 1) & (deg >= 3)
    vertex_labels, n_junc = _label_8conn_periodic(junc_mask, periodic)

    end_or_dot_coords = np.argwhere((skel == 1) & ((deg == 1) | (deg == 0)))
    n_vertices = n_junc
    for r, c in end_or_dot_coords:
        n_vertices += 1
        vertex_labels[r, c] = n_vertices

    G_multi = nx.MultiGraph()
    for v in range(1, n_vertices + 1):
        G_multi.add_node(v)

    # 2. Edges from degree-2 path components
    path_mask = (skel == 1) & (deg == 2)
    path_labels, n_paths = _label_8conn_periodic(path_mask, periodic)

    for p in range(1, n_paths + 1):
        dilated = convolve((path_labels == p).astype(int), S8, mode=mode, cval=0) > 0
        incident_vertices = np.unique(vertex_labels[dilated & (vertex_labels > 0)])
        if len(incident_vertices) == 2:
            G_multi.add_edge(int(incident_vertices[0]), int(incident_vertices[1]))
        elif len(incident_vertices) == 1:
            v = int(incident_vertices[0])
            G_multi.add_edge(v, v)
        elif len(incident_vertices) == 0:
            v_ring = f"ring_{p}"
            G_multi.add_node(v_ring)
            G_multi.add_edge(v_ring, v_ring)
        else:
            for idx in range(len(incident_vertices)):
                for jdx in range(idx + 1, len(incident_vertices)):
                    G_multi.add_edge(int(incident_vertices[idx]), int(incident_vertices[jdx]))

    # 3. Direct edges between distinct vertices that are 8-adjacent with no degree-2 path in between
    direct_pairs = set()
    for dr, dc in NEIGHBOR_OFFSETS_HALF:
        if periodic:
            shifted = np.roll(vertex_labels, shift=(-dr, -dc), axis=(0, 1))
        else:
            shifted = np.zeros_like(vertex_labels)
            r_src = slice(max(0, dr), vertex_labels.shape[0] + min(0, dr))
            c_src = slice(max(0, dc), vertex_labels.shape[1] + min(0, dc))
            r_dst = slice(max(0, -dr), vertex_labels.shape[0] + min(0, -dr))
            c_dst = slice(max(0, -dc), vertex_labels.shape[1] + min(0, -dc))
            shifted[r_dst, c_dst] = vertex_labels[r_src, c_src]

        mask_adj = (vertex_labels > 0) & (shifted > 0) & (vertex_labels != shifted)
        for u, v in zip(vertex_labels[mask_adj], shifted[mask_adj]):
            pair = (int(min(u, v)), int(max(u, v)))
            direct_pairs.add(pair)

    for u, v in direct_pairs:
        G_multi.add_edge(u, v)

    return G_multi

def _multigraph_wl_hash(G_multi):
    G_simple = nx.Graph()
    for v in G_multi.nodes():
        n_selfloops = G_multi.number_of_edges(v, v)
        G_simple.add_node(v, label=f"{G_multi.degree(v)}_{n_selfloops}")

    for u, v in G_multi.edges():
        if u != v and not G_simple.has_edge(u, v):
            G_simple.add_edge(u, v, multiplicity=str(G_multi.number_of_edges(u, v)))

    return nx.weisfeiler_lehman_graph_hash(G_simple, node_attr="label", edge_attr="multiplicity")

def graph_reduction(x_sample, Nx, Ny, periodic):
    N = x_sample.shape[0]
    hashes = []
    for i in range(N):
        img = x_sample[i, :].reshape(Nx, Ny)
        h_fg = _multigraph_wl_hash(_skeleton_to_reduced_multigraph(img, periodic))
        h_bg = _multigraph_wl_hash(_skeleton_to_reduced_multigraph(1 - img, periodic))
        hashes.append((h_fg, h_bg))

    counts = np.array(list(Counter(hashes).values()))
    n_unique = len(counts)
    p = counts / N
    n_eff = 1.0 / np.sum(p**2)

    return n_unique, n_eff

def _local_thickness_brush(mask, periodic):
    mask = mask.astype(bool)
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=int)
    if np.all(mask):
        return np.full(mask.shape, min(mask.shape), dtype=int)

    Nx, Ny = mask.shape
    if periodic:
        px, py = Nx // 2, Ny // 2
        padded = np.pad(mask, ((px, px), (py, py)), mode='wrap')
    else:
        px, py = 0, 0
        padded = mask

    max_edt = np.max(distance_transform_edt(padded))
    d_max = int(np.ceil(2 * max_edt)) + 1

    thickness = np.zeros((Nx, Ny), dtype=int)
    thickness[mask] = 1

    for d in range(3, d_max + 1, 2):
        r_c = (d - 1) // 2
        grid = np.arange(d) - r_c
        X, Y = np.meshgrid(grid, grid)
        brush = (X**2 + Y**2) <= r_c**2

        opened = binary_opening(padded, structure=brush)
        if periodic:
            opened = opened[px : px + Nx, py : py + Ny]

        if np.any(opened):
            thickness[opened] = d
        else:
            break

    return thickness

def get_feature_size_distribution(x, Nx, Ny, periodic):
    x = x.reshape(Nx, Ny)
    # Step 1: Measure Phase 1 (solid)
    fs_solid = _local_thickness_brush(x == 1, periodic)
    # Step 2: Measure Phase 0 (void)
    fs_void = _local_thickness_brush(x == 0, periodic)

    return fs_solid, fs_void