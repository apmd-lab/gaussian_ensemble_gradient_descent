#define PY_ARRAY_UNIQUE_SYMBOL MY_MODULE_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <Python.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "fill_required_pixels.h"
#include "make_touch.h"
#include "touch2pix.h"
#include "utils.h"

#define IDX(i, j, W) ((i) * (W) + (j))

static inline int periodic_dist_sq(int x1, int y1, int x2, int y2, int Nx, int Ny) {
    int dx = abs(x1 - x2);
    if (dx > Nx / 2) dx = Nx - dx;
    int dy = abs(y1 - y2);
    if (dy > Ny / 2) dy = Ny - dy;
    return dx * dx + dy * dy;
}

static inline int min_cand_dist_sq(int tx, int ty, int cx, int cy, int Nx, int Ny, int symmetry) {
    int d_min = periodic_dist_sq(tx, ty, cx, cy, Nx, Ny);
    if (symmetry == 1) {
        int sym_x = (Nx - 1 - cx + Nx) % Nx;
        int d_sym = periodic_dist_sq(tx, ty, sym_x, cy, Nx, Ny);
        if (d_sym < d_min) d_min = d_sym;
    } else if (symmetry == 2) {
        int sym_x = (Nx - 1 - cx + Nx) % Nx;
        int sym_y = (Ny - 1 - cy + Ny) % Ny;
        int d2 = periodic_dist_sq(tx, ty, sym_x, cy, Nx, Ny);
        if (d2 < d_min) d_min = d2;
        int d3 = periodic_dist_sq(tx, ty, cx, sym_y, Nx, Ny);
        if (d3 < d_min) d_min = d3;
        int d4 = periodic_dist_sq(tx, ty, sym_x, sym_y, Nx, Ny);
        if (d4 < d_min) d_min = d4;
    }
    return d_min;
}

// Helper to check if placing a touch at (cx, cy) with phase is_solid
// would leave any unassigned pixel unreachable (avail_solid == 0 && avail_void == 0).
static int is_candidate_touch_valid(int cx, int cy, int is_solid,
                                    const int* touch_solid,
                                    const int* touch_void,
                                    const int* pix_solid,
                                    const int* brush_pts_dx,
                                    const int* brush_pts_dy,
                                    int n_brush_pts,
                                    int r_c,
                                    int Nx, int Ny, int symmetry) {

    const int* touch_same = is_solid ? touch_solid : touch_void;
    const int* touch_opp = is_solid ? touch_void : touch_solid;

    const int r_c_sq = r_c * r_c;
    const int r_opp_forbid_sq = (2 * r_c) * (2 * r_c);
    const int R_check = 3 * r_c;
    const int R_check_sq = R_check * R_check;

    // Check all pixels in the interaction radius around (cx, cy) and its reflection
    int n_centers = (symmetry == 1) ? 2 : (symmetry == 2 ? 4 : 1);
    int center_x[8] = {cx, (Nx - 1 - cx + Nx) % Nx, cx, (Nx - 1 - cx + Nx) % Nx, 0, 0, 0, 0};
    int center_y[8] = {cy, cy, (Ny - 1 - cy + Ny) % Ny, (Ny - 1 - cy + Ny) % Ny, 0, 0, 0, 0};

    for (int c = 0; c < n_centers; ++c) {
        int ccx = center_x[c];
        int ccy = center_y[c];
        if (c > 0 && ccx == center_x[0] && ccy == center_y[0]) continue; // Avoid duplicate center

        for (int dpx = -R_check; dpx <= R_check; ++dpx) {
            int px = (ccx + dpx + Nx * 2) % Nx;
            for (int dpy = -R_check; dpy <= R_check; ++dpy) {
                if (dpx * dpx + dpy * dpy > R_check_sq) continue; // Skip circular corners

                int py = (ccy + dpy + Ny * 2) % Ny;
                int p_idx = px * Ny + py;
                if (pix_solid[p_idx] != 0) continue; // Already assigned

                // If pixel is covered by new touch, it becomes assigned and is safe
                if (min_cand_dist_sq(px, py, cx, cy, Nx, Ny, symmetry) <= r_c_sq) {
                    continue;
                }

                // Check if pixel has ANY available touch of the same phase
                int avail_same = 0;
                for (int b = 0; b < n_brush_pts; ++b) {
                    int tx = (px + brush_pts_dx[b] + Nx * 2) % Nx;
                    int ty = (py + brush_pts_dy[b] + Ny * 2) % Ny;
                    if (touch_same[tx * Ny + ty] == 0) {
                        avail_same = 1;
                        break;
                    }
                }
                if (avail_same) continue; // Pixel has available same-phase touch: safe!

                // Pixel has 0 available same-phase touches (it was already required opposite phase).
                // Check if AT LEAST ONE available opposite-phase touch survives this candidate:
                int avail_opp = 0;
                for (int b = 0; b < n_brush_pts; ++b) {
                    int tx = (px + brush_pts_dx[b] + Nx * 2) % Nx;
                    int ty = (py + brush_pts_dy[b] + Ny * 2) % Ny;
                    if (touch_opp[tx * Ny + ty] == 0) {
                        // Check if this touch is NOT forbidden by candidate
                        if (min_cand_dist_sq(tx, ty, cx, cy, Nx, Ny, symmetry) > r_opp_forbid_sq) {
                            avail_opp = 1;
                            break;
                        }
                    }
                }

                if (!avail_opp) {
                    // Candidate destroys all touches for pixel (px, py)!
                    return 0; // Invalid
                }
            }
        }
    }

    return 1; // All checked pixels remain reachable
}

void fill_required_pixels(int* ind_max,
                          int* touch_solid,
                          int* touch_void,
                          int* pix_solid,
                          float* score_solid,
                          int* refconv0,
                          int* refconv1,
                          int* refconv2,
                          int Nx,
                          int Ny,
                          int symmetry,
                          int brush_size) {

    int size = Nx * Ny;
    int r_c = (brush_size - 1) / 2;
    int max_brush_pts = brush_size * brush_size;

    int* last_affected = (int*)calloc(size, sizeof(int));
    int* updated = (int*)malloc(size * sizeof(int));
    int* brush_pts_dx = (int*)malloc(max_brush_pts * sizeof(int));
    int* brush_pts_dy = (int*)malloc(max_brush_pts * sizeof(int));
    int* cand_indices = (int*)malloc(max_brush_pts * sizeof(int));
    float* cand_scores = (float*)malloc(max_brush_pts * sizeof(float));

    if (!last_affected || !updated || !brush_pts_dx || !brush_pts_dy ||
        !cand_indices || !cand_scores) {
        free(last_affected);
        free(updated);
        free(brush_pts_dx);
        free(brush_pts_dy);
        free(cand_indices);
        free(cand_scores);
        return;
    }

    // Extract relative brush point offsets dynamically
    int n_brush_pts = 0;
    for (int dx = -r_c; dx <= r_c; ++dx) {
        for (int dy = -r_c; dy <= r_c; ++dy) {
            if (dx * dx + dy * dy <= r_c * r_c) {
                brush_pts_dx[n_brush_pts] = dx;
                brush_pts_dy[n_brush_pts] = dy;
                n_brush_pts++;
            }
        }
    }

    if (ind_max != NULL) {
        roll2d(refconv2, last_affected, ind_max[0], ind_max[1], Nx, Ny);
        if (symmetry == 1) {
            int sym_x = (Nx - 1 - ind_max[0] + Nx) % Nx;
            roll2d(refconv2, updated, sym_x, ind_max[1], Nx, Ny);
            for (int k = 0; k < size; ++k) last_affected[k] |= updated[k];
        } else if (symmetry == 2) {
            int sym_x = (Nx - 1 - ind_max[0] + Nx) % Nx;
            int sym_y = (Ny - 1 - ind_max[1] + Ny) % Ny;
            roll2d(refconv2, updated, sym_x, ind_max[1], Nx, Ny);
            for (int k = 0; k < size; ++k) last_affected[k] |= updated[k];
            roll2d(refconv2, updated, ind_max[0], sym_y, Nx, Ny);
            for (int k = 0; k < size; ++k) last_affected[k] |= updated[k];
            roll2d(refconv2, updated, sym_x, sym_y, Nx, Ny);
            for (int k = 0; k < size; ++k) last_affected[k] |= updated[k];
        }
    } else {
        // NULL ind_max triggers a full-grid pass to resolve all required pixels (e.g. after pre-fill)
        for (int k = 0; k < size; ++k) {
            if (pix_solid[k] == 0) last_affected[k] = 1;
        }
    }

    int required = 1;
    while (required) {
        required = 0;

        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                int idx = IDX(i, j, Ny);
                if (pix_solid[idx] == 0 && last_affected[idx]) {

                    // Fast check for required pixel using brush offsets (with early break)
                    int all_solid = 1, all_void = 1;
                    for (int b = 0; b < n_brush_pts; ++b) {
                        int tx = (i + brush_pts_dx[b] + Nx * 2) % Nx;
                        int ty = (j + brush_pts_dy[b] + Ny * 2) % Ny;
                        int tidx = tx * Ny + ty;
                        if (touch_solid[tidx] == 0) all_solid = 0;
                        if (touch_void[tidx] == 0) all_void = 0;
                        if (!all_solid && !all_void) break; // Neither phase is required
                    }

                    if (all_solid || all_void) {
                        int* touch = (all_solid ? touch_void : touch_solid);
                        int req_is_solid = !all_solid;

                        // Gather candidate touch locations covering pixel (i, j)
                        int n_cands = 0;
                        for (int b = 0; b < n_brush_pts; ++b) {
                            int tx = (i + brush_pts_dx[b] + Nx * 2) % Nx;
                            int ty = (j + brush_pts_dy[b] + Ny * 2) % Ny;
                            int tidx = tx * Ny + ty;
                            if (touch[tidx] == 0) {
                                float score = all_solid ? -score_solid[tidx] : score_solid[tidx];
                                if (n_cands < max_brush_pts) {
                                    cand_indices[n_cands] = tidx;
                                    cand_scores[n_cands] = score;
                                    n_cands++;
                                }
                            }
                        }

                        // Sort candidates by score descending
                        for (int a = 0; a < n_cands - 1; ++a) {
                            for (int b = a + 1; b < n_cands; ++b) {
                                if (cand_scores[b] > cand_scores[a]) {
                                    float temp_s = cand_scores[a];
                                    cand_scores[a] = cand_scores[b];
                                    cand_scores[b] = temp_s;
                                    int temp_i = cand_indices[a];
                                    cand_indices[a] = cand_indices[b];
                                    cand_indices[b] = temp_i;
                                }
                            }
                        }

                        // Pick the highest scoring candidate that doesn't create unreachable pixels
                        int chosen_idx = -1;
                        for (int c = 0; c < n_cands; ++c) {
                            int k = cand_indices[c];
                            int cx = k / Ny;
                            int cy = k % Ny;
                            if (is_candidate_touch_valid(cx, cy, req_is_solid,
                                                         touch_solid, touch_void, pix_solid,
                                                         brush_pts_dx, brush_pts_dy, n_brush_pts,
                                                         r_c, Nx, Ny, symmetry)) {
                                chosen_idx = k;
                                break;
                            }
                        }

                        // Safety fallback if no candidate passed validation
                        if (chosen_idx < 0 && n_cands > 0) {
                            chosen_idx = cand_indices[0];
                        }

                        if (chosen_idx >= 0) {
                            int ix = chosen_idx / Ny;
                            int iy = chosen_idx % Ny;
                            int ind_req[2] = {ix, iy};

                            make_touch(!all_solid,
                                       all_solid,
                                       ind_req,
                                       touch_solid,
                                       touch_void,
                                       pix_solid,
                                       refconv0,
                                       refconv1,
                                       Nx,
                                       Ny,
                                       symmetry);

                            roll2d(refconv2, updated, ix, iy, Nx, Ny);
                            for (int k = 0; k < size; ++k)
                                last_affected[k] |= updated[k];

                            if (symmetry == 1) {
                                int sym_x = (Nx - 1 - ix + Nx) % Nx;
                                roll2d(refconv2, updated, sym_x, iy, Nx, Ny);
                                for (int k = 0; k < size; ++k)
                                    last_affected[k] |= updated[k];
                            } else if (symmetry == 2) {
                                int sym_x = (Nx - 1 - ix + Nx) % Nx;
                                int sym_y = (Ny - 1 - iy + Ny) % Ny;
                                roll2d(refconv2, updated, sym_x, iy, Nx, Ny);
                                for (int k = 0; k < size; ++k) last_affected[k] |= updated[k];
                                roll2d(refconv2, updated, ix, sym_y, Nx, Ny);
                                for (int k = 0; k < size; ++k) last_affected[k] |= updated[k];
                                roll2d(refconv2, updated, sym_x, sym_y, Nx, Ny);
                                for (int k = 0; k < size; ++k) last_affected[k] |= updated[k];
                            }

                            required = 1;
                            goto LOOP_END;
                        }
                    }
                    last_affected[idx] = 0;
                }
            }
        }
        LOOP_END:;
    }

    free(last_affected);
    free(updated);
    free(brush_pts_dx);
    free(brush_pts_dy);
    free(cand_indices);
    free(cand_scores);
}