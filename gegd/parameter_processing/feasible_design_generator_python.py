import os
directory = os.path.dirname(os.path.realpath(__file__))

import numpy as np
from scipy.ndimage import convolve, zoom, gaussian_filter, maximum_filter
from numba import jit
import time

def make_feasible(
    weight,
    N_x,
    N_y,
    brush_size,
    brush_type,
    padding=None,
    periodic=True,
    symmetry='None',
    dim=2,
    upsample_ratio=1,
):
    # Reshape weight into a 1D or 2D array
    if dim == 1:
        weight_sq = weight.reshape(-1,1)
    else:
        weight_sq = weight.reshape(N_x, N_y)
    
    N_x = int(N_x*upsample_ratio)
    N_y = int(N_y*upsample_ratio)
    
    if upsample_ratio > 1:
        brush_size = int(np.floor(upsample_ratio*np.sqrt((brush_size - 2)**2 + 1) + 2))
        
        if periodic:
            weight_sq = zoom(weight_sq, upsample_ratio, order=0, mode='wrap')
        else:
            weight_sq = zoom(weight_sq, upsample_ratio, order=0, mode='grid-constant')
        weight_sq = apply_symmetry(symmetry, weight_sq, N_x, N_y)
    
    if not periodic:
        pad_value = -1 if padding == 'void' else (1 if padding == 'solid' else 0)

        # Padding weight_sq and creating the mask for padded regions
        if dim == 1:
            weight_sq = np.pad(weight_sq, (brush_size, brush_size), constant_values=pad_value)

            # Create a mask to identify padded regions for 1D
            mask = np.ones(weight_sq.shape, dtype=bool)
            mask[brush_size:-brush_size] = False
        else:
            weight_sq = np.pad(weight_sq, ((brush_size, brush_size), (brush_size, brush_size)), constant_values=pad_value)

            # Create a mask to identify padded regions for 2D
            mask = np.ones(weight_sq.shape, dtype=bool)
            mask[brush_size:-brush_size, brush_size:-brush_size] = False
    else:
        mask = np.zeros(weight_sq.shape, dtype=bool)  # No padded regions if periodic
    
    # Initialize the void and solid arrays
    pix_solid = np.zeros(weight_sq.shape)
    pix_void = np.zeros(weight_sq.shape)

    # Initialize the touch arrays
    touch_void = np.zeros(weight_sq.shape)
    touch_solid = np.zeros(weight_sq.shape)
    touch_void_pos = np.zeros(weight_sq.shape)
    touch_void_neg = np.zeros(weight_sq.shape)
    touch_solid_pos = np.zeros(weight_sq.shape)
    touch_solid_neg = np.zeros(weight_sq.shape)
    
    # Define a mask for the brush
    r_c = int((brush_size - 1)/2)
    r_c_plus2 = int((brush_size + 1)/2)
    r_conv = int(brush_size + 2*r_c)
    r_conv2 = int(brush_size + 4*r_c)
    if dim == 1:
        brush_shape = np.ones(brush_size).astype(int)
        brush_shape_plus2 = np.ones(brush_size + 6).astype(int)
        brush_conv_shape = np.ones(r_conv)
        brush_conv_shape2 = np.ones(r_conv2)
    elif dim == 2:
        x, y = np.meshgrid(np.arange(brush_size), np.arange(brush_size))
        x_plus2, y_plus2 = np.meshgrid(np.arange(brush_size + 2), np.arange(brush_size + 2))
        if brush_type == 'circle':
            distance = np.sqrt((x - r_c) ** 2 + (y - r_c) ** 2)
            brush_shape = (distance <= r_c).astype(int)
            
            distance = np.sqrt((x_plus2 - r_c_plus2) ** 2 + (y_plus2 - r_c_plus2) ** 2)
            brush_shape_plus2 = (distance <= r_c_plus2).astype(int)
        elif brush_type == 'notched_sq':
            brush_shape = np.ones((brush_size, brush_size)).astype(int)
            brush_shape[0,0] = 0
            brush_shape[0,-1] = 0
            brush_shape[-1,0] = 0
            brush_shape[-1,-1] = 0
        
        brush_conv_shape = np.zeros((r_conv, r_conv))
        brush_conv_shape[r_c:-r_c,r_c:-r_c] = brush_shape
        brush_conv_shape = convolve(brush_conv_shape, brush_shape, mode='constant', cval=0)
        brush_conv_shape = np.where(brush_conv_shape > 0, 1, 0)
        
        brush_conv_shape2 = np.zeros((r_conv2, r_conv2))
        brush_conv_shape2[r_c:-r_c,r_c:-r_c] = brush_conv_shape
        brush_conv_shape2 = convolve(brush_conv_shape2, brush_shape, mode='constant', cval=0)
        brush_conv_shape2 = np.where(brush_conv_shape2 > 0, 1, 0)

    delta_array = np.zeros_like(weight_sq)
    delta_array[0,0] = 1
    reference_convolution0 = convolve(delta_array, brush_shape, mode='wrap').astype(bool)
    reference_convolution1 = convolve(delta_array, brush_conv_shape, mode='wrap').astype(bool)
    reference_convolution2 = convolve(delta_array, brush_conv_shape2, mode='wrap').astype(bool)

    # Compute the reward array
    if periodic:
        score_solid = convolve(weight_sq, brush_shape, mode='wrap')
    else:
        score_solid = convolve(weight_sq, brush_shape, mode='constant', cval=pad_value)
    score_void = -score_solid.copy()
    
    # Pre-fill touches and pixels
    score_solid_pre_fill = convolve(weight_sq, brush_shape_plus2, mode='wrap')
    touch_solid_pos[score_solid_pre_fill>=np.sum(brush_shape_plus2)] = 1
    touch_void_neg[score_solid_pre_fill>=np.sum(brush_shape_plus2)] = 1
    
    touch_solid_neg[score_solid_pre_fill<=-np.sum(brush_shape_plus2)] = 1
    touch_void_pos[score_solid_pre_fill<=-np.sum(brush_shape_plus2)] = 1
    
    if periodic:
        pix_solid += convolve(touch_solid_pos.astype(int), brush_shape, mode='wrap').astype(bool).astype(int)
        touch_solid += touch_solid_pos
        touch_void -= convolve(touch_solid_pos.astype(int), brush_conv_shape, mode='wrap').astype(bool).astype(int)
        
        pix_solid -= convolve(touch_solid_neg.astype(int), brush_shape, mode='wrap').astype(bool).astype(int)
        touch_void += touch_solid_neg
        touch_solid -= convolve(touch_solid_neg.astype(int), brush_conv_shape, mode='wrap').astype(bool).astype(int)
    else:
        pix_solid += convolve(touch_solid_pos.astype(int), brush_shape, mode='constant').astype(bool).astype(int)
        touch_solid += touch_solid_pos
        touch_void -= convolve(touch_solid_pos.astype(int), brush_conv_shape, mode='constant').astype(bool).astype(int)
        
        pix_solid -= convolve(touch_solid_neg.astype(int), brush_shape, mode='constant').astype(bool).astype(int)
        touch_void += touch_solid_neg
        touch_solid -= convolve(touch_solid_neg.astype(int), brush_conv_shape, mode='wrap').astype(bool).astype(int)
    
    if np.sum(pix_solid==0) > 0:
        try:
            pix_solid = main_loop(score_solid,
                                  score_void,
                                  pix_solid,
                                  pix_void,
                                  touch_solid,
                                  touch_void,
                                  reference_convolution0,
                                  reference_convolution1,
                                  reference_convolution2,
                                  symmetry,
                                  periodic,
                                  brush_size,
                                  dim,
                                  N_x,
                                  N_y,
                                  )
        except:
            np.savez(directory + '/debug_make_feasible', weight=weight)
            assert False

    pix_solid[pix_solid<0] = 0
    if not periodic:
        if dim == 1:
            pix_solid = pix_solid[brush_size:-brush_size]
        else:
            pix_solid = pix_solid[brush_size:-brush_size,brush_size:-brush_size]
    
    return pix_solid.reshape(-1)

def find_index_max(symmetry, periodic, brush_size, dim, N_x, N_y, score, touch, max_value):
    ind_temp = np.argwhere(np.abs(score - max_value) < 1e-4)
    if symmetry == 'None':
        for i in range(ind_temp.shape[0]):
            if dim == 1:
                if touch[ind_temp[i,0]] == 0:
                    return ind_temp[i,0]
            else:
                if touch[ind_temp[i,0],ind_temp[i,1]] == 0:
                    return tuple(ind_temp[i,:])
    
    elif symmetry == 'D1':
        half_x = np.floor(N_x/2 + 0.5) + 2*(1 - periodic)*brush_size
        
        for i in range(ind_temp.shape[0]):
            if ind_temp[i,0] <= half_x:
                if touch[ind_temp[i,0],ind_temp[i,1]] == 0:
                    return tuple(ind_temp[i,:])
    
    elif symmetry == 'D2':
        half_x = np.floor(N_x/2 + 0.5) + 2*(1 - periodic)*brush_size
        half_y = np.floor(N_y/2 + 0.5) + 2*(1 - periodic)*brush_size
        
        for i in range(ind_temp.shape[0]):
            if ind_temp[i,0] <= half_x and ind_temp[i,1] <= half_y:
                if touch[ind_temp[i,0],ind_temp[i,1]] == 0:
                    return tuple(ind_temp[i,:])
    
    elif symmetry == 'D4':
        half_y = np.floor(N_y/2 + 0.5) + 2*(1 - periodic)*brush_size
        
        for i in range(ind_temp.shape[0]):
            if ind_temp[i,0] <= ind_temp[i,1] and ind_temp[i,1] <= half_y:
                if touch[ind_temp[i,0],ind_temp[i,1]] == 0:
                    return tuple(ind_temp[i,:])

def main_loop(score_solid,
              score_void,
              pix_solid,
              pix_void,
              touch_solid,
              touch_void,
              reference_convolution0,
              reference_convolution1,
              reference_convolution2,
              symmetry,
              periodic,
              brush_size,
              dim,
              N_x,
              N_y,
              ):
    
    n_iter = 0
    while True:
        t1 = time.time()
        solid = False
        void = False
        
        max_solid, max_void = -1e10, -1e10

        # Find max scores for solid and void where touches have not been applied
        if np.sum(touch_solid == 0) > 0:
            max_solid = np.nanmax(score_solid[touch_solid == 0])
        
        if np.sum(touch_void == 0) > 0:
            max_void = np.nanmax(score_void[touch_void == 0])
        
        if max_solid > max_void:
            solid = True
            ind_max = find_index_max(symmetry, periodic, brush_size, dim, N_x, N_y, score_solid, touch_solid, max_solid)
        elif max_void > max_solid:
            void = True
            ind_max = find_index_max(symmetry, periodic, brush_size, dim, N_x, N_y, score_void, touch_void, max_void)
        else:
            # Random choice if max_solid == max_void
            if np.random.rand() > 0.5:
                solid = True
                ind_max = find_index_max(symmetry, periodic, brush_size, dim, N_x, N_y, score_solid, touch_solid, max_solid)
            else:
                void = True
                ind_max = find_index_max(symmetry, periodic, brush_size, dim, N_x, N_y, score_void, touch_void, max_void)
        if ind_max is None:
            pass
        
        n_iter += 1
        n_empty = np.sum(pix_solid==0)
        touch_solid, touch_void, pix_solid, pix_void = make_touch(solid,
                                                                  void,
                                                                  ind_max,
                                                                  touch_solid,
                                                                  touch_void,
                                                                  pix_solid,
                                                                  pix_void,
                                                                  reference_convolution0,
                                                                  reference_convolution1,
                                                                  symmetry,
                                                                  )
        
        if n_empty != np.sum(pix_solid==0):
            touch_solid, touch_void, pix_solid, pix_void = fill_required_pixels(ind_max,
                                                                                touch_solid,
                                                                                touch_void,
                                                                                pix_solid,
                                                                                pix_void,
                                                                                score_solid,
                                                                                score_void,
                                                                                reference_convolution0,
                                                                                reference_convolution1,
                                                                                reference_convolution2,
                                                                                symmetry,
                                                                                )
        
        t2 = time.time()
        if n_iter%10 == 0:
            print('Iter ' + str(n_iter) + ' | Unassigned Solid Pixels: ' + str(np.sum(pix_solid==0)) + ' | Unassigned Solid Touches: ' + str(np.sum(touch_solid==0)) + ' | Unassigned Void Touches: ' + str(np.sum(touch_void==0)) + ' | Time: ' + str(t2 - t1), flush=True)
        if np.sum(pix_solid==0) == 0:
            break
    
    return pix_solid

def fill_required_pixels(ind_max,
                         touch_solid,
                         touch_void,
                         pix_solid,
                         pix_void,
                         score_solid,
                         score_void,
                         reference_convolution0,
                         reference_convolution1,
                         reference_convolution2,
                         symmetry,
                         ):
    
    last_affected = np.roll(reference_convolution2, ind_max, axis=(0,1)).astype(bool)
    
    required = True
    while required:
        required = False
        
        pix_to_check = np.argwhere((pix_solid == 0)*last_affected)
        for n in pix_to_check:
            # if n[0] > 109 and n[0] < 114 and n[1] > 14 and n[1] < 24:
            #     pass
            required_mask = np.roll(reference_convolution0, n, axis=(0,1))
            
            if np.all(touch_solid[required_mask]):
                score_required = np.where(required_mask*(touch_void==0), score_void, np.nan)
                if not np.isnan(np.nanmax(score_required)):
                    ind_required_max = np.argwhere(score_required == np.nanmax(score_required))[0,:].astype(int)

                    touch_solid, touch_void, pix_solid, pix_void = make_touch(False,
                                                                              True,
                                                                              ind_required_max,
                                                                              touch_solid,
                                                                              touch_void,
                                                                              pix_solid,
                                                                              pix_void,
                                                                              reference_convolution0,
                                                                              reference_convolution1,
                                                                              symmetry,
                                                                              )
                    
                    last_affected += np.roll(reference_convolution2, ind_required_max, axis=(0,1))
                    
                    required = True
                    break
                else:
                    pass
                
            elif np.all(touch_void[required_mask]):
                score_required = np.where(required_mask*(touch_solid==0), score_solid, np.nan)
                if not np.isnan(np.nanmax(score_required)):
                    ind_required_max = np.argwhere(score_required == np.nanmax(score_required))[0,:].astype(int)

                    touch_solid, touch_void, pix_solid, pix_void = make_touch(True,
                                                                              False,
                                                                              ind_required_max,
                                                                              touch_solid,
                                                                              touch_void,
                                                                              pix_solid,
                                                                              pix_void,
                                                                              reference_convolution0,
                                                                              reference_convolution1,
                                                                              symmetry,
                                                                              )
                    
                    last_affected += np.roll(reference_convolution2, ind_required_max, axis=(0,1))
                    
                    required = True
                    break
                else:
                    pass
            else:
                last_affected[tuple(n)] = 0
        
    return touch_solid, touch_void, pix_solid, pix_void

def make_touch(solid,
               void,
               ind_touch,
               touch_solid,
               touch_void,
               pix_solid,
               pix_void,
               reference_convolution0,
               reference_convolution1,
               symmetry,
               ):
    
    if solid:
        touch_solid, touch_void = touch2pix(ind_touch,
                                            touch_solid,
                                            touch_void,
                                            pix_solid,
                                            pix_void,
                                            reference_convolution1,
                                            symmetry,
                                            )
    elif void:
        touch_void, touch_solid = touch2pix(ind_touch,
                                            touch_void,
                                            touch_solid,
                                            pix_void,
                                            pix_solid,
                                            reference_convolution1,
                                            symmetry,
                                            )

    pix_mask = np.roll(reference_convolution0, ind_touch, axis=(0,1))
    pix_solid[(pix_mask & (pix_solid == 0))] = (-1)**(solid + 1)
    
    if symmetry != 'None':
        pix_solid = apply_symmetry(symmetry, pix_solid, pix_solid.shape[0], pix_solid.shape[1])
    
    return touch_solid, touch_void, pix_solid, pix_void

def touch2pix(ind_touch,
              touch1,
              touch2,
              pix1,
              pix2,
              reference_convolution1,
              symmetry,
              ):
    
    touch_mask = np.roll(reference_convolution1, ind_touch, axis=(0,1))

    # Update touch1 to indicate a successful touch
    touch1[tuple(ind_touch)] = 1
    # Update touch2 to mark areas covered by the touch as invalid (-1)
    touch2[(touch_mask & (touch2 == 0))] = -1

    if symmetry != 'None':
        touch1 = apply_symmetry(symmetry, touch1, touch1.shape[0], touch1.shape[1])
        touch2 = apply_symmetry(symmetry, touch2, touch2.shape[0], touch1.shape[1])

    return touch1, touch2

def apply_symmetry(symmetry, x, N_x, N_y):
    if symmetry == 'None':
        x_sym = x.copy()
        
    elif symmetry == 'D1':
        x_half = x[:int(np.floor(N_x/2 + 0.5)),:]
        x_sym = np.zeros((N_x, N_y))
        
        if N_x % 2 == 0:
            x_sym[:int(N_x/2),:] = x_half
            x_sym[int(N_x/2):,:] = np.flipud(x_half)
        
        else:
            x_sym[:int((N_x+1)/2),:] = x_half
            x_sym[int((N_x+1)/2):,:] = np.flipud(x_half[:-1,:])
    
    elif symmetry == 'D2':
        x_quart = x[:int(np.floor(N_x/2 + 0.5)),:int(np.floor(N_y/2 + 0.5))]
        x_sym = np.zeros((N_x, N_y))
        
        if N_x % 2 == 0 and N_y % 2 == 0:
            x_sym[:int(N_x/2),:int(N_y/2)] = x_quart
            x_sym[int(N_x/2):,:int(N_y/2)] = np.flipud(x_quart)
            x_sym[:int(N_x/2),int(N_y/2):] = np.fliplr(x_quart)
            x_sym[int(N_x/2):,int(N_y/2):] = np.fliplr(np.flipud(x_quart))
        
        elif N_x % 2 == 1 and N_y % 2 == 0:
            x_sym[:int((N_x+1)/2),:int(N_y/2)] = x_quart
            x_sym[int((N_x+1)/2):,:int(N_y/2)] = np.flipud(x_quart[:-1,:])
            x_sym[:int((N_x+1)/2),int(N_y/2):] = np.fliplr(x_quart)
            x_sym[int((N_x+1)/2):,int(N_y/2):] = np.fliplr(np.flipud(x_quart[:-1,:]))
        
        elif N_x % 2 == 0 and N_y % 2 == 1:
            x_sym[:int(N_x/2),:int((N_y+1)/2)] = x_quart
            x_sym[int(N_x/2):,:int((N_y+1)/2)] = np.flipud(x_quart)
            x_sym[:int(N_x/2),int((N_y+1)/2):] = np.fliplr(x_quart[:,:-1])
            x_sym[int(N_x/2):,int((N_y+1)/2):] = np.fliplr(np.flipud(x_quart[:,:-1]))
        
        elif N_x % 2 == 1 and N_y % 2 == 1:
            x_sym[:int((N_x+1)/2),:int((N_y+1)/2)] = x_quart
            x_sym[int((N_x+1)/2):,:int((N_y+1)/2)] = np.flipud(x_quart[:-1,:])
            x_sym[:int((N_x+1)/2),int((N_y+1)/2):] = np.fliplr(x_quart[:,:-1])
            x_sym[int((N_x+1)/2):,int((N_y+1)/2):] = np.fliplr(np.flipud(x_quart[:-1,:-1]))

    elif symmetry == 'D4':
        if N_x % 2 == 0:
            x_triu = x[:int(N_x/2),:int(N_x/2)][np.triu_indices(int(N_x/2))]
            
            x_quart = np.zeros((int(N_x/2), int(N_x/2)))
            x_quart[np.triu_indices(int(N_x/2))] = x_triu
            x_quart = x_quart + x_quart.T - np.diag(np.diag(x_quart))
            
            x_sym = np.zeros((N_x, N_x))
            x_sym[:int(N_x/2),:int(N_x/2)] = x_quart
            x_sym[int(N_x/2):,:int(N_x/2)] = np.rot90(x_quart)
            x_sym[int(N_x/2):,int(N_x/2):] = np.rot90(x_quart, k=2)
            x_sym[:int(N_x/2),int(N_x/2):] = np.rot90(x_quart, k=-1)
        
        else:
            x_triu = x[:int((N_x+1)/2),:int((N_x+1)/2)][np.triu_indices(int((N_x+1)/2))]
            
            x_quart = np.zeros((int((N_x+1)/2), int((N_x+1)/2)))
            x_quart[np.triu_indices(int((N_x+1)/2))] = x_triu
            x_quart = x_quart + x_quart.T - np.diag(np.diag(x_quart))
            
            x_sym = np.zeros((N_x, N_x))
            x_sym[:int((N_x+1)/2),:int((N_x+1)/2)] = x_quart
            x_sym[int((N_x+1)/2):,:int((N_x+1)/2)] = np.rot90(x_quart)[1:,:]
            x_sym[int((N_x+1)/2):,int((N_x+1)/2):] = np.rot90(x_quart, k=2)[1:,1:]
            x_sym[:int((N_x+1)/2),int((N_x+1)/2):] = np.rot90(x_quart, k=-1)[:,1:]
    
    return x_sym