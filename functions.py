import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random



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
    # Initialize vegetation matrix with zeros
    v = np.zeros((rows, cols), dtype=float)
    # Set of all free cells as coordinate tuples
    free_cells_list = [(i, j) for i in range(rows) for j in range(cols)]
    free_cells_set = set(free_cells_list)
    nb_veg_final, nb_veg = int(rows * cols * ratio_v), 0
    # Values to sample from, positive densities only
    v_sample = [val for val in v_source.values() if val > 0]
    # Start at a random free cell
    idx, node = random.randrange(len(free_cells_list)), free_cells_list[idx]
    
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
        # Neighbors
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