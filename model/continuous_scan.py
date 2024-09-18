import torch
import numpy as np

#################################################################################
#               Mamba scan functions that preserve image continuity             #
#################################################################################

def get_continuous_paths(N):
    # Note that N is always even since we use image resolution of 256, 512, 1024 with the SD VAE encoder
    """

    for odd N, use 
    [
        (0, 0, 1, 1),
        (N - 1, N - 1, -1, -1),
        (N - 1, 0, -1, 1),
        (0, N - 1, 1, -1),
    ]
    """
    paths_lr = []
    reverse_lr = []
    for start_row, start_col, dir_row, dir_col in [
        (0, 0, 1, 1),
        (N - 1, 0, -1, 1),
    ]:
        path = lr_tranverse(N, start_row, start_col, dir_row, dir_col)
        paths_lr.append(path)
        reverse_lr.append(reverse_permut(path))
    
    
    """
    to scan all 4 tb direction for even N, use
    for start_row, start_col, dir_row, dir_col in [
        (0, 0, 1, 1),
        (N - 1, 0, -1, 1),
        (0, N - 1, 1, -1),
        (N - 1, N - 1, -1, -1),
    ]:
    
    """
    paths_tb = []
    reverse_tb = []
    for start_row, start_col, dir_row, dir_col in [
        (N - 1, 0, -1, 1),
        (N - 1, N - 1, -1, -1),
    ]:
        path =tb_tranverse(N, start_row, start_col, dir_row, dir_col)
        paths_tb.append(path)
        reverse_tb.append(reverse_permut(path))
    
    
    return paths_lr, paths_tb, reverse_lr, reverse_tb
    

def lr_tranverse(N,start_row=0, start_col=0, dir_row=1, dir_col=1):
    path = []
    for i in range(N):
        for j in range(N):
            # If the row number is even, move right; otherwise, move left
            col = j if i % 2 == 0 else N - 1 - j
            path.append((start_row + dir_row * i) * N + start_col + dir_col * col)
    return path

def tb_tranverse(N, start_row=0, start_col=0, dir_row=1, dir_col=1):
    path = []
    for j in range(N):
        for i in range(N):
            # If the column number is even, move down; otherwise, move up
            row = i if j % 2 == 0 else N - 1 - i
            path.append((start_row + dir_row * row) * N + start_col + dir_col * j)
    return path

def reverse_permut(permutation):
    n = len(permutation)
    reverse = [0] * n
    for i in range(n):
        reverse[permutation[i]] = i
    return reverse

def cross_scan_continuous(x, num_scans =4):
    B, C, W, H = x.size()
    N= W
    
    xs = x.new_empty((B, num_scans, C, H * W))
    
    paths_lr, paths_tb, reverse_lr, reverse_tb = get_continuous_paths(N)
    paths_lr = torch.tensor(paths_lr, device=x.device, dtype=torch.long)
    paths_tb = torch.tensor(paths_tb, device=x.device, dtype=torch.long)
    reverse_lr = torch.tensor(reverse_lr, device=x.device, dtype=torch.long)
    reverse_tb = torch.tensor(reverse_tb, device=x.device, dtype=torch.long)
    
    for i in range(paths_lr.size(0)):
        xs[:, i] = torch.index_select(x.flatten(2,3), -1, paths_lr[i])
        
    for i in range(paths_tb.size(0)):
        xs[:, i+num_scans//2] = torch.index_select(x.flatten(2,3), -1, paths_tb[i])
    
    
    return xs, paths_lr, paths_tb, reverse_lr, reverse_tb
    
def cross_merge_continuous(ys, paths_lr, paths_tb, reverse_lr, reverse_tb):
    
    B, K, D, W, H = ys.shape
    L = W*H
    ys = ys.view(B, K, D, -1)
    ys = ys.permute(0,2,1,3) # B, D, K, L
    
    corresponding_scan_paths = torch.concat([reverse_lr,reverse_tb], dim=0).view(1,1,K,L)
    corresponding_scan_paths = corresponding_scan_paths.repeat(B,D,1,1)
    y = torch.gather(ys, -1, corresponding_scan_paths)
    y = torch.sum(y,dim=2) # B, K, L
    
    return y    