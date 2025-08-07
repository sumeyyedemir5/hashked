import numpy as np

def save_binvox(voxel_data, filename):
    """
    voxel_data: 3D numpy bool array, voxel doluluk durumu
    filename: kaydedilecek .binvox dosya yolu
    """
    with open(filename, 'wb') as f:
        # Binvox header yaz
        f.write(b"#binvox 1\n")
        f.write(f"dim {voxel_data.shape[0]} {voxel_data.shape[1]} {voxel_data.shape[2]}\n".encode())
        f.write(b"translate 0 0 0\n")
        f.write(b"scale 1\n")
        f.write(b"data\n")
        
        # voxel_data bool array -> run-length encoding (RLE) ile yazılır
        flat_voxels = voxel_data.flatten(order='F')  # column-major order
        count = 0
        current = flat_voxels[0]
        
        for i in range(len(flat_voxels)):
            if flat_voxels[i] == current and count < 255:
                count += 1
            else:
                f.write(bytes([int(current), count]))
                current = flat_voxels[i]
                count = 1
        # Son kalanları yaz
        if count > 0:
            f.write(bytes([int(current), count]))
