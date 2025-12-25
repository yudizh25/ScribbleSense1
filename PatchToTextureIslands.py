import numpy as np
import math
import random
import cv2

def random_transform(tile):
    angle = random.choice([0, 90, 180, 270])
    if angle == 90:
        tile = cv2.rotate(tile, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        tile = cv2.rotate(tile, cv2.ROTATE_180)
    elif angle == 270:
        tile = cv2.rotate(tile, cv2.ROTATE_90_COUNTERCLOCKWISE)

    flip_code = random.choice([None, 0, 1])
    if flip_code is not None:
        tile = cv2.flip(tile, flip_code)
    return tile

def generate_masked_tiling(original_path, tile_path, block_w, block_h, gap=5, bg_tolerance=10):
    
    original = cv2.imread(original_path)
    texture = cv2.imread(tile_path)

    if original is None or texture is None:
        raise ValueError("Texture id None")

    H, W = original.shape[:2]
    base_tile = cv2.resize(texture, (block_w, block_h))

    bg_color_numpy = original[0, 0]
    

    bg_color_scalar = (int(bg_color_numpy[0]), int(bg_color_numpy[1]), int(bg_color_numpy[2]))
    
    print(f"Detected Background Color: {bg_color_scalar}")

    bg_color_array = np.full_like(original, bg_color_scalar, dtype=np.uint8)
    diff = cv2.absdiff(original, bg_color_array)

    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    _, content_mask = cv2.threshold(gray_diff, bg_tolerance, 255, cv2.THRESH_BINARY)

    tiled_layer = original.copy()

    step_x = block_w + gap
    step_y = block_h + gap
    nx = (W + gap) // step_x
    ny = (H + gap) // step_y

    offset_x = (W - (nx * step_x - gap)) // 2
    offset_y = (H - (ny * step_y - gap)) // 2

    print(f"Generating grid: {nx}x{ny} blocks...")

    for i in range(nx):
        for j in range(ny):
            x_start = offset_x + i * step_x
            y_start = offset_y + j * step_y
            
            if x_start >= W or y_start >= H: continue

            paste_w = min(block_w, W - x_start)
            paste_h = min(block_h, H - y_start)
            
            if paste_w <= 0 or paste_h <= 0: continue

            tile = random_transform(base_tile.copy())
            tile_cropped = tile[0:paste_h, 0:paste_w]

            tiled_layer[y_start:y_start+paste_h, x_start:x_start+paste_w] = tile_cropped

    content_mask_3ch = cv2.merge([content_mask, content_mask, content_mask])

    result = np.where(content_mask_3ch == 255, tiled_layer, original)

    return content_mask, result

if __name__ == "__main__":

    original_path = "/data/xxxx/textures/eastern_bunny_result.jpg" 
    
    tile_path = "/data/xxxx/local_texture.png"
    
    block_w, block_h = 32, 32
    gap_size = 5
    
    try:
        mask, final_result = generate_masked_tiling(
            original_path, tile_path, block_w, block_h, gap=gap_size, bg_tolerance=10
        )
        
        cv2.imwrite("mask_debug.png", mask)
        cv2.imwrite("result_masked.png", final_result)
        print("Done. Saved result_masked.png")
        
    except Exception as e:
        import traceback
        traceback.print_exc() 