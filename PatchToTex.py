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

def generate_texture_mask_with_colored_gaps(original_path, doodle_path, tile_path, block_w, block_h, gap=5):
 
    original = cv2.imread(original_path) 
    doodle = cv2.imread(doodle_path)     
    texture = cv2.imread(tile_path)      

    if original is None or doodle is None or texture is None:
        raise ValueError("Input error")

    base_tile = cv2.resize(texture, (block_w, block_h))

    
    diff = cv2.absdiff(original, doodle)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, global_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

    contours, *_ = cv2.findContours(global_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No scribbled area")
    

    x, y, W, H = cv2.boundingRect(max(contours, key=cv2.contourArea))
    print(f"Bounding Box: x={x}, y={y}, W={W}, H={H}")


    tile_mask_layer = np.zeros_like(global_mask) 
    tile_layer = np.zeros_like(original)

    step_x = block_w + gap
    step_y = block_h + gap
    nx = (W + gap) // step_x
    ny = (H + gap) // step_y
    
    start_offset_x = (W - (nx * step_x - gap)) // 2
    start_offset_y = (H - (ny * step_y - gap)) // 2

    for i in range(nx):
        for j in range(ny):
            x_start = x + start_offset_x + i * step_x
            y_start = y + start_offset_y + j * step_y
            x_end = x_start + block_w
            y_end = y_start + block_h

            if x_end > original.shape[1] or y_end > original.shape[0]:
                continue
            
            tile = random_transform(base_tile.copy())
            

            tile_layer[y_start:y_end, x_start:x_end] = tile

            tile_mask_layer[y_start:y_end, x_start:x_end] = 255


    final_tile_mask = cv2.bitwise_and(global_mask, tile_mask_layer)

    inv_tile_mask = cv2.bitwise_not(tile_mask_layer)
    gap_mask = cv2.bitwise_and(global_mask, inv_tile_mask)
    

    bg_mask = cv2.bitwise_not(global_mask)

    img_bg = cv2.bitwise_and(original, original, mask=bg_mask)
    

    img_gap = cv2.bitwise_and(doodle, doodle, mask=gap_mask)
    

    img_tile = cv2.bitwise_and(tile_layer, tile_layer, mask=final_tile_mask)

    result = cv2.add(img_bg, img_gap)
    result = cv2.add(result, img_tile)

    return global_mask, result


if __name__ == "__main__":

    original_path = "/data/xxxx/eastern_bunny_result.jpg" 
    doodle_path = "/data/xxxx/scribble_texture.png"
    tile_path = "/data/xxxx/local_texture.png"
    
    block_w, block_h = 32, 32
    gap_size = 5
    
    try:
        mask, result = generate_texture_mask_with_colored_gaps(
            original_path, doodle_path, tile_path, block_w, block_h, gap=gap_size
        )
        cv2.imwrite("result_colored_gaps.png", result)
        print("Done. Saved result_colored_gaps.png")
    except Exception as e:
        print(f"Error: {e}")