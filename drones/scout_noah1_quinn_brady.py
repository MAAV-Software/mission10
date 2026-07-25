import os
import cv2
import pathlib
import math
from ultralytics import YOLOWorld

# --- CONSTANTS ---
# Your specific Half-FOV angles (from your code snippet)
HALF_FOV_X = 32.5  # Degrees
HALF_FOV_Y = 25.5  # Degrees

def take_picture(img_dir):
    picture_and_locs = []

    # Updated simulator locations
    camera_locs = [
        (0, 6.0338, 2), (0, 5.0338, 2), (0, 4.0338, 2),
        (0, 3.0338, 2), (0, 2.0338, 2), (-0.5, 2.0338, 2),
        (-0.5, 3.0338, 2), (-0.5, 4.0338, 2), (-0.5, 5.0338, 2),
        (-0.5, 6.0338, 2), (-1, 6.0338, 2), (-1, 5.0338, 2),
        (-1, 4.0338, 2), (-1, 3.0338, 2), (-1, 2.0338, 2),
    ]

    picture_paths = []
    # Check if directory exists to avoid errors
    if os.path.exists(img_dir):
        with os.scandir(img_dir) as entries:
            # Sort by modification time to match the simulation order
            sorted_entries = sorted(entries, key=lambda e: e.stat().st_mtime)
            for entry in sorted_entries:
                if entry.name != ".DS_Store":
                    picture_paths.append(img_dir / entry.name)
    
    # Match locations to images
    limit = min(len(camera_locs), len(picture_paths))
    for i in range(limit):
        picture_and_locs.append((camera_locs[i], picture_paths[i]))
    
    return picture_and_locs

def map_0(drone_output, coords_list, model):
    """
    Processes images to find bounding boxes. 
    NOTE: Model is passed in as an argument to avoid re-loading it 15 times.
    """
    for entry in drone_output:
        camera_loc, image_path = entry
        
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            continue
            
        img_height, img_width, _ = image.shape
        
        # Run Prediction
        results = model.predict(image, conf=0.4, verbose=False)
        bounding_boxes = results[0].boxes

        # Process results
        for box in bounding_boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            cls = int(box.cls[0].item())
            class_name = model.names[cls]
            
            # Normalize coordinates (0 to 1) so we can do the math later
            norm_x_min = x_min / img_width
            norm_y_min = y_min / img_height
            norm_x_max = x_max / img_width
            norm_y_max = y_max / img_height
            
            # Save raw data to list
            # We save the camera location (x,y,z) and the box details
            coords_list.append((class_name, camera_loc[0], camera_loc[1], camera_loc[2], 
                                norm_x_min, norm_y_min, norm_x_max, norm_y_max))

def main():
    # --- PATH SETUP ---
    # NOTE: If using Colab, change this to pathlib.Path("/content")
    parent_dir = pathlib.Path(__file__).parent 
    img_dir = parent_dir.parent / "taken_images_vzsc"
    
    # --- LOAD MODEL ONCE ---
    weights_path = parent_dir.parent / "weights" / "blender-weights.pt"
    if weights_path.exists():
        model = YOLOWorld(weights_path)
    else:
        print("Weights not found, loading standard YOLO...")
        model = YOLOWorld('yolov8s-world.pt') 

    # --- EXECUTION ---
    coords_list = [] 
    drone_output = take_picture(img_dir)
    
    if not drone_output:
        print("No images found. Check your img_dir path.")
        return

    # 1. Detect objects (2D)
    map_0(drone_output, coords_list, model)
    
    print(f"Detected {len(coords_list)} objects. Calculating global coordinates...")

    # 2. Calculate Global Coordinates (3D)
    final_locations = []

    with open("results.txt", "w") as f:
    
        for item in coords_list:
            (class_name, cam_x, cam_y, cam_z, pic_x_min, pic_y_min, pic_x_max, pic_y_max) = item
            
            # --- MATH LOGIC ---
            
            # 1. Calculate the center of the mine in the image (Normalized 0.0 to 1.0)
            mine_center_x_norm = (pic_x_min + pic_x_max) / 2
            mine_center_y_norm = (pic_y_min + pic_y_max) / 2
            
            # 2. Calculate the total visible span (in meters) at this altitude
            # Formula: Span = 2 * Altitude * tan(half_angle)
            # Your specific code used '4' which implies Altitude(2) * 2. 
            # I made it dynamic (cam_z * 2) so it works if drone changes height.
            altitude = cam_z 
            
            visible_span_horizontal = 2 * altitude * math.tan(math.radians(HALF_FOV_X))
            visible_span_vertical = 2 * altitude * math.tan(math.radians(HALF_FOV_Y))
            
            # 3. Calculate deviation from center (in meters)
            # (Center of image is 0.5)
            # If mine is at 0.6, it is (0.1 * total_width) to the right
            offset_x_meters = (mine_center_x_norm - 0.5) * visible_span_horizontal
            offset_y_meters = (mine_center_y_norm - 0.5) * visible_span_vertical
            
            # 4. Apply to Global Camera Coordinates
            # Camera X + Offset X
            global_mine_x = cam_x + offset_x_meters
            
            # Camera Y - Offset Y
            # NOTE: In images, Y increases downwards. In 3D world, Y usually increases "forward" or "up".
            # If the mine is at the bottom of the picture (high Y pixel), it is "behind" the drone (lower Y world).
            # Hence, we SUBTRACT the Y offset.
            global_mine_y = cam_y - offset_y_meters
            
            final_locations.append((class_name, global_mine_x, global_mine_y, 0.0))
            
            f.write(f"Mine found at Global: X={global_mine_x:.4f}, Y={global_mine_y:.4f}\n")
        

if __name__ == "__main__":
    main()