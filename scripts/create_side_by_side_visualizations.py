import os
import cv2
import numpy as np
from pathlib import Path

def create_side_by_side_visualization(folder_path):
    """
    Creates a side-by-side visualization of three overlay images.
    
    Args:
        folder_path: Path to the folder containing the images
    """
    # Define the image names we want to combine
    image_names = [
        "left000000_2_overlay.png",
        "left000000_3_overlay.png", 
        "left000000_4_overlay.png"
    ]
    
    # Load the three images
    images = []
    for img_name in image_names:
        img_path = os.path.join(folder_path, img_name)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
            else:
                print(f"Warning: Could not load {img_path}")
        else:
            print(f"Warning: Image not found: {img_path}")
    
    if len(images) != 3:
        print(f"Error: Could not load all 3 images from {folder_path}")
        return None
    
    # Concatenate images horizontally
    side_by_side = np.hstack(images)
    
    # Save the combined image
    output_path = os.path.join(folder_path, "side_by_side.png")
    cv2.imwrite(output_path, side_by_side)
    
    return output_path

def main():
    # Base directory containing all the subfolders
    base_dir = "/local/home/dkorth/Projects/dynamic-scene-graphs/multirun/2025-07-24/14-53-49"
    
    # Get all subfolders (0-53)
    subfolders = []
    for i in range(54):
        subfolder_path = os.path.join(base_dir, str(i))
        if os.path.exists(subfolder_path):
            subfolders.append(subfolder_path)
    
    print(f"Found {len(subfolders)} subfolders to process")
    
    # Process each subfolder
    for i, subfolder in enumerate(subfolders):
        print(f"Processing subfolder {i+1}/{len(subfolders)}: {subfolder}")
        
        output_path = create_side_by_side_visualization(subfolder)
        
        if output_path:
            print(f"Created visualization: {output_path}")
        else:
            print(f"Failed to create visualization for {subfolder}")
    
    print("Processing complete!")

if __name__ == "__main__":
    main() 