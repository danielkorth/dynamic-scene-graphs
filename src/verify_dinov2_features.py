import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import shutil

def load_obj_points_file(obj_points_file, max_objects=10):
    """
    Load a single obj_points file.
    
    Args:
        obj_points_file (str): Path to obj_points file
        max_objects (int): Maximum number of objects to load (randomly sampled if more). If None, load all.
        
    Returns:
        dict: Dictionary mapping obj_id to crop image and dinov2_features
    """
    obj_data = {}
    
    print(f"Loading obj_points file: {obj_points_file}")
    
    obj_points = np.load(obj_points_file, allow_pickle=True).item()
    
    # Collect all objects with DINOv2 features
    valid_objects = []
    for obj_id, data in obj_points.items():
        if 'dinov2_features' in data and data['dinov2_features'] is not None:
            valid_objects.append((obj_id, data))
    
    # Randomly sample if we have more than max_objects (only if max_objects is specified)
    if max_objects is not None and len(valid_objects) > max_objects:
        import random
        random.shuffle(valid_objects)
        valid_objects = valid_objects[:max_objects]
        print(f"Randomly sampled {max_objects} objects from {len(obj_points)} total objects")
    
    # Create the final dictionary
    for obj_id, data in valid_objects:
        unique_key = f"obj_{obj_id}"
        
        obj_data[unique_key] = {
            'crop': data['crop'],
            'dinov2_features': data['dinov2_features'],
            'obj_id': obj_id
        }
    
    print(f"Loaded {len(obj_data)} objects with DINOv2 features")
    return obj_data

def load_obj_points_history(obj_points_dir):
    """
    Load all obj_points files from the history folder.
    
    Args:
        obj_points_dir (str): Path to obj_points_history folder
        
    Returns:
        dict: Dictionary mapping obj_id to crop image and dinov2_features
    """
    obj_data = {}
    
    # Find all obj_points files
    obj_points_files = glob.glob(os.path.join(obj_points_dir, "obj_points_*.npy"))
    obj_points_files.sort()
    
    print(f"Found {len(obj_points_files)} obj_points files")
    
    for file_path in obj_points_files:
        obj_points = np.load(file_path, allow_pickle=True).item()
        
        for obj_id, data in obj_points.items():
            if 'dinov2_features' in data and data['dinov2_features'] is not None:
                # Create unique key for each object across time
                unique_key = f"obj_{obj_id}_frame_{os.path.basename(file_path).split('_')[-1].split('.')[0]}"
                
                obj_data[unique_key] = {
                    'crop': data['crop'],
                    'dinov2_features': data['dinov2_features'],
                    'obj_id': obj_id,
                    'frame': os.path.basename(file_path).split('_')[-1].split('.')[0]
                }
    
    print(f"Loaded {len(obj_data)} objects with DINOv2 features")
    return obj_data

def calculate_similarity_matrix(obj_data):
    """
    Calculate cosine similarity matrix between all DINOv2 features.
    
    Args:
        obj_data (dict): Dictionary of object data with features
        
    Returns:
        tuple: (similarity_matrix, object_keys)
    """
    # Extract features and keys
    object_keys = list(obj_data.keys())
    features = np.array([obj_data[key]['dinov2_features'].flatten() for key in object_keys])
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(features)
    
    return similarity_matrix, object_keys

def calculate_cross_similarity_matrix(obj_data1, obj_data2):
    """
    Calculate cosine similarity matrix between objects from two different files.
    
    Args:
        obj_data1 (dict): Dictionary of object data from first file
        obj_data2 (dict): Dictionary of object data from second file
        
    Returns:
        tuple: (cross_similarity_matrix, keys1, keys2)
    """
    # Extract features and keys from both files
    keys1 = list(obj_data1.keys())
    keys2 = list(obj_data2.keys())
    
    features1 = np.array([obj_data1[key]['dinov2_features'].flatten() for key in keys1])
    features2 = np.array([obj_data2[key]['dinov2_features'].flatten() for key in keys2])
    
    # Calculate cross-similarity matrix (file1 objects vs file2 objects)
    cross_similarity_matrix = cosine_similarity(features1, features2)
    
    return cross_similarity_matrix, keys1, keys2

def find_top_k_similar_pairs(cross_similarity_matrix, keys1, keys2, obj_data1, obj_data2, k=10):
    """
    Find top-K most similar pairs between objects from two files.
    
    Args:
        cross_similarity_matrix (np.ndarray): Cross-similarity matrix
        keys1 (list): Object keys from first file
        keys2 (list): Object keys from second file
        obj_data1 (dict): Object data from first file
        obj_data2 (dict): Object data from second file
        k (int): Number of top pairs to return
        
    Returns:
        list: List of tuples (similarity, key1, key2, obj_id1, obj_id2)
    """
    # Get all similarity values with their indices
    similarities = []
    for i, key1 in enumerate(keys1):
        for j, key2 in enumerate(keys2):
            similarity = cross_similarity_matrix[i, j]
            obj_id1 = obj_data1[key1]['obj_id']
            obj_id2 = obj_data2[key2]['obj_id']
            similarities.append((similarity, key1, key2, obj_id1, obj_id2))
    
    # Sort by similarity (descending) and return top-k
    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities[:k]

def create_similarity_visualization(similarity_matrix, object_keys, obj_data, output_dir):
    """
    Create visualizations showing crops and their similarities.
    
    Args:
        similarity_matrix (np.ndarray): Similarity matrix
        object_keys (list): List of object keys
        obj_data (dict): Object data dictionary
        output_dir (str): Output directory for visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a large figure with subplots
    n_objects = len(object_keys)
    fig, axes = plt.subplots(n_objects, n_objects, figsize=(3*n_objects, 3*n_objects))
    
    if n_objects == 1:
        axes = axes.reshape(1, 1)
    
    # Plot each cell
    for i, key_i in enumerate(object_keys):
        for j, key_j in enumerate(object_keys):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: show the crop image
                crop = obj_data[key_i]['crop']
                if crop is not None:
                    ax.imshow(crop)
                    ax.set_title(f"{key_i}\n(obj_{obj_data[key_i]['obj_id']})", fontsize=8)
                ax.axis('off')
            else:
                # Off-diagonal: show similarity score and small crop images
                similarity = similarity_matrix[i, j]
                
                # Create a colored background based on similarity
                ax.set_facecolor(plt.cm.RdYlBu_r(similarity))
                
                # Add similarity text
                ax.text(0.5, 0.5, f'{similarity:.3f}', 
                       ha='center', va='center', fontsize=10, fontweight='bold',
                       transform=ax.transAxes)
                
                # Add small crop images in corners
                if obj_data[key_i]['crop'] is not None:
                    ax.imshow(obj_data[key_i]['crop'], extent=[0, 0.4, 0, 0.4])
                if obj_data[key_j]['crop'] is not None:
                    ax.imshow(obj_data[key_j]['crop'], extent=[0.6, 1, 0.6, 1])
                
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_matrix_visualization.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, 
                xticklabels=[f"obj_{obj_data[key]['obj_id']}" for key in object_keys],
                yticklabels=[f"obj_{obj_data[key]['obj_id']}" for key in object_keys],
                annot=True, fmt='.3f', cmap='RdYlBu_r', center=0.5)
    plt.title('DINOv2 Feature Similarity Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create crop grid
    n_cols = min(5, n_objects)
    n_rows = (n_objects + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, key in enumerate(object_keys):
        row = idx // n_cols
        col = idx % n_cols
        
        if n_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]
        
        crop = obj_data[key]['crop']
        if crop is not None:
            ax.imshow(crop)
            ax.set_title(f"{key}\n(obj_{obj_data[key]['obj_id']})", fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(n_objects, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        if n_rows == 1:
            axes[col].axis('off')
        else:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_crops_grid.png'), dpi=150, bbox_inches='tight')
    plt.close()

def create_cross_similarity_visualization(cross_similarity_matrix, keys1, keys2, obj_data1, obj_data2, top_k_pairs, output_dir):
    """
    Create visualizations for cross-file similarity comparisons.
    
    Args:
        cross_similarity_matrix (np.ndarray): Cross-similarity matrix
        keys1 (list): Object keys from first file
        keys2 (list): Object keys from second file
        obj_data1 (dict): Object data from first file
        obj_data2 (dict): Object data from second file
        top_k_pairs (list): Top-K similar pairs
        output_dir (str): Output directory for visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create cross-similarity heatmap
    plt.figure(figsize=(max(12, len(keys2)), max(10, len(keys1))))
    sns.heatmap(cross_similarity_matrix, 
                xticklabels=[f"File2_obj_{obj_data2[key]['obj_id']}" for key in keys2],
                yticklabels=[f"File1_obj_{obj_data1[key]['obj_id']}" for key in keys1],
                annot=True, fmt='.3f', cmap='RdYlBu_r', center=0.5)
    plt.title('Cross-File DINOv2 Feature Similarity Matrix\n(File1 objects vs File2 objects)')
    plt.xlabel('File 2 Objects')
    plt.ylabel('File 1 Objects')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_similarity_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create top-K similar pairs visualization
    if len(top_k_pairs) > 0:
        n_pairs = min(len(top_k_pairs), 10)  # Show max 10 pairs
        fig, axes = plt.subplots(n_pairs, 3, figsize=(12, 4*n_pairs))
        
        if n_pairs == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (similarity, key1, key2, obj_id1, obj_id2) in enumerate(top_k_pairs[:n_pairs]):
            # File 1 crop
            crop1 = obj_data1[key1]['crop']
            if crop1 is not None:
                axes[idx, 0].imshow(crop1)
            axes[idx, 0].set_title(f'File1: obj_{obj_id1}')
            axes[idx, 0].axis('off')
            
            # Similarity score
            axes[idx, 1].text(0.5, 0.5, f'Similarity:\n{similarity:.4f}', 
                             ha='center', va='center', fontsize=16, fontweight='bold',
                             transform=axes[idx, 1].transAxes)
            axes[idx, 1].set_facecolor(plt.cm.RdYlBu_r(similarity))
            axes[idx, 1].axis('off')
            
            # File 2 crop
            crop2 = obj_data2[key2]['crop']
            if crop2 is not None:
                axes[idx, 2].imshow(crop2)
            axes[idx, 2].set_title(f'File2: obj_{obj_id2}')
            axes[idx, 2].axis('off')
        
        plt.suptitle(f'Top {n_pairs} Most Similar Object Pairs', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_k_similar_pairs.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # Create separate grids for each file's objects
    for file_idx, (obj_data, keys, file_name) in enumerate([(obj_data1, keys1, "File1"), (obj_data2, keys2, "File2")]):
        n_objects = len(keys)
        n_cols = min(5, n_objects)
        n_rows = (n_objects + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_objects == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, key in enumerate(keys):
            if n_objects == 1:
                ax = axes[0]
            elif n_rows == 1:
                ax = axes[0, idx]
            else:
                row = idx // n_cols
                col = idx % n_cols
                ax = axes[row, col]
            
            crop = obj_data[key]['crop']
            if crop is not None:
                ax.imshow(crop)
                ax.set_title(f"{key}\n(obj_{obj_data[key]['obj_id']})", fontsize=10)
            ax.axis('off')
        
        # Hide empty subplots
        for idx in range(n_objects, n_rows * n_cols):
            if n_rows == 1:
                if n_objects > 1:
                    axes[0, idx].axis('off')
            else:
                row = idx // n_cols
                col = idx % n_cols
                axes[row, col].axis('off')
        
        plt.suptitle(f'{file_name} Objects', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{file_name.lower()}_crops_grid.png'), dpi=150, bbox_inches='tight')
        plt.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify DINOv2 features by calculating similarity matrix')
    parser.add_argument('input_path', type=str, help='Path to obj_points_history folder or specific obj_points file')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for visualizations')
    parser.add_argument('--single_file', action='store_true', help='Treat input as a single obj_points file instead of a directory')
    parser.add_argument('--file2', type=str, default=None, help='Second obj_points file for cross-file similarity comparison')
    parser.add_argument('--max_objects', type=int, default=None, help='Maximum number of objects to load from each file')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top similar pairs to show in cross-file comparison')
    
    args = parser.parse_args()
    
    # Create temporary output directory if not specified
    if args.output_dir is None:
        # temp_dir = tempfile.mkdtemp(prefix='dinov2_verification_')
        # print(f"Using temporary directory: {temp_dir}")
        temp_dir = "tmp"
    else:
        temp_dir = args.output_dir
        os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Check if cross-file comparison is requested
        if args.file2 is not None:
            # Cross-file comparison mode
            print("Cross-file comparison mode enabled")
            print(f"File 1: {args.input_path}")
            print(f"File 2: {args.file2}")
            
            # Load both files
            print("Loading obj_points from file 1...")
            obj_data1 = load_obj_points_file(args.input_path, max_objects=args.max_objects)
            
            print("Loading obj_points from file 2...")
            obj_data2 = load_obj_points_file(args.file2, max_objects=args.max_objects)
            
            if len(obj_data1) == 0 or len(obj_data2) == 0:
                print("No objects with DINOv2 features found in one or both files!")
                return
            
            # Calculate cross-similarity matrix
            print("Calculating cross-similarity matrix...")
            cross_similarity_matrix, keys1, keys2 = calculate_cross_similarity_matrix(obj_data1, obj_data2)
            
            # Find top-K similar pairs
            print(f"Finding top-{args.top_k} most similar pairs...")
            top_k_pairs = find_top_k_similar_pairs(cross_similarity_matrix, keys1, keys2, obj_data1, obj_data2, k=args.top_k)
            
            # Print statistics
            print(f"Cross-similarity matrix shape: {cross_similarity_matrix.shape}")
            print(f"File 1 objects: {len(keys1)}")
            print(f"File 2 objects: {len(keys2)}")
            print(f"Mean cross-similarity: {np.mean(cross_similarity_matrix):.3f}")
            print(f"Std cross-similarity: {np.std(cross_similarity_matrix):.3f}")
            print(f"Min cross-similarity: {np.min(cross_similarity_matrix):.3f}")
            print(f"Max cross-similarity: {np.max(cross_similarity_matrix):.3f}")
            
            # Print top-K pairs
            print(f"\nTop {len(top_k_pairs)} most similar pairs:")
            for i, (similarity, key1, key2, obj_id1, obj_id2) in enumerate(top_k_pairs):
                print(f"  {i+1}. File1_obj_{obj_id1} <-> File2_obj_{obj_id2}: {similarity:.4f}")
            
            # Create visualizations
            print("Creating cross-file visualizations...")
            create_cross_similarity_visualization(cross_similarity_matrix, keys1, keys2, obj_data1, obj_data2, top_k_pairs, temp_dir)
            
        else:
            # Single file mode (original functionality)
            print("Loading obj_points...")
            if args.single_file or os.path.isfile(args.input_path):
                # Load single file
                obj_data = load_obj_points_file(args.input_path, max_objects=args.max_objects)
            else:
                # Load from directory (use first file)
                obj_points_files = glob.glob(os.path.join(args.input_path, "obj_points_*.npy"))
                obj_points_files.sort()
                if len(obj_points_files) == 0:
                    print("No obj_points files found in directory!")
                    return
                print(f"Using first file: {obj_points_files[0]}")
                obj_data = load_obj_points_file(obj_points_files[0], max_objects=args.max_objects)
            
            if len(obj_data) == 0:
                print("No objects with DINOv2 features found!")
                return
            
            # Calculate similarity matrix
            print("Calculating similarity matrix...")
            similarity_matrix, object_keys = calculate_similarity_matrix(obj_data)
            
            # Print some statistics
            print(f"Similarity matrix shape: {similarity_matrix.shape}")
            print(f"Mean similarity: {np.mean(similarity_matrix):.3f}")
            print(f"Std similarity: {np.std(similarity_matrix):.3f}")
            print(f"Min similarity: {np.min(similarity_matrix):.3f}")
            print(f"Max similarity: {np.max(similarity_matrix):.3f}")
            
            # Create visualizations
            print("Creating visualizations...")
            create_similarity_visualization(similarity_matrix, object_keys, obj_data, temp_dir)
        
        print(f"Visualizations saved to: {temp_dir}")
        print("Files created:")
        for file in os.listdir(temp_dir):
            print(f"  - {file}")
            
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        if args.output_dir is None:
            print(f"\nTemporary directory will be cleaned up: {temp_dir}")
            print("To keep the files, copy them to a permanent location.")

if __name__ == "__main__":
    main() 