import os
import shutil
from pathlib import Path
import logging

def cleanup_intermediate_files(intermediate_folder, force=False):
    """
    Delete all files and subdirectories in an intermediate file folder.
    
    Args:
        intermediate_folder (str or Path): Path to the intermediate files folder
        force (bool): If True, ignore errors and force deletion. Default False.
    
    Returns:
        bool: True if cleanup successful, False otherwise
    """
    try:
        folder_path = Path(intermediate_folder)
        
        # Check if folder exists
        if not folder_path.exists():
            logging.warning(f"Intermediate folder does not exist: {folder_path}")
            return True
        
        if not folder_path.is_dir():
            logging.error(f"Path is not a directory: {folder_path}")
            return False
        
        # Count items before deletion
        items_to_delete = list(folder_path.iterdir())
        item_count = len(items_to_delete)
        
        if item_count == 0:
            logging.info(f"Intermediate folder is already empty: {folder_path}")
            return True
        
        # Delete all contents
        deleted_count = 0
        for item in items_to_delete:
            try:
                if item.is_file():
                    item.unlink()  # Delete file
                    deleted_count += 1
                elif item.is_dir():
                    shutil.rmtree(item)  # Delete directory and contents
                    deleted_count += 1
            except Exception as e:
                if force:
                    logging.warning(f"Failed to delete {item}, continuing: {e}")
                    continue
                else:
                    logging.error(f"Failed to delete {item}: {e}")
                    return False
        
        logging.info(f"Successfully cleaned up {deleted_count}/{item_count} items from {folder_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")
        return False
