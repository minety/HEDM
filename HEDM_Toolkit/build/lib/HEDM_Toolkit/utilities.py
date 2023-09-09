import os
import shutil
import pkg_resources

def copy_demo_func():
    try:
        # Get the path of the resource files
        source_dir = pkg_resources.resource_filename('HEDM_Toolkit', 'Demo/')
        
        # Get the current working directory
        dest_dir = os.getcwd()

        # Check if the source directory exists
        if not os.path.exists(source_dir):
            print(f"Error: Source directory {source_dir} does not exist.")
            return

        # Use shutil module to copy files/directories
        for item in os.listdir(source_dir):
            s = os.path.join(source_dir, item)
            d = os.path.join(dest_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, False, None)
            else:
                shutil.copy2(s, d)
                
        print("Demo files copied successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
