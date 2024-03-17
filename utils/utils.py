import os 


def create_folders_if_not_exist(path, all = False):
    """
    Given a path, checks if each folder in the path exists, and creates them if they don't.

    Args:
        path (str): The path to check and create folders for.
    """
    # Split the path into individual folders
    folders = path.split('/')
    
    # Initialize a variable to build the current folder path
    current_folder = ''
    
    # Iterate through the folders and check/create them
    for idx, folder in enumerate(folders):
        if not(all) and idx == len(folders) - 1:
            break
        current_folder = os.path.join(current_folder, folder)
        
        
        # Check if the current folder exists
        if not os.path.exists(current_folder):
            # If it doesn't exist, create it
            os.makedirs(current_folder)

            
