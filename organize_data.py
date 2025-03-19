import os
import shutil
from pathlib import Path

def create_directories():
    """Create necessary subdirectories."""
    directories = [
        'data/Training/smoker',
        'data/Training/non_smoker',
        'data/Validation/smoker',
        'data/Validation/non_smoker',
        'data/Testing/smoker',
        'data/Testing/non_smoker'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def organize_images():
    """Organize images into appropriate subdirectories."""
    source_dirs = ['data/Training', 'data/Validation', 'data/Testing']

    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            print(f"Directory {source_dir} does not exist.")
            continue

        for filename in os.listdir(source_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            if filename.lower().startswith('smoking'):
                dest_class = 'smoker'
            elif filename.lower().startswith('notsmoking'):
                dest_class = 'non_smoker'
            else:
                print(f"Ignored file (unrecognized format): {filename}")
                continue

            source_path = os.path.join(source_dir, filename)
            dest_dir = os.path.join(source_dir, dest_class)
            dest_path = os.path.join(dest_dir, filename)

            try:
                shutil.move(source_path, dest_path)
                print(f"Moved: {filename} -> {dest_class}")
            except Exception as e:
                print(f"Error moving {filename}: {str(e)}")

def main():
    print("Creating subdirectories...")
    create_directories()

    print("\nOrganizing images...")
    organize_images()

    print("\nOrganization complete!")

if __name__ == "__main__":
    main()
