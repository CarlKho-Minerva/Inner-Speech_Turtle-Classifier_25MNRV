import os
from pathlib import Path

def print_directory_structure(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = '│   ' * level
        print(f'{indent}└── {os.path.basename(root)}/')
        subindent = '│   ' * (level + 1)
        for f in files:
            print(f'{subindent}└── {f}')

if __name__ == '__main__':
    current_dir = os.getcwd()
    print(f"Directory structure for: {current_dir}\n")
    print_directory_structure(current_dir)
