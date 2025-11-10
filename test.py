import os

EXCLUDE_DIRS = {'.venv','.idea', '.env', '__pycache__','.git','objects'}

def print_tree(root, prefix=""):
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            if name in EXCLUDE_DIRS:
                continue
            print(f"{prefix}├── {name}")
            print_tree(path, prefix + "│   ")
        else:
            print(f"{prefix}├── {name}")

with open("structure.txt", "w", encoding="utf-8") as f:
    from contextlib import redirect_stdout
    with redirect_stdout(f):
        print_tree(".")