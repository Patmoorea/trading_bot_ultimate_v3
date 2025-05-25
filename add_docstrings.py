import ast

def add_docstrings(filepath):
    with open(filepath, 'r+') as f:
        tree = ast.parse(f.read())
        # Implémentation à compléter
