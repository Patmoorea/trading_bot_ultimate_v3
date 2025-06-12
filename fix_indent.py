#!/usr/bin/env python3
import re
from datetime import datetime

def fix_indentation(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    fixed_lines = []
    indent_level = 0
    in_class = False
    in_function = False
    
    for line in lines:
        # Suppression des espaces en début et fin
        stripped = line.strip()
        
        # Lignes vides
        if not stripped:
            fixed_lines.append('\n')
            continue
        
        # Gestion des blocs
        if stripped.startswith('class '):
            in_class = True
            in_function = False
            indent_level = 0
        elif stripped.startswith(('def ', 'async def ')):
            in_function = True
            indent_level = 1 if in_class else 0
        elif stripped.startswith(('if ', 'elif ', 'else:', 'try:', 'except ', 'finally:')):
            indent_level += 1
        elif stripped == 'except:' or stripped.startswith('except '):
            indent_level = max(1, indent_level)
        
        # Calcul de l'indentation
        indent = ' ' * (4 * indent_level)
        
        # Ajout de la ligne avec indentation
        if stripped.endswith(':'):
            fixed_lines.append(f"{indent}{stripped}\n")
            indent_level += 1
        else:
            fixed_lines.append(f"{indent}{stripped}\n")
        
        # Réduction de l'indentation après certains blocs
        if stripped in ['return', 'break', 'continue', 'pass']:
            indent_level = max(0, indent_level - 1)
        elif stripped.startswith(('except', 'else:', 'finally:')):
            indent_level = max(0, indent_level - 1)
    
    # Sauvegarde du fichier
    with open(filename, 'w') as file:
        file.writelines(fixed_lines)

if __name__ == '__main__':
    # Sauvegarde de backup
    with open('src/main.py', 'r') as source:
        with open('src/main.py.backup', 'w') as backup:
            backup.write(source.read())
            
    # Correction de l'indentation
    fix_indentation('src/main.py')
    print("Indentation corrigée! Une sauvegarde a été créée dans main.py.backup")
