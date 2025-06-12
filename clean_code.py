#!/usr/bin/env python3
import re
from datetime import datetime

def fix_code(filename):
    # Timestamp actuel
    current_time = "2025-06-09 03:54:15"
    
    with open(filename, 'r') as file:
        content = file.read()
    
    # 1. Mise à jour du timestamp
    content = re.sub(
        r'CURRENT_TIME = "[^"]*"',
        f'CURRENT_TIME = "{current_time}"',
        content
    )
    
    # 2. Correction de l'indentation de base
    lines = content.split('\n')
    fixed_lines = []
    indent_level = 0
    in_class = False
    in_function = False
    
    for line in lines:
        stripped = line.strip()
        
        # Lignes vides
        if not stripped:
            fixed_lines.append('')
            continue
            
        # Ajustement du niveau d'indentation
        if stripped.startswith('class '):
            in_class = True
            in_function = False
            indent_level = 0
        elif stripped.startswith(('def ', 'async def ')):
            in_function = True
            indent_level = 1 if in_class else 0
        elif stripped.startswith(('if ', 'elif ', 'else:', 'try:', 'except ', 'finally:')):
            indent_level += 1
        
        # Calcul de l'indentation
        indent = ' ' * (4 * indent_level)
        
        # Ajout de la ligne indentée
        fixed_lines.append(indent + stripped)
        
        # Ajustement après les blocs
        if stripped.endswith(':'):
            indent_level += 1
        elif stripped in ('return', 'break', 'continue', 'pass'):
            indent_level = max(0, indent_level - 1)
        elif stripped.startswith(('except', 'else:', 'finally:')):
            indent_level = max(0, indent_level - 1)
    
    # 3. Sauvegarde du fichier corrigé
    with open(filename, 'w') as file:
        file.write('\n'.join(fixed_lines))

if __name__ == '__main__':
    fix_code('src/main.py')
    print("Code nettoyé et indenté correctement!")
