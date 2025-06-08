#!/usr/bin/env python3
import re
from datetime import datetime

def fix_indentation(filename):
    with open(filename, 'r') as file:
        content = file.read()
    
    # Mise à jour des timestamps
    current_time = "2025-06-08 07:00:40"
    content = re.sub(r'current_time = "[^"]*"', f'current_time = "{current_time}"', content)
    content = re.sub(r'CURRENT_TIME = "[^"]*"', f'CURRENT_TIME = "{current_time}"', content)
    
    # Mise à jour du current user
    content = re.sub(r'current_user = "[^"]*"', 'current_user = "Patmoorea"', content)
    content = re.sub(r'CURRENT_USER = "[^"]*"', 'CURRENT_USER = "Patmoorea"', content)
    
    # Correction des messages d'erreur
    error_template = '''            error_time = datetime.utcnow().strftime(r"%Y-%m-%d %H:%M:%S")
            error_message = f"""🚨 ERREUR CRITIQUE - BOT ARRÊTÉ
Date: {error_time} UTC
User: {self.current_user}
Erreur: {str(e)}
Action requise: Vérification manuelle nécessaire"""
            await self.telegram.send_message(error_message)
            logger.error(f"[{error_time}] Erreur: {str(e)}")
            sys.exit(1)'''
    
    # Remplacer tous les blocs d'erreur
    content = re.sub(
        r'await self\.telegram\.send_message\([^)]*\)[\s\S]*?sys\.exit\(1\)',
        error_template,
        content,
        flags=re.DOTALL
    )
    
    # Correction de l'indentation des classes et méthodes
    lines = content.split('\n')
    fixed_lines = []
    class_level = 0
    method_level = 0
    in_class = False
    in_method = False
    
    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            fixed_lines.append('')
            continue
            
        # Détection du niveau d'indentation
        if stripped.startswith('class '):
            in_class = True
            in_method = False
            class_level = 0
            fixed_lines.append(stripped)
            continue
            
        if stripped.startswith(('def ', 'async def ')):
            in_method = True
            method_level = 4 if in_class else 0
            fixed_lines.append(' ' * method_level + stripped)
            continue
            
        if in_method:
            fixed_lines.append(' ' * (method_level + 4) + stripped)
        elif in_class:
            fixed_lines.append(' ' * 4 + stripped)
        else:
            fixed_lines.append(stripped)
    
    # Sauvegarde du fichier corrigé
    with open(filename, 'w') as file:
        file.write('\n'.join(fixed_lines))

if __name__ == '__main__':
    fix_indentation('src/main.py')
