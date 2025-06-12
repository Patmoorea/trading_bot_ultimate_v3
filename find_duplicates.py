import os
import hashlib
from collections import defaultdict

def file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

duplicates = defaultdict(list)
for root, _, files in os.walk('.'):
    for filename in files:
        if filename.endswith('.py'):
            path = os.path.join(root, filename)
            file_id = (filename, file_hash(path))
            duplicates[file_id].append(path)

print("Doublons potentiels trouvés:")
for (name, _), paths in duplicates.items():
    if len(paths) > 1:
        print(f"\nFichier: {name}")
        for p in paths:
            print(f"- {p}")
