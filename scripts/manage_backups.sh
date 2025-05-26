#!/bin/bash

# Création d'un backup daté
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="organized_backups/$BACKUP_DATE"

# Créer la structure de dossiers
mkdir -p "$BACKUP_DIR"/{src,tests}

# Copier les fichiers importants
cp -r backup/src/strategies "$BACKUP_DIR/src/"
cp -r backup/src/technical "$BACKUP_DIR/src/"
cp -r backup/src/utils "$BACKUP_DIR/src/"
cp -r backup/src/visualization "$BACKUP_DIR/src/"
cp -r backup/tests_new "$BACKUP_DIR/tests/"

# Créer un fichier d'index
echo "Backup créé le: $(date)" > "$BACKUP_DIR/index.txt"
echo "Par: $(whoami)" >> "$BACKUP_DIR/index.txt"
echo "Contenu:" >> "$BACKUP_DIR/index.txt"
find "$BACKUP_DIR" -type f -not -name "index.txt" >> "$BACKUP_DIR/index.txt"

echo "Backup créé dans: $BACKUP_DIR"
