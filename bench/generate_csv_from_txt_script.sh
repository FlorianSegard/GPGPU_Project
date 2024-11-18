#!/bin/bash
for file in *.txt; do
    # Définir le nom de sortie
    output_file="${file%.txt}.csv"

    # Extraire uniquement les lignes après "Profiling result:" et supprimer les lignes non nécessaires
    awk '/"Type","Time\(%)"/ {flag=1; next} /MemsetD2D_BL/ {flag=0} flag {print}' "$file" > "$output_file"
done
