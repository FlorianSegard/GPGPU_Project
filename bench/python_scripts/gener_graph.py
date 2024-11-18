import pandas as pd
import matplotlib.pyplot as plt
import os

files = os.listdir()
# Sélectionner uniquement les fichiers CSV
files = [file for file in files if file.endswith('.csv')]
files.sort()

for file in files:
    # Charger les données depuis le fichier CSV
    df = pd.read_csv(file)
    if file == "shared_hyst_erode_dilate lil_clown_studio.csv":
        continue
    
    
    
    # Nettoyer la colonne 'Unnamed: 7'
    renommages = {
        "hysteresis_kernel(ImageView<bool>, ImageView<bool>, int, int, bool*)": "Hysteresis Kernel",
        "[CUDA memcpy DtoH]": "Memcpy DtoH",
        "[CUDA memcpy HtoD]": "Memcpy HtoD",
        "check_background_kernel(ImageView<lab>, ImageView<lab>, ImageView<lab>, ImageView<int>, ImageView<float>, int, int, int)": "Check Background",
        "rgbtolab_converter_kernel(ImageView<rgb8>, ImageView<lab>, int, int)": "RGB to LAB Converter",
        "erode(ImageView<float>, ImageView<float>, int, int, int)": "Erosion",
        "dilate(ImageView<float>, ImageView<float>, int, int, int)": "Dilation",
        "hysteresis_thresholding(ImageView<float>, ImageView<bool>, int, int, float)": "Hysteresis Threshold",
        "red_mask_kernel(ImageView<bool>, ImageView<rgb8>, int, int)": "Red Mask Kernel",
        "ConvertNV12BLtoNV12": "NV12 Conversion",
        "[CUDA memset]": "CUDA Memset",
        "[CUDA memcpy DtoD]": "Memcpy DtoD"
    }
    df['Unnamed: 7'] = df['Unnamed: 7'].replace(renommages)
    
    # Sélectionner les colonnes pertinentes
    labels = df['Unnamed: 7']  # Noms des activités
    sizes = df['%']  # Pourcentages
    number_of_calls = df['Unnamed: 3']  # Nombre d'appels

    # Regrouper les activités en une catégorie 'Autres' si le pourcentage est inférieur à un seuil
    seuil = 2.0
    mask = sizes < seuil
    if mask.any():
        autres_label = "Autres"
        autres_size = sizes[mask].sum()
        autres_calls = number_of_calls[mask].sum()
        labels = pd.concat([labels[~mask], pd.Series([autres_label])], ignore_index=True)
        sizes = pd.concat([sizes[~mask], pd.Series([autres_size])], ignore_index=True)
        number_of_calls = pd.concat([number_of_calls[~mask], pd.Series([autres_calls])], ignore_index=True)

    # Créer le diagramme en camembert
    plt.figure(figsize=(12, 12))
    wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, 
                                       wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'linestyle': 'solid'})

    # Ajouter le nombre d'appels à côté de chaque pourcentage
    for i, autotext in enumerate(autotexts):
        # Ajouter le nombre d'appels à chaque pourcentage
        autotext.set_text(f"{autotext.get_text()}\n({number_of_calls[i]} appels)")
        autotext.set_fontsize(10)
        autotext.set_color('black')
    
    # Mettre un titre et ajuster l'aspect
    plt.title('Répartition des activités GPU' + f' ({file})', fontsize=15)
    plt.axis('equal')  # Pour garder un aspect circulaire

    # Enregistrer le graphique avec un nom de fichier sans l'extension .csv
    file = file.split('.')[0]
    plt.savefig(f'graph_{file}.png')
    #plt.show()  # Optionnel : afficher le graphique
