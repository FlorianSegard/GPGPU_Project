import pandas as pd
import matplotlib.pyplot as plt
import os

files = os.listdir()
# Sélectionner uniquement les fichiers CSV
files = [file for file in files if file.endswith('.csv')]
for file in files:
    df = pd.read_csv(file)
    #print(df.columns)
    #print(df.head())
    if 's' not in df.columns:
        print("\n")
        print("\n")
        print('-------------')
        print("Column 's' not found in file: ", file)
        print('-------------')
        print("\n")
        print("\n")

    else :
        sum = df['s'].sum()
        print("Filename: ", file)
        print("Total time in s: ", round(sum, 3))

    