import pandas as pd
import matplotlib.pyplot as plt

"""
base model
rollback
Shared Hystérésis
Shared Erode/Dilate and hysteresis 
Shared Erode/Dilate and hysteresis modèle final

"""
results = """
Filename:  Base model ACET
Total time in s:  2.287
Filename:  Base model lil_clown_studio
Total time in s:  6.923
Filename:  Base model Nuit_blanches
Total time in s:  12.416
Filename:  Rollback lil_clown_studio
Total time in s:  6.551
Filename:  Rollback ACET
Total time in s:  2.101
Filename:  Rollback Nuit_blanches
Total time in s:  11.58
Filename: Shared Hystérésis ACET
Total time in s: 1.749
Filename:  Shared Hystérésis lil_clown_studio
Total time in s:  5.364
Filename:  Shared Hystérésis Nuit_blanches
Total time in s:  8.17
Filename:  Shared Erode/Dilate and hysteresis ACET
Total time in s:  2.055
Filename:  Shared Erode/Dilate and hysteresis lil_clown_studio
Total time in s:  6.065
Filename:  Shared Erode/Dilate and hysteresis Nuit_blanches
Total time in s:  11.71
Filename: Modèle final ACET
Total time in s:  1.749
Filename:  Modèle final lil_clown_studio
Total time in s:  5.34
Filename:  Modèle final Nuit_blanches
Total time in s:  10.121
"""

# Parse the results
lines = results.strip().split("\n")
data = {
    "Filename": [],
    "Sample": [],
    "Total time in s": []
}

for i in range(0, len(lines), 2):  # Process in pairs (Filename, Total time in s)
    filename = lines[i].split(":", 1)[1].strip()
    time = float(lines[i + 1].split(":")[1].strip())
    # Extract the sample name from the filename (e.g., ACET, lil_clown_studio)
    sample = filename.split()[-1].replace("", "")
    data["Filename"].append(filename)
    data["Sample"].append(sample)
    data["Total time in s"].append(time)

# Create a DataFrame
df = pd.DataFrame(data)

# Define color palette
colors = {
    "ACET": "skyblue",
    "lil_clown_studio": "lightgreen",
    "Nuit_blanches": "lightcoral",
}

# Map colors to samples
df["Color"] = df["Sample"].map(colors)
# Sort the DataFrame by the 'Sample' column to group samples together
df = df.sort_values(by="Sample")

# Create a horizontal bar chart with grouped data
plt.figure(figsize=(12, 8))
plt.barh(df["Filename"], df["Total time in s"], color=df["Color"])

# Add labels and title
plt.xlabel("Total time in seconds")
plt.ylabel("Filename")
plt.title("Execution Time Grouped by Sample")

# Add legend
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[sample]) for sample in colors]
plt.legend(handles, colors.keys(), title="Sample", loc="lower right")

# Display the chart
plt.tight_layout()
plt.savefig("grouped_chart.png")
plt.show()




