import pandas as pd
import matplotlib.pyplot as plt

results = """
Filename:  base_model ACET.csv
Total time in s:  2.287
Filename:  base_model lil_clown_studio.csv
Total time in s:  6.923
Filename:  base_model Nuit_blanches.csv
Total time in s:  12.416
Filename: hyst_trace ACET.csv
Total time in s: 1.749
Filename:  hyst_trace lil_clown_studio.csv
Total time in s:  5.364
Filename:  hyst_trace Nuit_blanches.csv
Total time in s:  8.17
Filename:  rollback  lil_clown_studio.csv
Total time in s:  6.551
Filename:  rollback ACET.csv
Total time in s:  2.101
Filename:  rollback Nuit_blanches.csv
Total time in s:  11.58
Filename:  shared_hyst_erode_dilate ACET.csv
Total time in s:  2.055
Filename:  shared_hyst_erode_dilate lil_clown_studio.csv
Total time in s:  6.065
Filename:  shared_hyst_erode_dilate Nuit_blanches.csv
Total time in s:  11.71
Filename: shared_hyst_erode_dilate_opti ACET.csv
Total time in s:  1.749
Filename:  shared_hyst_erode_dilate_opti lil_clown_studio.csv
Total time in s:  5.34
Filename:  shared_hyst_erode_dilate_opti Nuit_blanches.csv
Total time in s:  10.121
"""

# Parse the results
lines = results.strip().split("\n")
data = {
    "Filename": [],
    "Method": [],
    "Total time in s": []
}

for i in range(0, len(lines), 2):  # Process in pairs (Filename, Total time in s)
    filename = lines[i].split(":", 1)[1].strip()
    time = float(lines[i + 1].split(":")[1].strip())
    method = filename.split()[0]  # Extract the method name
    data["Filename"].append(filename)
    data["Method"].append(method)
    data["Total time in s"].append(time)

# Create a DataFrame
df = pd.DataFrame(data)

# Define color palette
colors = {
    "base_model": "skyblue",
    "hyst_trace": "lightgreen",
    "rollback": "lightcoral",
    "shared_hyst_erode_dilate": "lightgoldenrodyellow",
    "shared_hyst_erode_dilate_opti": "lightpink",
}

# Map colors to methods
df["Color"] = df["Method"].map(colors)

# Create a horizontal bar chart
plt.figure(figsize=(12, 8))
plt.barh(df["Filename"], df["Total time in s"], color=df["Color"])

# Add labels and title
plt.xlabel("Total time in seconds")
plt.ylabel("Filename")
plt.title("Execution Time Grouped by Method")

# Add legend
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[method]) for method in colors]
plt.legend(handles, colors.keys(), title="Method", loc="lower right")

# Display the chart
plt.tight_layout()
plt.savefig("execution_time_by_method.png")
plt.show()
