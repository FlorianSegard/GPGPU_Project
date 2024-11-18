import re
from collections import defaultdict

file_path = 'gloab_env_ACET.txt'

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

data = read_file(file_path)

# Regex pattern to extract the category and the time in seconds
pattern = r"(\w+): (\d+\.\d+) seconds"

category_times = defaultdict(list)

matches = re.findall(pattern, data)

for category, time in matches:
    category_times[category].append(float(time))

statistics = {}

# Calculate the total time
total_time = sum(sum(times) for times in category_times.values())

# Compute statistics for each category
for category, times in category_times.items():
    max_time = max(times)
    min_time = min(times)
    avg_time = sum(times) / len(times)
    num_calls = len(times)
    total_category_time = sum(times)
    percentage_time = (total_category_time / total_time) * 100

    statistics[category] = {
        "Max": max_time,
        "Min": min_time,
        "Average": avg_time,
        "Number of calls": num_calls,
        "Total Time": total_category_time,
        "Percentage Time": percentage_time
    }

# Output statistics for each category
print(f"Total Time: {total_time:.6f} seconds\n")
for category, stats in statistics.items():
    print(f"Category: {category}")
    for stat, value in stats.items():
        print(f"  {stat}: {value:.6f}")
    print()
