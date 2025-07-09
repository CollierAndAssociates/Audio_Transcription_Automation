import os

# Directory containing your output files
output_dir = "./output"
merged_filename = "merged_output.txt"

with open(merged_filename, "w", encoding="utf-8") as merged_file:
    for filename in sorted(os.listdir(output_dir)):
        file_path = os.path.join(output_dir, filename)

        if os.path.isfile(file_path) and filename.endswith(".txt"):
            merged_file.write(f"{'='*80}\n")
            merged_file.write(f"FILE: {filename}\n")
            merged_file.write(f"{'='*80}\n\n")

            with open(file_path, "r", encoding="utf-8") as f:
                merged_file.write(f.read())
                merged_file.write("\n\n")  # Add spacing between files

print(f"Merged output written to: {merged_filename}")