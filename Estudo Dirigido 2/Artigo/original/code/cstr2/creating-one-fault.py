import sys
import os

if len(sys.argv) != 2:
    print("Usage: python script.py <fault_number>")
    sys.exit(1)

fault = int(sys.argv[1])

# Run the command
command = f'type "..\\cfg\\fault_{fault}.txt" | .\\cstr2.exe'
os.system(command)

# Run processX.py
os.system('python processX.py')

# Generate the new filename
old_filename = 'X.csv'
new_filename = f'X_{fault}.csv'

# Check if the file already exists and delete it
if os.path.exists(new_filename):
    os.remove(new_filename)

# Rename the file
os.rename(old_filename, new_filename)