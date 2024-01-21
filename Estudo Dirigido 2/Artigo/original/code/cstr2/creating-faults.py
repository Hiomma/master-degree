import os

# Define the range of faults (1 to 22)
start_fault = 1
end_fault = 22

# Iterate over the range of faults
for fault in range(start_fault, end_fault + 1):
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