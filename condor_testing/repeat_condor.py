import os
import time

for i in range(0, 5):
    cmd = "python3 repeat_script.py " + str(i)
    os.system(cmd)
    correct_output = False
    while not correct_output:
        with open('repeat.txt', 'r') as f:
            last_line = f.readlines()[-1]
            if last_line == str(str(i) + "\n"):
                correct_output = True
            else:
                time.sleep(.5)
    print("Ran")


