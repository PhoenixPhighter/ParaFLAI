import sys

f = open("repeat.txt", "a")
f.write(sys.argv[1] + "\n")
f.close()