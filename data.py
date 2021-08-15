import csv
from random import randint

f = open("data.csv", "w")

writer = csv.writer(f)

numcols = 10
numlabels = 3
numrows = 1000
cols = [f"col{x}" for x in range(10)]
labels = [f"label{x}" for x in range(numlabels)]


def build_row():
    return [randint(0, 10) for _ in range(numcols)] + [
        randint(0, 3) for _ in range(numlabels)
    ]


writer.writerow(cols + labels)

for _ in range(numrows):
    writer.writerow(build_row())
