import csv
import sys
import random

def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        return lines

def mark_random(input_file, output_file, num=50):
    # Skip the labels
    lines = read_tsv(input_file)[1:]
    size = len(lines)
    rset = set(random.sample(range(size), num))
    with open(output_file, "w") as of:
        for i, line in enumerate(lines):
            oline = ''
            if i in rset:
                oline = 'r'
            oline += '\t'.join(line) + '\n'
            of.write(oline)

def generate_mlm_data(input_file, output_file):
    # Skip the labels
    lines = read_tsv(input_file)[1:]
    with open(output_file, "w") as of:
        for line in lines:
            of.write(line[3] + "\n")
            of.write(line[4] + "\n")

