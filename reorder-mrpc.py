import csv
import argparse
import sys

MRPC_SIZE = 3668

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

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--original", default=None, type=str, required=True)
    parser.add_argument("--new", default=None, type=str, required=True)
    args = parser.parse_args()

    olines = read_tsv(args.original)
    with open(args.new, 'w') as of:
        for line in olines:
            outline = line[0] + '\t' + line[2] + '\t' + line[1] + '\t' + line[-1] + '\t' + line[-2]
            of.write(outline + "\n")

if __name__ == "__main__":
    main()

