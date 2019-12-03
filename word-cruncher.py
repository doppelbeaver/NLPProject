import os
import csv
import sys
import argparse

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
    parser.add_argument("--removed_dir", default=None, type=str, required=True)
    arg = parser.parse_args()

    olines = read_tsv(args.original)
    guid_to_textb = {}
    for line in olines:
        guid_to_textb[line[1]] = line[3].lower()

    ol = []
    with open(args.original) as of:
        line = of.readline()
        while line:
            guid = line.split("\t")[1]
            ol.append(guid)
            line = of.readline()

    counts_filepath = os.path.join(args.removed_dir, "counts.tsv")
    print("Word\t\tTotal\tWorse\tBetter")
    with open(counts_filepath) as cf:
        line = cf.readline()
        fwd_idx = 0
        while line:
            split_line = line.split("\t")
            word = split_line[0]
            count = split_line[1]
            om = set()
            for guid in ol:
                textb = guid_to_textb[guid]
                words = textb.split(" ")
                if word in words:
                    om.add(guid)

            suffix = word
            if '/' in suffix:
                suffix = "whack" + str(fwd_idx)
                fwd_idx += 1
            word_filepath = os.path.join(args.removed_dir,
                    "dev_preds_"+suffix+".tsv")

            wm = set()
            if os.exists(word_filepath):
                with open(word_filepath) as wf:
                    wline = wf.readline()
                    while wline:
                        guid = wline.split("\t")[1]
                        wm.add(guid)
                        wline = wf.readline()
            print(f"{word}\t\t {count}\t{len(wm - om)}\t{len(om - wm)}")

            line = cf.readline()
    
if __name__ == "__main__":
    main()

