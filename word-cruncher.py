import os
import argparse

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--original", default=None, type=str, required=True)
    parser.add_argument("--removed_dir", default=None, type=str, required=True)

    om = set()
    with open(args.original) as of:
        line = of.readline()
        while line:
            guid = line.split("\t")[1]
            om.add(guid)
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
            suffix = word
            if '/' in suffix:
                suffix = "whack" + str(fwd_idx)
                fwd_idx += 1
            word_filepath = os.path.join(args.removed_dir,
                    "dev_preds_"+suffix+".tsv")
            wm = set()
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

