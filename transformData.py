import csv 
import string
import pdb 
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--input", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--output", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--d", default=None, type=str, required = True, help="What data? Quora (Q) or Microsoft (M)")

args = parser.parse_args()
if args.d == "Q":
    with open(args.input) as tsvfile, open(args.output,"w") as  t:
        tsvreader = csv.reader (tsvfile, delimiter = '\t', quoting = csv.QUOTE_NONE)
        temp = csv.writer(t, delimiter="\t", quoting=csv.QUOTE_NONE,escapechar='', quotechar='')

        i = 0 
        for row in tsvreader:
            try:    
                if i != 0:  
                    row[4] = row[3]
                    row[2] = row[1]
                    row[5] = '1'
                temp.writerow(row)
                i +=1
            except IndexError:
                print(row, i)
                continue
                
elif args.d == "M":
    with open(args.input) as tsvfile, open(args.output,"w") as  t:
        tsvreader = csv.reader (tsvfile, delimiter = '\t', quoting = csv.QUOTE_NONE)
        temp = csv.writer(t, delimiter="\t", quoting=csv.QUOTE_NONE,escapechar='', quotechar='')

        i = 0 
        for row in tsvreader:
            if i != 0:  
                row[4] = row[3]
                row[0] = '1'
            temp.writerow(row)
            i +=1
