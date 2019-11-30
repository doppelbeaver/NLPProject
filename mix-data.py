from random import sample

MRPC_SIZE = 3668
QQP_SIZE = 363870

rset = set(sample(range(QQP_SIZE), MRPC_SIZE))

filename = "QQP/train.tsv"
outfilename = "MRPC/train.tsv"
with open(filename, 'r') as f:
    with open(outfilename, 'a') as of:
        f.readline()
        line = f.readline()
        i = 0
        while line:
            if i in rset:
                fidx = line.find("\t")
                lidx = line.rfind("\t")
                of.write(line[lidx+1:-1] + line[fidx:lidx] + "\n")
            line = f.readline()
            i += 1

