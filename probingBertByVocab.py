from transformers import *

import argparse
import torch
import numpy as np
import tensorflow as tf
import csv
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import pdb
import os
import mmap


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}
parser = argparse.ArgumentParser()

parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
parser.add_argument("--p", default='bert-base-cased', type=str,
                        help="Pretrained model (bert-base-cased)")
parser.add_argument("--m", default='../../output/MRPC/checkpoint-2250/', type=str, 
                        help="Directory of fine-tuned model")

parser.add_argument("--i", default="../../glue_data/MRPC/dev_new.tsv", type=str, help="Directory of the original duplication file")
parser.add_argument("--v", default = "../../output/MRPC/vocab.txt", type=str, help="Directory of the vocabulary" )

args = parser.parse_args()
config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

list_of_vocab = []
with open(args.v) as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter = '\t', quoting = csv.QUOTE_NONE)
    for row in tqdm(tsvreader):
        list_of_vocab.append(row[0])


# Load dataset, tokenizer, model from pretrained model/vocabulary

model = model_class.from_pretrained(args.m)
tokenizer = tokenizer_class.from_pretrained(args.m, do_lower_case=args.do_lower_case)

for w in tqdm(list_of_vocab):
    count = 0 
    wrong = 0
    idx = 0
    output_dir = ""
    if w == "/":
        output_dir = "whack_out.txt"
    else:
        output_dir = w+"_out.txt"
    with open(args.i) as tsvfile, open(output_dir, "w+") as t:
        tsvreader = csv.reader(tsvfile, delimiter = '\t', quoting = csv.QUOTE_NONE)
        temp = csv.writer(t, delimiter="\t", quoting=csv.QUOTE_NONE,escapechar='', quotechar='')
        for row in tqdm(tsvreader, total=get_num_lines(args.i)):
            if idx != 0:
                sentence_0 = tokenizer.tokenize(row[3])
                sentence_1 = list(filter(lambda x: x != w, sentence_0))
                if len(sentence_0) > len(sentence_1):
                    count +=1
                    inputs_1 = tokenizer.encode_plus(sentence_0, sentence_1, add_special_tokens=True, return_tensors='pt')
                    pred_1 = model(inputs_1['input_ids'], token_type_ids=inputs_1['token_type_ids'])[0].argmax().item()
                    if pred_1 == 0:
                        wrong +=1
                        temp.writerow([1, row[1], row[2], row[3], ' '.join(sentence_1)])
                else:
                    continue
        
            idx+=1
    if count != 0:
        ex_prediction = [w, wrong, count, str(float(wrong/count)*100.0)]
        print(ex_prediction)
    else:
        print(f"No example has: {w}")
    if wrong == 0:
        os.remove(output_dir)

		



