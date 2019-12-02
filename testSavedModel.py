from transformers import *

import argparse
import torch
import numpy as np
import csv
from tqdm import tqdm

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
parser.add_argument("-m", default='../../output/MRPC/checkpoint-2250/', type=str, 
                        help="Directory of fine-tuned model")

parser.add_argument("-i", default="../../glue_data/MRPC/dev.tsv", type=str, help="Directory of the input file")
parser.add_argument("-o", default="../../glue_data/MRPC/dev_new.tsv", type=str, help="Directory of the output file")
args = parser.parse_args()
config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]


# Load dataset, tokenizer, model from pretrained model/vocabulary
model = model_class.from_pretrained(args.m)
tokenizer = tokenizer_class.from_pretrained(args.m, do_lower_case=args.do_lower_case)


#Print out the wrong prediction
with open(args.i) as tsvfile, open(args.o,"w") as  t:
    tsvreader = csv.reader (tsvfile, delimiter = '\t', quoting = csv.QUOTE_NONE)
    temp = csv.writer(t, delimiter="\t", quoting=csv.QUOTE_NONE,escapechar='', quotechar='')

    i = 0 
    for row in tqdm(tsvreader):
    	if i != 0:
            sentence_0 = row[3]
            sentence_1 = row[4]
            inputs_1 = tokenizer.encode_plus(sentence_0, sentence_1, add_special_tokens=True, return_tensors='pt')
            label = row[0]
            pred_1 = model(inputs_1['input_ids'], token_type_ids=inputs_1['token_type_ids'])[0].argmax().item()
            if pred_1 != int(label):
                temp.writerow(row)
    	i +=1
		



