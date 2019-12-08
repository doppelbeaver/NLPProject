from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers.data import InputExample
from torch.utils.data.distributed import DistributedSampler
from scipy.special import softmax

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer,
                                  DistilBertConfig,
                                  DistilBertForSequenceClassification,
                                  DistilBertTokenizer)

from transformers import AdamW, WarmupLinearSchedule

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features

from utils import read_tsv
from lime.lime_text import LimeTextExplainer

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, 
                                                                                RobertaConfig, DistilBertConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}


def get_dataset(args, task, examples, tokenizer, evaluate=True):
    processor = processors[task]()
    output_mode = output_modes[task]
    label_list = processor.get_labels()
    if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
        # HACK(label indices are swapped in RoBERTa pretrained model)
        label_list[1], label_list[2] = label_list[2], label_list[1] 
    features = convert_examples_to_features(examples,
                                            tokenizer,
                                            label_list=label_list,
                                            max_length=args.max_seq_length,
                                            output_mode=output_mode,
                                            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


class Classifier:
    def __init__(self, args, model, tokenizer):
        self.model = model
        self.args = args
        self.tokenizer = tokenizer

    def predict_proba(self, input_tuples):
        """
        input_tuples: Array of tuple of strings
        """
        examples = []
        for i, t in enumerate(input_tuples):
            examples.append(InputExample(guid=i,
                text_a=t[0],
                text_b=t[1],
                # Add a dummy label
                label='1'))

        eval_dataset = get_dataset(self.args, self.args.task_name, examples, self.tokenizer, evaluate=True)
        self.args.eval_batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset) 
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        preds = None
        for batch in tqdm(eval_dataloader, desc="Explanation"):
            self.model.eval()
            batch = tuple(t.to(self.args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if self.args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = self.model(**inputs)
                _, logits = outputs[:2]
            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        probs = softmax(preds, axis=1)
        return probs


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model explanations will be written.")
    parser.add_argument("--input_file", default=None, type=str, required=True,
                        help="The input file where the examples are present.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--generate_html", action='store_true',
                        help="Set this flag if you want to generate html for each example.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=False)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    model.to(args.device)

    c = Classifier(args, model, tokenizer)
    explainer = LimeTextExplainer(class_names=["Not paraphrases", "Paraphrases"])
    lines = read_tsv(args.input_file)
    oneoutfile = os.path.join(args.output_dir, 'expl_summary.tsv')
    with open(oneoutfile, "w") as of:
        for i in trange(len(lines), desc="Instance"):
            line = lines[i]
            exp = explainer.explain_instance(line[3], line[4], int(line[0]), c.predict_proba, num_features=6, num_samples=10)

            summary_line = str(i+1)
            for e in exp.as_list():
                summary_line += "\t" + e[0] + "\t" + "%3f" % e[1]
            summary_line += "\n"
            of.write(summary_line)

            if args.generate_html:
                outfilename = os.path.join(args.output_dir, '_'.join([str(i+1), 'exp.html']))
                exp.save_to_file(outfilename)

if __name__ == "__main__":
    main()

