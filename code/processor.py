import os
import glob
from tqdm import tqdm
import json
import re

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'sep_pos': sep_pos,
                'turn_ids': turn_ids,
                'cls_pos':cls_pos
            }
            for input_ids, input_mask, segment_ids, sep_pos, turn_ids, cls_pos in choices_features
        ]
        self.label = label

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

class MuTualProcessor(DataProcessor):
    """Processor for the MuTual data set."""
    def __init__(self,logger):
        self.logger = logger

    def get_train_examples(self, data_dir):
        """See base class."""
        self.logger.info("LOOKING AT {} train".format(data_dir))
        file = os.path.join(data_dir, 'train')
        file = self._read_txt(file)
        return self._create_examples(file, 'train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        self.logger.info("LOOKING AT {} dev".format(data_dir))
        file = os.path.join(data_dir, 'dev')
        file = self._read_txt(file)
        return self._create_examples(file, 'dev')

    def get_test_examples(self, data_dir):
        """See base class."""
        self.logger.info("LOOKING AT {} test".format(data_dir))
        file = os.path.join(data_dir, 'test')
        file = self._read_txt(file)
        return self._create_examples(file, 'test')

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_txt(self, input_dir):
        lines = []
        files = glob.glob(input_dir + "/*txt")
        for file in tqdm(files, desc="read files"):
            with open(file, 'r', encoding='utf-8') as fin:
                data_raw = json.load(fin)
                data_raw["id"] = file
                lines.append(data_raw)
        return lines
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data_raw) in enumerate(tqdm(lines,desc="create examples")):
            id = "%s-%s" % (set_type, data_raw["id"])
            article = data_raw["article"]

            article = re.split(r"(f : |m : |M: |F: )", article)  
            #print(article)
            article = ["".join(i) for i in zip(article[1::2], article[2::2])]  

            truth = str(ord(data_raw['answers']) - ord('A'))
            options = data_raw['options']

            examples.append(
                InputExample(
                    guid=id,
                    text_a = [options[0], options[1], options[2], options[3]],
                    text_b=article, # this is not efficient but convenient
                    label=truth))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, max_utterance_num,
                                 tokenizer,logger, output_mode=None):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    
    for (ex_index, example) in enumerate(tqdm(examples,desc="create features")):
        #if ex_index % 10000 == 0:
        #    logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        choices_features = []
        all_tokens = []
        text_a = example.text_a
        text_b = example.text_b

        #for ending_idx, (text_a, text_b) in enumerate(zip(example.text_a, example.text_b)): 
        text_a[0] = re.sub("f : |m : |M: |F: ","",text_a[0])
        text_a[1] = re.sub("f : |m : |M: |F: ","",text_a[1])
        text_a[2] = re.sub("f : |m : |M: |F: ","",text_a[2])
        text_a[3] = re.sub("f : |m : |M: |F: ","",text_a[3])
        

        tokens_a = tokenizer.tokenize(text_a[0])
        tokens_a = ["[CLS]"] + tokens_a + ["[SEP]"]
        tokens_b = tokenizer.tokenize(text_a[1])
        tokens_b = ["[CLS]"] + tokens_b + ["[SEP]"]
        tokens_c = tokenizer.tokenize(text_a[2])
        tokens_c = ["[CLS]"] + tokens_c + ["[SEP]"]
        tokens_d = tokenizer.tokenize(text_a[3])
        tokens_d = ["[CLS]"] + tokens_d
        tokens_options = tokens_a + tokens_b + tokens_c + tokens_d

        tokens_article = []

        for idx, text in enumerate(text_b):  
            if len(text.strip()) > 0:  
                text = re.sub("f : |m : |M: |F: ","",text)
                tokens_article.extend(["[CLS]"]+tokenizer.tokenize(text) + ["[SEP]"])  
                                                
        tokens_article.pop(0)
        _truncate_seq_pair(tokens_options, tokens_article, max_seq_length-2)

        tokens = ["[CLS]"]
        turn_ids = [0]

        context_len = []
        sep_pos = []
        cls_pos = [0]

            
        tokens_article_raw = " ".join(tokens_article)
        tokens_article = []
        current_pos = 0
        for toks in tokens_article_raw.split("[SEP]")[-max_utterance_num - 1:-1]:
            context_len.append(len(toks.split()) + 1)
            tokens_article.extend(toks.split())
            tokens_article.extend(["[SEP]"])
            current_pos += context_len[-1]
            turn_ids += [len(sep_pos)] * context_len[-1]
            sep_pos.append(current_pos)
            cls_pos.append(current_pos+1)
        cls_pos.pop()
                
        tokens += tokens_article

        segment_ids = [0] * (len(tokens))

        tokens_options += ["[SEP]"]
        

        for index,toks in enumerate(tokens_options):
            if toks == "[CLS]":
                cls_pos.append(len(tokens)+index)

        tokens += tokens_options

        segment_ids += [1] * (len(tokens_options))
            
        turn_ids += [len(sep_pos)] * len(tokens_options) 
        sep_pos.append(len(tokens) - 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        turn_ids += padding

        context_len += [-1] * (max_utterance_num - len(context_len))
        sep_pos += [0] * (max_utterance_num + 1 - len(sep_pos))
        num_nodes = len(cls_pos)
        cls_pos += [0] * (max_utterance_num + 1 - len(cls_pos))
        cls_pos[-1] = num_nodes

        assert len(sep_pos) == max_utterance_num + 1
        assert len(cls_pos) == max_utterance_num + 1
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(context_len) == max_utterance_num 
        assert len(turn_ids) == max_seq_length 

        choices_features.append((input_ids, input_mask, segment_ids, sep_pos, turn_ids, cls_pos))  
            
            
        all_tokens.append(tokens)


        label_id = label_map[example.label] 
        

        if ex_index < 2:  
            logger.debug("*** Example ***")
            logger.debug("guid: %s" % (example.guid))
            for choice_idx, (input_ids, input_mask, segment_ids, sep_pos, turn_ids, cls_pos) in enumerate(choices_features):
                logger.debug("choice: {}".format(choice_idx))
                logger.debug("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.debug("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.debug("tokens: %s" % " ".join([str(x) for x in all_tokens[choice_idx]]))
                logger.debug("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.debug("sep_pos: %s" % " ".join([str(x) for x in sep_pos]))
                logger.debug("turn_ids: %s" % " ".join([str(x) for x in turn_ids]))
                logger.debug("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(
                example_id = example.guid, 
                choices_features = choices_features,
                label = label_id
                )
        )

    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop(0)
