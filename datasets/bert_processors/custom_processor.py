import os

from datasets.bert_processors.abstract_processor import BertProcessor, InputExample


class CustomProcessor(BertProcessor):
    NAME = 'Custom'
    NUM_CLASSES = 40
    IS_MULTILABEL = False

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, 'CUSTOM', 'train.csv')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, 'CUSTOM', 'dev.csv')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, 'CUSTOM', 'test.csv')), 'test')

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if len(line) >= 2:                
                text_a = line[1]
                label = line[0]            
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples