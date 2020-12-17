import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, AlbertTokenizer , BertConfig
from flair.models import TextClassifier
from flair.data import Sentence
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re


class BertModel:
    def __init__(self, path='src/Bert', model_type='huseinzol05/bert-base-bahasa-cased'):
        self.path = path
        self.model_type = model_type
        self.tokenizer = AlbertTokenizer.from_pretrained(self.path, do_lower_case=True)
        self.model = BertForSequenceClassification.from_pretrained(self.path, num_labels=3)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        self.model.to(self.device)
        self.model.eval()

    def convert_to_features(self, sentence):

        text_a = sentence
        text_b = None
        max_length = 512
        pad_on_left = False
        pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        pad_token_segment_id = 0
        mask_padding_with_zero = True

        inputs = self.tokenizer.encode_plus(
            text_a,
            text_b,
            add_special_tokens=True,
            max_length=max_length)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        return [input_ids, attention_mask, token_type_ids]

    def convert_to_tensors(self, features):
        input_ids = torch.tensor([features[0]],
                                 dtype=torch.long)

        attention_mask = torch.tensor([features[1]],
                                      dtype=torch.long)

        token_type_ids = torch.tensor([features[2]],
                                      dtype=torch.long)

        inputs = {'input_ids': input_ids.to(self.device),
                  'attention_mask': attention_mask.to(self.device),
                  'token_type_ids': token_type_ids.to(self.device)}
        return inputs

    def interpret_result(self, output):
        result = {}
        logits = F.softmax(output[0][0], dim=0)
        logits_label = torch.argmax(logits, dim=0)
        logits_label = logits_label.detach().cpu().numpy().tolist()
        score = round(logits[logits_label].detach().cpu().numpy().tolist(), 5)
        logits = logits.detach().cpu().numpy().tolist()
        logits = [round(logit, 4) for logit in logits]
        result['label'] = logits_label
        result['confidence'] = score
        result['logits'] = logits
        return result

    def predict(self, text):
        features = self.convert_to_features(text)
        tensor = self.convert_to_tensors(features)
        outputs = self.model(**tensor)
        result = self.interpret_result(outputs)
        score = result['logits']
        return score[2], score[1], score[0]


class Vader:
    def __init__(self):
        self.classifier = SentimentIntensityAnalyzer()

    def predict(self, text):
        pred = self.classifier.polarity_scores(text)
        return pred['pos'], pred['neu'], pred['neg']


class Flair:
    def __init__(self):
        self.classifier = TextClassifier.load('en-sentiment')

    def predict(self, text):
        sentence = Sentence(text)
        self.classifier.predict(sentence)
        pred = str(sentence.labels[0])
        score = float(re.findall("\d+\.\d+", pred)[0])
        neutral = 0
        if pred[0] == 'P':
            positive = score
            negative = 0
        else:
            positive = 0
            negative = score

        return positive, neutral, negative
