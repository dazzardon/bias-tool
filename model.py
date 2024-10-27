# model.py

import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel

class BertForTokenAndSequenceJointClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_token_labels = config.num_labels
        self.num_sequence_labels = 2  # Propaganda or Non-Propaganda

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.token_classifier = nn.Linear(config.hidden_size, self.num_token_labels)
        self.sequence_classifier = nn.Linear(config.hidden_size, self.num_sequence_labels)

        self.init_weights()

        # Label mappings
        self.token_tags = {
            0: 'O',
            1: 'Appeal_to_Authority',
            2: 'Appeal_to_fear-prejudice',
            3: 'Bandwagon,Reductio_ad_hitlerum',
            4: 'Black-and-White_Fallacy',
            5: 'Causal_Oversimplification',
            6: 'Doubt',
            7: 'Exaggeration,Minimisation',
            8: 'Flag-Waving',
            9: 'Loaded_Language',
            10: 'Name_Calling,Labeling',
            11: 'Repetition',
            12: 'Slogans',
            13: 'Thought-terminating_Cliches',
            14: 'Whataboutism,Straw_Men,Red_Herring'
        }

        self.sequence_tags = {
            0: 'Non-Propaganda',
            1: 'Propaganda'
        }

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        sequence_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        sequence_output = self.dropout(sequence_output)
        sequence_logits = self.sequence_classifier(sequence_output)

        token_output = outputs.last_hidden_state
        token_output = self.dropout(token_output)
        token_logits = self.token_classifier(token_output)

        return {
            'sequence_logits': sequence_logits,
            'token_logits': token_logits
        }
