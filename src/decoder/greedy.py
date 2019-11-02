import editdistance
import numpy as np
import torch
from src.decoder.decoder import Decoder


class GreedyDecoder(Decoder):
    def __init__(self, vocab):
        super().__init__(self, vocab)

    def convert_to_string(self, tokens, seq_len=None):
        if not seq_len:
            seq_len = tokens.size(0)
        out = []
        for i in range(seq_len):
            if len(out) == 0:
                out.append(tokens[i])
            else:
                if tokens[i] != tokens[i - 1]:
                    out.append(tokens[i])
        return ''.join(self.vocab_list[i] for i in out)

    def decode(self, logits, seq_lens):
        decoded = []
        tlogits = logits.transpose(0, 1)
        _, tokens = torch.max(tlogits, 2)
        for i in range(tlogits.size(0)):
            output_str = self.convert_to_string(tokens[i], seq_lens[i])
            decoded.append(output_str)
        return decoded

