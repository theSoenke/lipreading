import numpy as np
import editdistance


class Decoder():
    def __init__(self, vocab):
        self.vocab_list = [char for char in vocab]

    def predict(self, batch_size, logits, y, lengths, y_lengths, n_show=5):
        decoded = self.decode(logits, lengths)

        cursor = 0
        gt = []
        n = min(n_show, logits.size(1))
        samples = []
        for b in range(batch_size):
            y_str = ''.join([self.vocab_list[ch] for ch in y[cursor: cursor + y_lengths[b]]])
            gt.append(y_str)
            cursor += y_lengths[b]
            if b < n:
                samples.append([y_str, decoded[b]])

        return decoded, gt, samples

    def decode(self, logits, seq_lens):
        raise NotImplementedError

    def wer(self, s1, s2):
        s1_words, s2_words = s1.split(), s2.split()
        distance = editdistance.eval(s1_words, s2_words)
        return distance /  max(len(s1_words), len(s2_words))

    def cer(self, s1, s2):
        s1, s2 = s1.replace(' ', ''), s2.replace(' ', '')
        distance = editdistance.eval(s1, s2)
        return distance / max(len(s1), len(s2))

    def cer_batch(self, decoded, gt):
        return self.compare_batch(decoded, gt, self.cer)

    def wer_batch(self, decoded, gt):
        return self.compare_batch(decoded, gt, self.wer)

    def compare_batch(self, decoded, gt, func):
        assert len(decoded) == len(gt), f'batch size mismatch: {len(decoded)}!={len(gt)}'

        results = []
        for i, batch in enumerate(decoded):
            for sentence in range(len(batch)):
                error = func(decoded[i][sentence], gt[i])
                results.append(error)

        return np.mean(results)
