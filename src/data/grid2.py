import json
import os
import pdb
import pickle
import random

import torch
from PIL import Image
from progressbar import *
from skimage import io
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms


def round(x):
    return math.floor(x + 0.5)


def ctc_collate(batch):
    '''
    Stack samples into CTC style inputs.
    Modified based on default_collate() in PyTorch.
    By Yuan-Hang Zhang.
    '''
    xs, ys, lens, indices = zip(*batch)
    max_len = max(lens)
    x = default_collate(xs)
    x.narrow(2, 0, max_len)
    y = []
    for sub in ys:
        y += sub
    y = torch.IntTensor(y)
    lengths = torch.IntTensor(lens)
    y_lengths = torch.IntTensor([len(label) for label in ys])
    ids = default_collate(indices)

    return x, y, lengths, y_lengths, ids


class GRIDDataset(Dataset):
    def __init__(self, path, dset='train'):
        self.max_timesteps = 75
        self.align_path = os.path.join(path, 'aligns')
        self.data_path = os.path.join(path, 'mouths')
        self.dset = dset
        self.dataset = []

        print('Loading videos from {} set'.format(self.dset))
        cache_path = '{}.pkl'.format(self.dset)
        if not os.path.exists(cache_path):
            self.preprocess(cache_path)

        self.dataset, count_v, self.vocab, self.vocab_mapping = pickle.load(open(cache_path, 'rb'))
        print('{}: n_videos = {}, n_samples = {}, vocab_size = {}'.format(self.dset, count_v, len(self.dataset), len(self.vocab)))
        print('vocab = {}'.format('|'.join(self.vocab)))

    import pdb

    def preprocess(self, cache_path):
        vocab_unordered = {}
        vocab_unordered[' '] = True

        count_s, count_v = 0, 0
        pbar = ProgressBar().start()
        for dir_s in [p for p in os.listdir(self.data_path) if p.startswith('s')]:
            count_s += 1
            # get speaker videos
            for dir_v in os.listdir(os.path.join(self.data_path, dir_s)):
                cur_path = os.path.join(self.data_path, dir_s, dir_v)
                # load filter
                # check if sub was transcribed
                sub_file = os.path.join(self.align_path, dir_s, '{}.align'.format(dir_v))
                flag_add = os.path.exists(sub_file)
                file = os.path.join(cur_path, 'mouth_000.png')
                size = io.imread(file).shape
                assert size[0] == 40 and size[1] == 60, "Wrong size, got %d/%d for %s" % (size[0], size[1], file)
                assert len(os.listdir(cur_path)) == 75, "Frames missing, got %d" % len(os.listdir(cur_path))

                d = {'s': dir_s, 'v': dir_v, 'words': [], 't_start': [], 't_end': []}
                for line in open(sub_file, 'r').read().splitlines():
                    tok = line.split(' ')
                    if tok[2] != 'sil' and tok[2] != 'sp':
                        # store sub and append
                        d['words'].append(tok[2])
                        d['t_start'].append(int(tok[0]))
                        d['t_end'].append(int(tok[1]))
                        # build vocabulary
                        for char in tok[2]:
                            vocab_unordered[char] = True
                if self.dset == 'train':
                    count_v += 1
                    d['test'] = False
                    for flip in (False, True):
                        # add word instances
                        for w_start in range(1, 7):
                            d_i = d.copy()
                            d_i['flip'], d_i['mode'], d_i['w_start'] = flip, 1, w_start
                            # NOTE: it appears the authors never used the mode option.
                            # All instances used were either whole sentences or individual words.
                            d_i['w_end'] = w_start + d_i['mode'] - 1
                            frame_v_start = max(round(1 / 1000 * d['t_start'][d_i['w_start'] - 1]), 1)
                            frame_v_end = min(round(1 / 1000 * d['t_end'][d_i['w_end'] - 1]), 75)
                            if frame_v_end - frame_v_start + 1 >= 3:
                                self.dataset.append(d_i)
                        # add whole sentences
                        d_i = d.copy()
                        d_i['mode'], d_i['flip'] = 7, flip
                        self.dataset.append(d_i)
            pbar.update(int(count_s / 33 * 100))
        pbar.finish()

        # generate vocabulary
        self.vocab = []
        for char in vocab_unordered:
            self.vocab.append(char)
        self.vocab.sort()
        # invert ordered to create the char->int mapping
        # key: 1..N (reserve 0 for blank symbol)
        self.vocab_mapping = {}
        for i, char in enumerate(self.vocab):
            self.vocab_mapping[char] = i + 1

        pickle.dump((self.dataset, count_v, self.vocab, self.vocab_mapping), open(cache_path, 'wb'))

    def read_data(self, sample, data_path, vocab_mapping):
        test_mode = sample['test'] or False
        mode = sample['mode'] or random.randint(1, 6)
        flip = sample['flip'] or False
        if mode < 7:
            w_start = sample['w_start'] or random.randint(1, len(sample['words']) - mode + 1)
            w_end = w_start + mode - 1

        min_frame_v, max_frame_v = 1, 75
        sub = ''
        frame_v_start, frame_v_end = -1, -1

        if test_mode:
            frame_v_start, frame_v_end = min_frame_v, max_frame_v
            sub = ' '.join(sample['words'])
        else:
            # check number of words to train on
            if mode == 7:
                frame_v_start, frame_v_end = min_frame_v, max_frame_v
                sub = ' '.join(sample['words'])
            else:
                words = []
                for w_i in range(w_start, w_end + 1):
                    words.append(sample['words'][w_i - 1])
                sub = ' '.join(words)

                frame_v_start = max(round(1 / 1000 * sample['t_start'][w_start - 1]), 1)
                frame_v_end = min(round(1 / 1000 * sample['t_end'][w_end - 1]), 75)

                # if too short, back off to whole sequence
                if frame_v_end - frame_v_start + 1 <= 2:
                    frame_v_start, frame_v_end = min_frame_v, max_frame_v
                    sub = ' '.join(sample['words'])

        # construct output tensor
        y = []
        # allow whitespaces to be predicted
        for char in sub:
            y.append(vocab_mapping[char])

        # load images
        cur_path = '{}/{}/{}/{}'.format(data_path, sample['s'], sample['v'], 'mouth')
        # randomly flip video
        if test_mode:
            flip = False
        else:
            flip = flip or random.random() > 0.5

        x = torch.FloatTensor(3, frame_v_end - frame_v_start + 1, 40, 60)
        transform_lst = []
        if flip:
            transform_lst.append(transforms.functional.hflip)
        transform_lst += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7136, 0.4906, 0.3283],
                                 std=[0.113855171, 0.107828568, 0.0917060521])
        ]
        data_transform = transforms.Compose(transform_lst)
        frame_count = 0
        for f_frame in range(frame_v_start, frame_v_end + 1):
            file = '{}_{:03d}.png'.format(cur_path, f_frame - 1)
            img = Image.open(file).convert('RGB')
            img = data_transform(img)
            x[:, frame_count, :, :] = img
            frame_count += 1

        return x, y, sub

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # images: bs x chan x T x H x W
        x = torch.zeros(3, self.max_timesteps, 40, 60)
        sample = self.dataset[index]
        # targets: bs-length tensor of targets (each one is the length of the target seq)
        frames, y, sub = self.read_data(sample, self.data_path, self.vocab_mapping)
        x[:, : frames.size(1), :, :] = frames
        # input lengths: bs-length tensor of integers, representing
        # the number of input timesteps/frames for the given batch element
        length = frames.size(1)

        return x, y, length, index
