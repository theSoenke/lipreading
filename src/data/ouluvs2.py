import glob
import os

import psutil
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from tables import Float32Col, Int32Col, IsDescription, StringCol, open_file
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class OuluVS2Dataset(Dataset):
    def __init__(self, path, mode='train'):
        self.path = path
        self.max_timesteps = 38
        self.speakers = {
            'train': (0, 40),
            'val': (43, 48),
            'test': (48, 53),
        }[mode]
        self.file_list = self.build_file_list()
        self.transcripts = [
            "Excuse me",
            "Goodbye",
            "Hello",
            "How are you",
            "Nice to meet you",
            "See you",
            "I am sorry",
            "Thank you",
            "Have a good time",
            "You are welcome",
        ]
        self.preprocess()

    def preprocess(self):
        vocab_unordered = {}

        # transcripts_path = os.path.join(self.path, 'transcript_sentence')
        # transcripts = os.listdir(transcripts_path)
        # for transcript in transcripts:
        #     file = os.path.join(transcripts_path, transcript)
        #     for line in open(file, 'r').read().splitlines():
        #         for char in line:
        #             vocab_unordered[char] = True

        for transcript in self.transcripts:
            transcript = transcript.lower()
            for char in transcript:
                vocab_unordered[char] = True
        self.vocab = []
        for char in vocab_unordered:
            self.vocab.append(char)
        self.vocab.sort()
        self.vocab_mapping = {' ': 0}
        for i, char in enumerate(self.vocab):
            self.vocab_mapping[char] = i + 1

    def build_file_list(self):
        videos = []
        video_path = self.path + 'cropped_mouth_mp4_phrase'
        pattern = video_path + "/**/*.mp4"
        files = glob.glob(pattern, recursive=True)
        max_frames = 0
        for file in files:
            split = file.split("/")[-1][:-4].split("_")
            speaker = int(split[0][1:])
            if speaker >= self.speakers[0] and speaker < self.speakers[1]:
                videos.append(file)
        return videos

    def load_utterance(self, speaker, utterance):
        y = []
        # transcript = os.path.join(self.path, 'transcript_sentence', 's' + str(speaker))
        # for line in open(transcript, 'r').read().splitlines():
        #     line.strip('.')
        #     words = line.split(' ')
        #     for char in line:
        #         y.append(self.vocab_mapping[char])

        transcript = self.transcripts[(utterance - 31) // 3].lower()
        for char in transcript:
            y.append(self.vocab_mapping[char])

        return y

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        x = torch.zeros(3, self.max_timesteps, 100, 120)
        frames, _, _ = torchvision.io.read_video(path)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((100, 120)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7136, 0.4906, 0.3283],
                                 std=[0.113855171, 0.107828568, 0.0917060521])
        ])

        for i, frame in enumerate(frames):
            img = transform(frame)
            x[:, i, :, :] = img

        split = path.split("/")[-1][:-4].split("_")
        speaker, view, utterance = [int(x[1:]) for x in split]
        view = [0, 30, 45, 60, 90][view-1]

        y = self.load_utterance(speaker, utterance)
        length = frames.size(0)

        return x, y, length, idx


class Video(IsDescription):
    yaw = Float32Col()
    view = Float32Col()
    speaker = Int32Col()
    utterance = Int32Col()
    file = StringCol(32)


def preprocess(path, output, workers=None):
    workers = psutil.cpu_count() if workers == None else workers
    if os.path.exists(output) == False:
        os.makedirs(output)

    output_path = "%s/ouluvs2.h5" % (output)
    if os.path.exists(output_path):
        os.remove(output_path)

    labels = None
    dataset = OuluVS2Dataset(path=path)
    preprocess_hdf5(
        dataset=dataset,
        output_path=output_path,
        table='train',
        workers=workers,
    )
    print("Saved preprocessed file: %s" % output_path)


def preprocess_hdf5(dataset, output_path, table, workers):
    file = open_file(output_path, mode="a")
    table = file.create_table("/", table, Video, expectedrows=len(dataset))
    row = table.row
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=workers)

    with tqdm(total=len(dataset)) as progress:
        for batch in data_loader:
            for i in range(len(batch['yaw'])):
                for column in batch:
                    row[column] = batch[column][i]
                row.append()
                progress.update(1)
    table.flush()
    file.close()
