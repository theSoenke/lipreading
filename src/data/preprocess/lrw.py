import imageio
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


def load_video(file):
    vid = imageio.get_reader(file,  'ffmpeg')
    frames = []
    for i in range(0, 29):
        image = vid.get_data(i)
        image = F.to_tensor(image)
        frames.append(image)

    return build_tensor(frames)


def build_tensor(frames):
    temporalVolume = torch.FloatTensor(1, 29, 112, 112)
    croptransform = transforms.CenterCrop((112, 112))

    for i in range(0, 29):
        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((122, 122)),
            croptransform,
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.4161, ], [0.1688, ]),
        ])(frames[i])

        temporalVolume[0][i] = result

    return temporalVolume
