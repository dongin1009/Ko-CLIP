from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import os
import random

def _transform_vizwiz(n_px):
    return Compose([
    Resize(n_px, interpolation=Image.BICUBIC),
    CenterCrop(n_px),
    lambda image: image.convert("RGB"),
    ToTensor(),
    Normalize((0.4874, 0.4378, 0.3860), (0.2842, 0.2729, 0.2748)),
])
def _transform_mscoco(n_px):
    # mean: COCO (0.4225, 0.4012, 0.3659)
    # std: COCO (0.2681, 0.2635, 0.2763)
    # mean: vizwiz_train (0.4901, 0.4380, 0.3844)
    # std: vizwiz_train (0.2829, 0.2720, 0.2731)
    # mean: vizwiz_all (0.4874, 0.4378, 0.3860)
    # std: vizwiz_all (0.2842, 0.2729, 0.2748)
    return Compose([
    Resize(n_px, interpolation=Image.BICUBIC),
    CenterCrop(n_px),
    lambda image: image.convert("RGB"),
    ToTensor(),
    Normalize((0.4225, 0.4012, 0.3659), (0.2681, 0.2635, 0.2763)),
])
def get_img_id_to_path_and_captions(annotations):
    img_id_to_img_path, img_id_to_captions = {}, {}
    for each_info in annotations:
        img_id = each_info['id']

        img_id_to_img_path[img_id] = each_info['file_path']
        if img_id not in img_id_to_captions:
            img_id_to_captions[img_id] = []
        img_id_to_captions[img_id] = each_info['caption_ko']
    return img_id_to_img_path, img_id_to_captions
        

class KoCLIP_CUSTOM_dataset(Dataset):
    """KoCLIP_CUSTOM_dataset. To train CLIP on COCO-captions and VizWiz-captions."""

    def __init__(self, annotation_file, img_dir, img_type='mscoco', context_length=77, input_resolution=224):
        
        super(KoCLIP_CUSTOM_dataset, self).__init__()
        if isinstance(annotation_file, str):
            self.annotation_file = annotation_file
            self.img_dir = img_dir
            annotations = read_json(annotation_file)
            self.img_id_to_filename, self.img_id_to_captions = get_img_id_to_path_and_captions(annotations)
            self.img_ids = list(self.img_id_to_filename.keys())

        elif isinstance(annotation_file, list):
            self.annotation_file = list(annotation_file)
            self.img_dir = list(img_dir)
        else:
            print('dataset args error!')

        self._tokenizer = BertTokenizer.from_pretrained('klue/roberta-large')
        self.context_length = context_length
        if img_type=='mscoco':
            self.transform = _transform_mscoco(input_resolution)
        elif img_type=='vizwiz':
            self.transform = _transform_vizwiz(input_resolution)

    def tokenize(self, text):
        return self._tokenizer(text, max_length=self.context_length, padding='max_length', truncation=True, return_tensors='pt')

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        # randomly pick one caption from the image captions
        text = random.choice(self.img_id_to_captions[img_id])

        img_filename = self.img_id_to_filename[img_id]

        img_path = os.path.join(self.img_dir, img_filename)
        img = Image.open(img_path)
        img_input = self.transform(img)
        text_input = self.tokenize(text)

        return img_input, text_input

def get_dataloader(config, dataset, is_train=True):
    if is_train:
        sampler = RandomSampler(dataset)
        batch_size = config.per_gpu_train_batch_size * max(1, config.n_gpu)
    else:
        sampler = SequentialSampler(dataset)
        batch_size = config.per_gpu_eval_batch_size * max(1, config.n_gpu)

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=config.num_workers)

    return dataloader