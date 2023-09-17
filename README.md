# Ko-CLIP
This repository contains code to train Korean [CLIP](https://github.com/openai/CLIP) on [MS-COCO](https://cocodataset.org/#home) with Korean annotations in [AI-HUB](https://aihub.or.kr/aihubdata/data/view.do?currMenu=120&topMenu=100&aihubDataSe=extrldata&dataSetSn=261). Additionally, to get more Korean annotations, we use [Naver Papago translator](https://papago.naver.com/) from English to Korean on [VizWiz](https://vizwiz.org/tasks-and-datasets/image-captioning/) data.

### Pretrained Model
The original CLIP has large-scaled dataset however ours dataset is much less than CLIP's. Due to lack Korean caption data, we use pretrained language and visual model to get representations on less dataset. 

#### Pretrained Language Model
- We fixed PLM as [klue/roberta-large](https://huggingface.co/klue/roberta-large) on huggingface to get more powerful text representation in Korean.

#### Pretrained Visual Model
- We used PVMs as [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) on huggingface and [RN101](https://pytorch.org/vision/stable/generated/torchvision.models.resnet101.html) on torchvision to get image representations.
- Actually, the images are not dependent in number of Korean dataset, but CLIP is trained pair of texts-images so Ko-CLIP trained limited images(which has Korean captions).

See [WandB dashboard](https://wandb.ai/dongin1009/koclip?workspace=user-dongin1009) for check training records and model performance with comparing pretrained visual models.

### Zero-shot classification
In zero-shot classification, we predict on CIFAR-10 and CIFAR-100 datasets.

We refer to [CLIP](https://github.com/openai/CLIP), [clip-training](https://github.com/revantteotia/clip-training) for train, [koclip](https://github.com/jaketae/koclip) idea, and other pretrained models.
