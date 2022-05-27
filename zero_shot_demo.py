import os
import torch
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from model import KoCLIP

from utils.util import set_seed, mkdir, setup_logger, load_config_file

import argparse
from tqdm import tqdm
from glob import glob


import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc, font_manager

from transformers import BertTokenizer, AutoConfig
# to plot Korean, set the korean font
font_fname = '/usr/nanum/NanumGothic.ttf' # for plot korean
prop = font_manager.FontProperties(fname=font_fname, size=19)
mpl.rcParams['font.family'] = 'NanumGothic'
mpl.rcParams['axes.unicode_minus']  = False

def tokenize(texts, tokenizer, context_length=77):
    return tokenizer(texts, max_length=context_length, padding='max_length', truncation=True, return_tensors='pt')

def zeroshot_classifier(model, classnames, templates, tokenizer, device):
    '''
    Creates texts for each class using templates and extracts their text embeddings.
    '''
    print("Getting text features from classnames")
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(' '.join(classname.split('_'))) for template in templates] #format with class
            print("class texts :")
            print(texts)
            texts = tokenize(texts, tokenizer).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            zeroshot_weights.append(class_embeddings)
        zeroshot_weights = torch.vstack(zeroshot_weights).to(device)        
    return zeroshot_weights.t()



def predict_class(model, images, image_names, dataset_classes, tokenizer, device, args):
    '''
    Classifies images by predicting their classes from "dataset_classes"
    '''
    with torch.no_grad():
        classnames = [classname for classname in dataset_classes]
        if args.template_version=='v1':
            templates = ["이것은 {}이다."] #v1
        elif args.template_version=='v2'
            templates = ["이것은 {}의 사진이다."] #v2
        templates = ["이것은 {}이다."]
        zeroshot_weights = zeroshot_classifier(model, classnames, templates, tokenizer, device)
        predictions = []
        for image, image_name in zip(images, image_names):
            image_input = image.to(device)
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity_scale = 35.0
            similarity = (similarity_scale * image_features @ zeroshot_weights).softmax(dim=-1)
            
            # top 5 predictions
            values, indices = similarity[0].cpu().topk(5)
            print("------------------------")
            print("img : ", image_name)
            print("predicted classes :")
            for value, index in zip(values, indices):
                print(f"{classnames[index]:>16s}: {100 * value.item():.2f}%")
            print("------------------------")   

            predictions.append((values, indices))     
    
    return predictions                     

def show_predictions(images, predictions, dataset_classes, save_dir):
    '''
    To give predictions in a nice figure
    '''
    plt.rcParams["font.family"] = "NanumGothic"
    mpl.rcParams["axes.unicode_minus"]=False
    plt.figure(figsize=(27, 27))
    if len(images) == 1:
        # zero-shot demo on a single image only
        image = images[0]
        top_probs =  [prediction[0] for prediction in predictions]
        top_labels = [prediction[1] for prediction in predictions]
        
        plt.figure(figsize=(8, 2))
        plt.subplot(1, 2, 1)
        plt.imshow(image.permute(1, 2, 0))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        y = np.arange(top_probs[0].shape[-1])
        plt.grid()
        plt.barh(y, top_probs[0])
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [dataset_classes[index] for index in top_labels[0].numpy()], fontproperties=prop)
        plt.xlabel("probability")

        plt.subplots_adjust(wspace=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "single_img_demo.png"))
        plt.show()

        return

    # for images in a directory
    plt.figure(figsize=(16, 2*(len(images))))
    top_probs = [prediction[0] for prediction in predictions]
    top_labels = [prediction[1] for prediction in predictions]
    
    for i, image in enumerate(images):
        plt.subplot(len(images)//2, 4, 2 * i + 1)
        plt.imshow(image.permute(1, 2, 0))
        plt.axis("off")

        plt.subplot(len(images)//2, 4, 2 * i + 2)
        y = np.arange(top_probs[i].shape[-1])
        plt.grid()
        plt.barh(y, top_probs[i])
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [dataset_classes[index] for index in top_labels[i].numpy()])
        plt.xlabel("probability")

    plt.subplots_adjust(wspace=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "demo.png"))
    plt.show()

def zero_shot_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", default=None, type=str, required=True, help="path of saved weights")
    parser.add_argument("--img_dir", default="test_images", type=str, required=False, help="directory containing test images. Please have even number of images for a nice demo figure")
    parser.add_argument("--img_path", default=None, type=str, required=False, help="Path of an image to classify")
    parser.add_argument("--show_predictions", action='store_true', help="To show predictions in a figure")
    parser.add_argument("--pvm", default='google/vit-large-patch16-224')
    parser.add_argument("--template_version", default='v1', type=str, help="set the template")
    args = parser.parse_args()

    demo_output_dir = "demo_output"
    # creating directory to store demo result
    mkdir(path=demo_output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Image transform and text tokenizer
    transform = Compose([Resize(224, interpolation=Image.BICUBIC), CenterCrop(224), lambda image: image.convert("RGB"), ToTensor(), Normalize((0.4225, 0.4012, 0.3659), (0.2681, 0.2635, 0.2763))])

    transform_no_norm = Compose([Resize(224, interpolation=Image.BICUBIC), CenterCrop(224), lambda image: image.convert("RGB"), ToTensor()])

    tokenizer = BertTokenizer.from_pretrained('klue/roberta-large')  

    # CIFAR100 classes
    dataset_classes = ['사과', '관상어', '아기', '곰', '비버', '침대', '꿀벌', '딱정벌레', '자전거', '병', '그릇', '소년', '다리', '버스', '나비', '낙타', '캔', '성', '애벌레', '소', '의자', '침팬지', '시계', '구름', '바퀴벌레', '소파', '게', '악어', '컵', '공룡', '돌고래', '코끼리', '가자미', '숲', '여우', '소녀', '햄스터', '집', '캥거루', '키보드', '램프', '제초기', '표범', '사자', '도마뱀', '랍스터', '남자', '단풍나무', '오토바이', '산', '쥐', '버섯', '참나무', '오렌지', '난초', '수달', '야자수', '배', '트럭', '소나무', '평원', '접시', '양귀비', '고슴도치', '주머니쥐', '토끼', '너구리', '가오리', '도로', '로켓', '장미', '바다', '물개', '상어', '땃쥐', '스컹크', '고층건물', '달팽이', '뱀', '거미', '다람쥐', '지하철', '해바라기', '피망', '테이블', '탱크', '전화기', '텔레비전', '호랑이', '경운기', '기차', '송어', '튤립', '거북이', '옷장', '고래', '버드나무', '늑대', '여자', '벌레']
    #['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

    model_params = {'pvm':args.pvm, 'embed_dim':512}
    model = KoCLIP(**model_params)

    # loading trained weights
    checkpoint = torch.load(args.checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    images, image_names, raw_images = [], [], []
    
    if args.img_path:
        # zero-shot demo on a single image only
        img_path = args.img_path
        image_name = os.path.split(img_path)[-1]
        image = transform(Image.open(img_path)).unsqueeze(0)
        raw_image = transform_no_norm(Image.open(img_path))        
        raw_images.append(raw_image)
        images.append(image)
        image_names.append(image_name)
    
    else :
        # zero-shot demo for images in a directory
        for img_path in glob(args.img_dir + '/*'):
            image_name = os.path.split(img_path)[-1]
            image = transform(Image.open(img_path)).unsqueeze(0)

            # un normalized image for display
            raw_image = transform_no_norm(Image.open(img_path))        
            raw_images.append(raw_image)
            images.append(image)
            image_names.append(image_name)
    
    predictions = predict_class(model, images, image_names, dataset_classes, tokenizer, device, args)

    if args.show_predictions:
        if args.img_path:
            
            try :
                show_predictions(raw_images, predictions, dataset_classes, demo_output_dir)
                print("==========")
                print(f"Please check the following for zero-shot prediction demo figure")
                print(" -- ", os.path.join(demo_output_dir, "single_img_demo.png"))
            except:
                print("Some error while generating demo figure for a single image.")

        else :
            # zero-shot demo for images in a directory
            try :
                show_predictions(raw_images, predictions, dataset_classes, demo_output_dir)
                print("==========")
                print(f"Please check the following for zero-shot prediction demo figure")
                print(" -- ", os.path.join(demo_output_dir, "demo.png"))
            except:
                print("Some error while generating demo figure. Please try putting even number of images in images directory for a nice demo figure.")

if __name__ == "__main__":
    zero_shot_demo()




# To compare vit & resnet

# plt.figure(figsize=(27, 27))
# top_probs_vit = [prediction[0] for prediction in p_vit]
# top_labels_vit = [prediction[1] for prediction in p_vit]

# top_probs_rn = [prediction[0] for prediction in p_rn]
# top_labels_rn = [prediction[1] for prediction in p_rn]

# plt.rcParams["font.family"] = "NanumGothic"
# mpl.rcParams["axes.unicode_minus"]=False
# for i, image in enumerate(raw_images):
#     plt.subplot(len(raw_images)//2, 6, 3 * i + 1)
#     plt.imshow(image.permute(1, 2, 0))
#     plt.axis("off")

#     plt.subplot(len(raw_images)//2, 6, 3 * i + 2)
#     y = np.arange(top_probs_vit[i].shape[-1])
#     plt.title('ViT-large-patch16', fontsize=15)
#     plt.grid()
#     plt.barh(y, top_probs_vit[i], height=0.5)
#     plt.gca().invert_yaxis()
#     plt.gca().set_axisbelow(True)
#     plt.yticks(y, [dataset_classes[index] for index in top_labels_vit[i].numpy()], fontproperties=prop)
#     plt.xlabel("probability")

#     plt.subplot(len(raw_images)//2, 6, 3 * i + 3)
#     y = np.arange(top_probs_rn[i].shape[-1])
#     plt.title('ResNet101', fontsize=15)
#     plt.grid()
#     plt.barh(y, top_probs_rn[i], height=0.5)
#     plt.gca().invert_yaxis()
#     plt.gca().set_axisbelow(True)
#     plt.yticks(y, [dataset_classes[index] for index in top_labels_rn[i].numpy()], fontproperties=prop)
#     plt.xlabel("probability")

# plt.subplots_adjust(wspace=0.5)
# plt.tight_layout()
# plt.savefig("demo_output/compare_pvm.png",dpi=300)
# plt.show()