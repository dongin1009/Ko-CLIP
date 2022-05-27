import os
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from model import KoCLIP

import argparse
from tqdm import tqdm

from utils import set_seed, mkdir, setup_logger, load_config_file
from transformers import BertTokenizer, AutoConfig

def tokenize(texts, tokenizer, context_length=77):
    return tokenizer(texts, max_length=context_length, padding='max_length', truncation=True, return_tensors='pt')

def zeroshot_classifier(model, classnames, templates, tokenizer, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = tokenize(texts, tokenizer).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            zeroshot_weights.append(class_embeddings)
        zeroshot_weights = torch.vstack(zeroshot_weights).to(device)
    return zeroshot_weights.t()

def evaluate(model, eval_dataloader, dataset_classes, tokenizer, device, args):
    top_1_correct = 0
    top_5_correct = 0
    class_wise_top_1_correct = {}
    class_wise_top_5_correct = {}
    class_wise_total_examples = {}
    
    with torch.no_grad():
        if args.template_version=='v1':
            templates = ["이것은 {}이다."] #v1
        elif args.template_version=='v2'
            templates = ["이것은 {}의 사진이다."] #v2
        classnames = [classname for classname in dataset_classes]
           
        zeroshot_weights = zeroshot_classifier(model, classnames, templates, tokenizer, device)
          
        for step, (images, labels) in enumerate(tqdm(eval_dataloader)):
            image_input = images.to(device)
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ zeroshot_weights).softmax(dim=-1)
            
            # top 5 predictions
            values, indices = similarity[0].topk(5)

            label_class = classnames[labels.item()]
            
            if label_class not in class_wise_total_examples:
                class_wise_top_1_correct[label_class] = 0
                class_wise_top_5_correct[label_class] = 0                
                class_wise_total_examples[label_class] = 0
            
            class_wise_total_examples[label_class] += 1

            if labels.item() == indices[0] :
                top_1_correct += 1
                class_wise_top_1_correct[label_class] += 1
                
            
            if labels.item() in indices :
                top_5_correct += 1
                class_wise_top_5_correct[label_class] += 1 
        
        top_1_accuracy = 100* top_1_correct / len(eval_dataloader)
        top_5_accuracy = 100* top_5_correct / len(eval_dataloader)

        class_wise_top_1_accuracy = { class_name : 100 * class_wise_top_1_correct[class_name] / class_wise_total_examples[class_name] for class_name in class_wise_top_1_correct }
        class_wise_top_5_accuracy = { class_name : 100 * class_wise_top_5_correct[class_name] / class_wise_total_examples[class_name] for class_name in class_wise_top_5_correct }
        
        return top_1_accuracy, top_5_accuracy, class_wise_top_1_accuracy, class_wise_top_5_accuracy

def save_accuracies(eval_result_output_file_path, top_1_accuracy, top_5_accuracy, class_wise_top_1_accuracy, class_wise_top_5_accuracy):
    with open(eval_result_output_file_path, "w") as fp:
        fp.write(f"Top 1 accuracy = {round(top_1_accuracy, 2)} %\n")
        fp.write(f"Top 5 accuracy = {round(top_5_accuracy, 2)} %\n")
        fp.write("----------------------------------------------------- \n")
        fp.write(f"Class wise accuracies (in %)\n\n")
        fp.write("{:<10}{:<10}{:<20}\n".format("Top-1", "Top-5", "Class name"))

        for class_name in class_wise_top_1_accuracy.keys():

            # fp.write(f"{class_name}\t\t\t{class_wise_top_1_accuracy[class_name]} \t {class_wise_top_5_accuracy[class_name]} \n")
            fp.write("{:<10}{:<10}{:<20}\n".format(round(class_wise_top_1_accuracy[class_name], 2), round(class_wise_top_5_accuracy[class_name], 2), class_name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", default=None, type=str, required=True, help="path of trained checkpoint")

    parser.add_argument("--data_dir", action=None, type=str, required=True, help="dataset directory")
    parser.add_argument("--pvm", default='google/vit-large-patch16-224')

    parser.add_argument("--template_version", default='v1', type=str, help="set the template")
    
    args = parser.parse_args()

    zero_shot_eval_output_dir = "zero_shot_eval_output"
    # creating directory to store demo result
    mkdir(path=zero_shot_eval_output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Image transform and text tokenizer
    transform = Compose([Resize(224, interpolation=Image.BICUBIC), CenterCrop(224), lambda image: image.convert("RGB"), ToTensor(), Normalize((0.4225, 0.4012, 0.3659), (0.2681, 0.2635, 0.2763))])

    tokenizer = BertTokenizer.from_pretrained('klue/roberta-large')  
    if 'RN101' in args.pvm:
        pvm = 'RN101'
    else:
        pvm = 'vit'
    model_params = {'pvm':args.pvm, 'embed_dim':512}
    model = KoCLIP(**model_params)


    # loading trained weights
    checkpoint = torch.load(args.checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # evaluating on CIFAR datasets
    if os.path.split(args.data_dir)[-1] == 'CIFAR100':
        imageDataset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform = transform)
        dataset_classes = ['사과', '관상어', '아기', '곰', '비버', '침대', '꿀벌', '딱정벌레', '자전거', '병', '그릇', '소년', '다리', '버스', '나비', '낙타', '캔', '성', '애벌레', '소', '의자', '침팬지', '시계', '구름', '바퀴벌레', '소파', '게', '악어', '컵', '공룡', '돌고래', '코끼리', '가자미', '숲', '여우', '소녀', '햄스터', '집', '캥거루', '키보드', '램프', '제초기', '표범', '사자', '도마뱀', '랍스터', '남자', '단풍나무', '오토바이', '산', '쥐', '버섯', '참나무', '오렌지', '난초', '수달', '야자수', '배', '트럭', '소나무', '평원', '접시', '양귀비', '고슴도치', '주머니쥐', '토끼', '너구리', '가오리', '도로', '로켓', '장미', '바다', '물개', '상어', '땃쥐', '스컹크', '고층건물', '달팽이', '뱀', '거미', '다람쥐', '지하철', '해바라기', '피망', '테이블', '탱크', '전화기', '텔레비전', '호랑이', '경운기', '기차', '송어', '튤립', '거북이', '옷장', '고래', '버드나무', '늑대', '여자', '벌레']
    if os.path.split(args.data_dir)[-1] == 'CIFAR10':
        imageDataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform = transform)
        dataset_classes = ['비행기', '자동차', '새', '고양이', '사슴', '개', '개구리', '말', '배', '트럭']
        #['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    eval_dataloader = DataLoader(imageDataset, sampler=SequentialSampler(imageDataset), batch_size=1)

    # now evaluate
    top_1_accuracy, top_5_accuracy, class_wise_top_1_accuracy, class_wise_top_5_accuracy = evaluate(model, eval_dataloader, dataset_classes, tokenizer, device, args)
    print("top_1_accuracy, top_5_accuracy :", round(top_1_accuracy, 2), round(top_5_accuracy, 2))
    eval_result_output_file_path = os.path.join(zero_shot_eval_output_dir, f'{os.path.split(args.data_dir)[-1]}_{pvm}_{args.template_version}.txt')
    save_accuracies(eval_result_output_file_path, top_1_accuracy, top_5_accuracy, class_wise_top_1_accuracy, class_wise_top_5_accuracy)

    print("--------------------")
    print("Check this for class-wise accuracies : ", eval_result_output_file_path)
    

if __name__ == "__main__":
    main()
