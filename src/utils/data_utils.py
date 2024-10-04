import json
import base64
from io import BytesIO
from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont

def get_dataset(args):
    if "webqa" in args.dataset:
        return WebQADataset(args)
    
def get_concat_h_multi_resize(im_list, resample=Image.BICUBIC):
    min_height = min(im.height for im in im_list)
    im_list_resize = [im.resize((int(im.width * min_height / im.height), min_height),resample=resample)
                      for im in im_list]
    total_width = sum(im.width for im in im_list_resize)
    dst = Image.new('RGB', (total_width, min_height))
    pos_x = 0
    for im in im_list_resize:
        dst.paste(im, (pos_x, 0))
        pos_x += im.width
    return dst

def concat_img_cap(img, cap):
    width, height = img.size 
    draw = ImageDraw.Draw(img)
    textwidth, textheight = draw.textsize(cap)
    margin = 10
    x = width - textwidth - margin
    y = height - textheight - margin
    draw.text((x, y), cap, fill="black")
    return img

class WebQADataset(Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        
        self.is_fact_given = args.is_fact_given
        
        with open("data/webqa/imgs.lineidx", "r") as fp_lineidx:
            self.lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]
            
        if "visual" in args.dataset:
            with open("data/webqa/visual_train_val.json", "r") as fin:
                self.datas = json.load(fin)
        elif "textual" in args.dataset:
            with open("data/webqa/textual_train_val.json", "r") as fin:
                self.datas = json.load(fin)
        else:
            with open("data/webqa/WebQA_data_first_release/WebQA_train_val.json", "r") as fin:
                self.datas = json.load(fin)
        self.keys = list(self.datas.keys())
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        key = self.keys[index]
        question = self.datas[key]["Q"].strip("\"")
        if not self.is_fact_given:
            return {"id": key, "question": question, "fact": None}
        text_facts = self.datas[key]["txt_posFacts"]
        img_facts = self.datas[key]["img_posFacts"]
        if len(text_facts) > 0:
            fact = ""
            for _, f in enumerate(text_facts):
                fact += "Fact {_}: " + f + "\n"
            return {"id": key, "question": question, "fact": {"text": fact, "image": None}}
        else:
            images = []
            for _, f in enumerate(img_facts):
                image_id = f["image_id"]
                caption = f["caption"]
                with open("data/webqa/imgs.tsv", "r") as fp:
                    fp.seek(self.lineidx[int(image_id)%10000000])
                    imgid, img_base64 = fp.readline().strip().split('\t')
                img = Image.open(BytesIO(base64.b64decode(img_base64)))
                # img_cap = concat_img_cap(img, caption)
                # img_cap.save("sample.jpg")
                images.append(img)
            fact = get_concat_h_multi_resize(images)
            # fact.save("sample.jpg")
            return {"id": key, "question": question, "fact": {"text": None, "image": fact}}

def load_dataset(args):
    model_nickname = args.model_name.split("/")[-1]
    if "viquae" in args.dataset:
        dataset_nickname = "viquae"
        if "cleaned" in args.dataset:
            with open(f"data/viquae/cleaned_dataset_mc_{model_nickname}.json", "r") as fin:
                dataset = json.load(fin)
        else:
            with open("data/viquae/multiple_choice_data.json", "r") as fin:
                dataset = json.load(fin)
    elif "infoseek" in args.dataset:
        dataset_nickname = "infoseek"
        if "cleaned" in args.dataset:
            with open(f"data/infoseek/cleaned_dataset_mc_{model_nickname}.json", "r") as fin:
                dataset = json.load(fin)
        else:
            with open("data/infoseek/sampled_val_mc.json", "r") as fin:
                dataset = json.load(fin)
    return dataset_nickname, dataset

if __name__ == "__main__":
    from src.utils.parser_utils import get_parser
    
    parser = get_parser()
    args = parser.parse_args()
    dataset = WebQADataset(args)
    dataset.__getitem__(0)