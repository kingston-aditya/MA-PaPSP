from datasets import load_dataset
from glob import glob
import torch
import PIL
import io
import torchvision.transforms as v2
from datasets import Image
from tokenizer import SimpleTokenizer
from config import get_config
conf = get_config()
import sys
sys.path.insert(1, conf["repo_dir"])
import os
# import pdb

tokenizer = SimpleTokenizer()

image_transform = v2.Compose(
    [
        v2.Resize(224),
        v2.CenterCrop(224),
        v2.ToTensor(),
        v2.Normalize([0.5], [0.5]),
    ]
)

def t2i_process_fn(batch):
    images = batch["image"]
    # pdb.set_trace()
    # print(batch.keys())
    captions = batch["caption"]
    batch_size = len(images)
    # print("batch_size",batch_size)
    batch["labels"] = torch.zeros(batch_size, dtype=torch.long)
    # clip_model = openclip_embeds(conf["model_type"], conf["pretrain"], conf["device"])
    for i in range(len(images)):
        try:
            images[i] = PIL.Image.open(io.BytesIO(images[i]["bytes"]) if images[i]["bytes"] is not None else images[i]["path"]).convert('RGB')
        except:
            print("corrupt!!!!")
            images[i] = None
            captions[i] = ""
    batch["caption"] = tokenizer(batch["caption"])
    batch["image"] = [image_transform(image) if image is not None else None for image in images]    
    # batch["caption"], batch["attn_mask"] = model.tokenize(captions)

    while all(x is None for x in batch["image"]):
        randidx = torch.randint(0, len(train_dataset), (1,)).item()
        batch = train_dataset[randidx]
        # Expand single items into batches
        batch = {key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value] for key, value in batch.items()}

    return batch
    # return batch["image"], batch["caption"], batch["labels"]


def return_cc12_train_dataset():
    data_files = glob(os.path.join(conf["data_dir"], "*.tar"))
    train_dataset = load_dataset(
        "webdataset",
        data_files=data_files,
        cache_dir=conf["data_dir"],
        split="train",
        # streaming=True
    )

    # train_dataset = train_dataset.take(1000000)
    train_dataset = train_dataset.rename_column("jpg", "image")
    train_dataset = train_dataset.rename_column("txt", "caption")
    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if not col in (["image", "caption"])])


    train_dataset = train_dataset.cast_column("image", Image(decode=False))
    train_dataset.set_transform(t2i_process_fn)

    return train_dataset
