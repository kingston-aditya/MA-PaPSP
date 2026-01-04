# Copyright - UMD, UCSD

import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import List, Union
from PIL import Image

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import open_clip

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm


import pdb as pdb_original
import glob, json
from itertools import product

import sys
sys.path.insert(1, "/nfshomes/asarkar6/aditya/JANe/")
from dataset import pets_dataloader

logger = get_logger(__name__, log_level="INFO")

class ForkedPdb(pdb_original.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb_original.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/nfshomes/asarkar6/aditya/PRISM/backgrounds/",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pixart-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--task_type",
        type=str,
        default="itm",
        help="itm, class, ccs, cap",
    )

    parser.add_argument(
        "--data",
        type=str,
        default="winoground",
        help="aro, winoground, sugarcrepe, whatsup, foil, coco, flickr, flowers, pets, ucf, ccs",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/nfshomes/asarkar6/trinity/",
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    args = parser.parse_args()
    return args

def get_the_dataset(dataset_name):
    data_call = {"aro": arochecklist_dataloader.return_aroattribute,
                 "winoground": winoground_dataloader.return_winoground,
                 "sugarcrepe": sugarcrepe_dataloader.return_sugarcrepe,
                 "whatsup": whatsup_dataloader.return_whatsup,
                 "foil": foil_dataloader.return_foil,
                 "coco": coco_dataloader.return_coco,
                 "flickr": flickr_dataloader.return_flickr,
                 "flowers": flowers_dataloader.return_flowers,
                 "pets": pets_dataloader.return_pets,
                 "ucf": ucf_dataloader.return_ucf,
                 "ccs": cc12_cc3_sbu_dataloader.return_ccs
                 }
    try:
        precomputed_dataset = data_call[dataset_name]()
    except:
        print(f"Dataset {dataset_name} is currently not supported.")

    return precomputed_dataset

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # load the clip models
    clip_model, preprocess = open_clip.create_model_from_pretrained('hf-hub:'+args.pretrained_model_name_or_path, precision="fp16", cache_dir=args.cache_dir)

    # no need to train image encoders
    clip_model.requires_grad_(False)
    clip_model.to(accelerator.device, dtype=weight_dtype)

    tokenizer = open_clip.get_tokenizer('hf-hub:'+args.pretrained_model_name_or_path)

    def collate_fn_class(batch):
        images = [item["images"] for item in batch]
        labels = [item["labels"] for item in batch]
        return {
            "images": images,
            "labels": labels
        } 
    
    def collate_fn_itm(batch):
        prompt_embeds = [item["prompt_embeds"] for item in batch]
        neg_prompt_embeds = [item["neg_prompt_embeds"] for item in batch]
        images = [item["images"] for item in batch]
        return {
            "prompts": prompt_embeds,
            "neg_prompts": neg_prompt_embeds,
            "images": images
        } 
    
    def collate_fn_cap(batch):
        prompt_embeds = [item["prompt_embeds"] for item in batch]
        images = [item["images"] for item in batch]
        return {
            "prompts": prompt_embeds,
            "images": images
        }
    
    def collate_fn_ccs(batch):
        prompt_embeds = [item["prompt_embeds"] for item in batch]
        return {
            "prompt_embeds": prompt_embeds
        }
    
    if args.task_type == "class":
        collate_fn_t = collate_fn_class
    elif args.task_type == "itm":
        collate_fn_t = collate_fn_itm
    elif args.task_type == "cap":
        collate_fn_t = collate_fn_cap
    elif args.task_type == "ccs":
        collate_fn_t = collate_fn_ccs
    else:
        raise ValueError(f"Task {args.task_type} not supported yet.")
    
    # DataLoaders creation.
    precomputed_dataset = get_the_dataset(args.data)
    train_dataloader = torch.utils.data.DataLoader(
            precomputed_dataset,
            shuffle=False,
            collate_fn=collate_fn_t,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )

    # Prepare everything with our `accelerator`.
    clip_model, train_dataloader = accelerator.prepare(clip_model, train_dataloader)

    clip_model.to(accelerator.device)
    clip_model.eval()

    # Infer!
    total_batch_size = args.train_batch_size * accelerator.num_processes
    args.num_train_epochs = math.ceil(len(train_dataloader) // total_batch_size)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(precomputed_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.num_train_epochs}")

    # Do inference!
    txt_embds = []; img_embds = []; neg_txt_embds = []
    for step, batch in enumerate(tqdm(train_dataloader, desc="Inferring")):
        # encode prompts
        text_prompts = batch["prompts"] if "prompts" in batch.keys() else None
        images = batch["images"] if "images" in batch.keys() else None
        labels = batch["labels"] if "labels" in batch.keys() else None
        neg_prompts = batch["neg_prompts"] if "neg_prompts" in batch.keys() else None

        with torch.no_grad(), torch.amp.autocast():
            # encode text prompt
            if text_prompts is not None:
                text_features = clip_model.encode_text(text_prompts.to(accelerator.device))
                text_features /= text_features.norm(dim=-1, keepdim=True)

            # encode labels
            if labels is not None and step == 0:
                label_features = clip_model.encode_text(labels.to(accelerator.device))
                label_features /= label_features.norm(dim=-1, keepdim=True)
            else:
                label_features = None

            # encode negative prompts
            if neg_prompts is not None:
                neg_text_features = clip_model.encode_text(neg_prompts.to(accelerator.device))
                neg_text_features /= neg_text_features.norm(dim=-1, keepdim=True)

            # encode images
            if images is not None:
                image_features = clip_model.encode_image(images.to(accelerator.device))
                image_features /= image_features.norm(dim=-1, keepdim=True)

            txt_embds.append(text_features.detach().cpu().numpy())
            img_embds.append(image_features.detach().cpu().numpy())
            neg_txt_embds.append(neg_text_features.detach().cpu().numpy())
        
    # save the embeddings
    if len(txt_embds) > 0:
        txt_embds = np.concatenate(txt_embds, axis=0)
        np.save(os.path.join(args.output_dir, f"{args.dataset_name}_txt_embds.npy"), txt_embds)
    
    if len(neg_txt_embds) > 0:
        neg_txt_embds = np.concatenate(neg_txt_embds, axis=0)
        np.save(os.path.join(args.output_dir, f"{args.dataset_name}_negtxt_embds.npy"), neg_txt_embds)
    
    if len(img_embds) > 0:
        img_embds = np.concatenate(img_embds, axis=0)
        np.save(os.path.join(args.output_dir, f"{args.dataset_name}_img_embds.npy"), img_embds)
    
    if label_features is not None:
        label_features = label_features.detach().cpu().numpy().reshape(1,-1)
        
    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)