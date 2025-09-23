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
from transformers import AutoModelForCausalLM, AutoTokenizer

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
from dataset import coco_dataloader, flickr_dataloader, flowers_dataloader, pets_dataloader, ucf_dataloader
from dataset import foil_dataloader, arochecklist_dataloader, winoground_dataloader, sugarcrepe_dataloader, whatsup_dataloader
from dataset import cc12_cc3_sbu_dataloader

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
        "--cache_dir",
        type=str,
        default="/nfshomes/asarkar6/trinity/",
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
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
    data_call = {"coco": coco_dataloader.return_coco,
                 "flickr": flickr_dataloader.return_flickr,
                 }
    try:
        precomputed_dataset = data_call[dataset_name]()
    except:
        print(f"Dataset {dataset_name} is currently not supported.")

    return precomputed_dataset

def create_messages(prompts):
    main_prompt1 = "You are Qwen, created by Alibaba Cloud. You are an intelligent and helpful AI assistant that can do the following task."
    main_prompt2 = "Given an input context describing a scene, your task is to identify all the noun phrases in the sentence and then create a new sentence using different noun phrases."
    main_prompt3 = "The new sentence must meet the following two requirements: \n 1. It must be fluent and grammatically correct. \n 2. It must make logical sense."
    main_prompt4 = "Here is one example: \n Sentence: A man and a woman are walking together. \n Answer: A child and a dog are walking together."
    main_prompt5 = "Explanation: There are two noun phrases in the original sentence: man and woman. In the new sentence, they are replaced with child and dog, respectively. The new sentence is fluent and grammatically correct, and it makes logical sense."
    main_prompt6 = "Generate 5 sentences for this prompt. Do NOT give any explanations. \n Sentence:"
    main_prompt = main_prompt1 + "\n" + main_prompt2 + "\n" + main_prompt3 + "\n" + main_prompt4 + "\n" + main_prompt5 

    messages = []
    for prompt in prompts:
        final_msg = [
            {"role": "system", "content": main_prompt},
            {"role": "user", "content": main_prompt6 + prompt + "\n Answer:"},
        ]
        messages.append(final_msg)
    return messages


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
    llm_model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=weight_dtype, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    # no need to train image encoders
    llm_model.requires_grad_(False)
    llm_model.to(accelerator.device, dtype=weight_dtype)
    
    def collate_fn_cap(batch):
        prompts = [item["prompts"] for item in batch]
        return {
            "prompts": prompts
        }
    
    # DataLoaders creation.
    precomputed_dataset = get_the_dataset(args.data)
    train_dataloader = torch.utils.data.DataLoader(
            precomputed_dataset,
            shuffle=False,
            collate_fn=collate_fn_cap,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )

    # Prepare everything with our `accelerator`.
    llm_model, train_dataloader = accelerator.prepare(llm_model, train_dataloader)

    llm_model.to(accelerator.device)
    llm_model.eval()

    # Infer!
    total_batch_size = args.train_batch_size * accelerator.num_processes
    args.num_train_epochs = math.ceil(len(train_dataloader) // total_batch_size)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(precomputed_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.num_train_epochs}")

    # Do inference!
    neg_txt_caps = []
    for _, batch in enumerate(tqdm(train_dataloader, desc="Inferring")):
        prompts = batch["prompts"]

        with torch.no_grad(), torch.amp.autocast():
            # encode text prompt
            output_prompts = create_messages(prompts)
            output_prompts = tokenizer.apply_chat_template(output_prompts, tokenize=False, aadd_generation_prompt=True)

            model_inputs = tokenizer(output_prompts, return_tensors="pt", padding="max_length", truncation=True).to(accelerator.device)
            generated_ids = llm_model.generate(**model_inputs, max_new_tokens=512)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

            # decode text prompt
            neg_txt_caps.append(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])

    np.save(os.path.join(args.output_dir, f"{args.dataset_name}_qwen1b_negtxt_caps.npy"), np.array(neg_txt_caps))
    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)