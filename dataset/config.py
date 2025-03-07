# model_type - openclip model type
# pretrain - pretrain dataset
# device - don't change

def get_config():
    return {
        "batch_size": 256,
        "embed_size": 512,
        "shard_num": 1,
        "model_type":"ViT-B-16",
        "pretrain":"dfn2b",
        "device":"cuda",
        "repo_dir":"/data/aditya/JANe/",
        "retrieval_size": 10,
        "FEx": 0.8,
        "out_pth": "/data/datasets/final_set/pets/train/dfn/",
        "data_dir": "/data/datasets/oxford_pets/"
    }