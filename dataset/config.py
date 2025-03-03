# model_type - openclip model type
# pretrain - pretrain dataset
# device - don't change

def get_config():
    return {
        "batch_size": 200,
        "embed_size": 512,
        "shard_num": 50,
        "model_type":"ViT-B/16",
        "pretrain":"openai",
        "device":"cuda",
        "repo_dir":"/data/aditya/JANe/",
        "retrieval_size": 10,
        "FEx": 0.8,
        "out_pth": "/data/aditya/JANe/data_files/",
        "data_dir": "/mnt/ssd/"
    }