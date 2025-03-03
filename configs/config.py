import os

# help
# lr -> learning rate
# embed_size -> CLIP embedding size
# model_folder and model_filename -> file where to save weights
# SA number -> number of SA blocks
# CA number -> number of CA blocks
# pipeline -> type of pipeline

def get_config():
    return {
        "batch_size": 200,
        "embed_size": 512,
        "repo_dir":"/data/aditya/JANe/",
        "retrieval_size": 10,
        "FEx": 0.8,
        "out_pth": "/data/aditya/JANe/data_files/"
    }

def get_weights_file_name(config, epoch):
    model_folder = config["model_folder"]
    model_basename = config["model_filename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return os.path.join(model_folder, model_filename)