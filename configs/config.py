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
        "batch_size": 10,
        "num_epochs": 20,
        "lr": 10**-4,
        "embed_size": 512,
        "model_folder": "/data/aditya/weights/",
        "model_filename": "tmodel_",
        "SA_number": 4,
        "CA_number": 2,
        "pipeline": 1,
        "coco_train_data":"/data/aditya/coco_embeds/coco_img_feat_1.npy",
        "coco_label_data":"/data/aditya/coco_embeds/coco_txt_feat_1.npy",
        "cc12m_img_folder":"",
        "cc12m_txt_folder":""
    }

def get_weights_file_name(config, epoch):
    model_folder = config["model_folder"]
    model_basename = config["model_filename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return os.path.join(model_folder, model_filename)