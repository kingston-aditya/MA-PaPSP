## Leveraging Data to Say No: Memory Augmented Plug-and-Play Selective Prediction

<p align="center">
    <img src="assets/teaser.png" width="50%">
</p>

[Aditya Sarkar](https://kingston-aditya.github.io/) &emsp;
[Yi Li](http://www.svcl.ucsd.edu/people/yili/) &emsp;
[Jiacheng Cheng](http://www.svcl.ucsd.edu/people/jiacheng/) <sup>†</sup> &emsp;
[Shlok Kumar Mishra](https://shlokk.github.io/shlokmishra.github.io/) &emsp;
[Nuno Vasconcelos](http://www.svcl.ucsd.edu/people/nuno/)

__NOTE - The repo is under development.__

<sup>†</sup> - corresponding author

## Installation
```bash
conda env create -f environment.yml
conda activate metaquery
```

## Inference
If you want to do inference with the model on a single node, you can use the following command.

```bash
OMP_NUM_THREADS=12 torchrun --nproc-per-node=8 train.py \
    --run_name test \
    --config_file llavaov0p5_sana.yaml \
    --base_dir /path/to/metaquery
```

If you want to do inference with the model on multiple nodes, use this command.

```bash
OMP_NUM_THREADS=12 torchrun --nproc-per-node=8 train.py \
    --run_name test \
    --config_file llavaov0p5_sana.yaml \
    --base_dir /path/to/metaquery
```

## License

The code is licensed [CC-by-NC](LICENSE). Third party content pulled from other locations are subject to their own licenses and you may have other legal obligations or restrictions that govern your use of that content.

## Citation
If you find MA-PaPSP useful for your research and applications, please cite using this BibTeX:

```bibtex
@article{sarkar2026mapapsp,
  title={Transfer between modalities with metaqueries},
  author={Pan, Xichen and Shukla, Satya Narayan and Singh, Aashu and Zhao, Zhuokai and Mishra, Shlok Kumar and Wang, Jialiang and Xu, Zhiyang and Chen, Jiuhai and Li, Kunpeng and Juefei-Xu, Felix and Hou, Ji and Xie, Saining},
  journal={arXiv preprint arXiv:2504.06256},
  year={2025}
}
```



