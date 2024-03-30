This repository contains the code, data, and models pertaining to our paper ["In-context Autoencoder for Context Compression in a Large Language Model"](https://arxiv.org/abs/2307.06945), which has been accepted by ICLR 2024.

![ICAE Illustration](./icae_demo.png)

## Resources

* [Model](https://huggingface.co/sggetao/icae/tree/main)
* [Data](https://huggingface.co/datasets/sggetao/PwC)

## Updates

### Version 2 (March 2024)
Two new ICAE models based on Mistral-7B were released. These include a pretrained model an instruction fine-tuned model. The inference code accompanying these models is also provided.

Compared with the V1 released model, the Mistral-7B ICAE models extend support to multi-span concatenation, as illustrated in Figure 6 of the paper.

In the release of V2, I move the dataset and models to my [huggingface repo](https://huggingface.co/sggetao).

### Version 1 (September 2023)
This is the original release of the In-context Autoencoder repository. This particular iteration includes the PwC dataset, the code, and a fine-tuned ICAE model based on Llama-2-7b-chat that is used in the paper.

The first version of the ICAE model is based in the Llama-2-7b-chat, which is used in the paper. It can be downloaded from [this link](https://huggingface.co/sggetao/icae/resolve/main/llama-2-7b-chat-finetuned-icae_zeroweight_llama2.pt). Please use the model with the code available in the [code/icae_v1](https://github.com/getao/icae/tree/main/code/icae_v1) directory.

## Cite Us

If our work contributes to your research, please cite our paper:

```bibtex
@article{ge2023context,
  title={In-context autoencoder for context compression in a large language model},
  author={Ge, Tao and Hu, Jing and Wang, Xun and Chen, Si-Qing and Wei, Furu},
  journal={arXiv preprint arXiv:2307.06945},
  year={2023}
}
```
