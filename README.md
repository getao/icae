This repository contains the code, data, and models pertaining to our paper ["In-context Autoencoder for Context Compression in a Large Language Model"](https://arxiv.org/abs/2307.06945), which has been accepted by ICLR 2024.

![ICAE Illustration](./icae_demo.png)

## Resources

* [Model](https://huggingface.co/sggetao/icae/tree/main)
* [Data](https://huggingface.co/datasets/sggetao/PwC)

## Updates

### Version 2 (March 2024)
Two new ICAE models based on Mistral-7B were released. These include a pretrained model and another fine-tuned with instructions. The inference code accompanying these models is also provided.

The Mistral-7B ICAE models extend support to multi-span concatenation, as illustrated in Figure 6 of the paper.

### Version 1 (September 2023)
This is the initial release of the In-context Autoencoder repository, which includes the PwC dataset, ICAE model code, and the Llama-2-7b-chat fine-tuned ICAE model evaluated in the paper.

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
