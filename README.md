# Normal Structure Regularisation for Open-set Supervised GAD (ICLR 2025)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
[![arXiv](https://img.shields.io/badge/NSReg-2310.08041-b31b1b.svg)](https://arxiv.org/abs/2311.06835)

This is the offical Pytorch Implementation of ICLR 2025 paper Open-Set Graph Anomaly Detection via Normal Structure Regularisation. 

By Qizhou Wang, Guansong Pang, Mahsa Salehi, Xiaokun Xia, Christopher Leckie.

## Requirements
Please see the env.yml file.

## Installation
```
conda env create -f env.yml
```

## Usage
Please downlaod the dataset and set the path in exp/config/mag_cs/dset.yaml before running the code.

Please use the following script to run the training code:

```bash
# bash <run_script_name> <mode> <meta_config_name>
bash run_scripts/mag_cs/run.sh run meta_mag_cs
```

## 📝 Citation
If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{wang2024nsreg,
  title={Open-Set Graph Anomaly Detection via Normal Structure Regularisation}, 
  author={Qizhou Wang and Guansong Pang and Mahsa Salehi and Xiaokun Xia and Christopher Leckie},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year={2025},
}
```

<!-- ## Contact
For questions about the paper or implementation, please open an issue or contact:
- [Your Name](mailto:your.email@institution.edu) -->

## 🧾 License
This repository is released under the Apache 2.0 license as found in the [LICENSE](./LICENSE) file.

---
**Note:** This repository is under active development. Code and detailed documentation will be released shortly.
