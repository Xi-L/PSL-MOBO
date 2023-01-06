# PSL-MOBO

Code for NeurIPS2022 Paper: Pareto Set Learning for Expensive Multi-Objective Optimization

The code is mainly designed to be simple and readable, it contains:

- <code>run.py</code> is a ~200-line main file to run the Pareto Set Learning (PSL) algorithm for MOBO;
- <code>model.py</code> is a simple FC Pareto Set model definition;
- <code>function.py</code> contains all the test problems used in the paper;
- <code>lhs.py</code> is an efficient latin-hypercube design implementation, which is for generating initial solutions;
- The folder <code>mobo</code> contains the files for surrogate model definition and training, which is borrowed from the [DGEMO repository](https://github.com/yunshengtian/DGEMO).


**Reference**

If you find our work is helpful to your research, please cite our paper:
```
@inproceedings{linpareto,
  title={Pareto Set Learning for Expensive Multi-Objective Optimization},
  author={Lin, Xi and Yang, Zhiyuan and Zhang, Xiaoyuan and Zhang, Qingfu},
  booktitle={Advances in Neural Information Processing Systems},
  volume={35},
  year={2022}
}
```
