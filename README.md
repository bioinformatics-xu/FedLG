# FedLG
Improving acute coronary syndrome prediction through federated learning with local-global collaboration for mutual correction

<!---
## Non-IID Settings
### Quantity Skew
* While the data distribution may still be consistent amongthe parties, the size of local dataset varies according to Dirichlet distribution.
-->
## Usage
An example to run FedLG:
```
python experiments.py --model=simple-cnn \
    --dataset=coronary \
    --alg=FedLG \
    --lr=0.001 \
    --batch-size=35 \
    --epochs=50 \
    --n_parties=3 \
    --mu=0.01 \
    --rho=0.9 \
    --comm_round=100 \
    --partition=noniid-labeldir \
    --beta=0.5\
    --device='cpu'\
    --datadir='./data/' \
    --logdir='./logs/' \
    --noise=0 \
    --sample=1 \
    --init_seed=0
```

## Description of parameters
| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model` | The model architecture. Options: `simple-cnn`, `vgg`, `resnet`, `mlp`. Default = `mlp`. |
| `dataset`      | Dataset to use. Options: `coronary`. Default = `coronary`. |
| `alg` | The training algorithm. Default = `FedLG`. |
| `lr` | Learning rate for the local models, default = `0.001`. |
| `batch-size` | Batch size, default = `64`. |
| `epochs` | Number of local training epochs, default = `5`. |
| `n_parties` | Number of parties, default = `3`. |
| `mu` | The proximal term parameter for FedProx, default = `1`. |
| `rho` | The parameter controlling the momentum SGD, default = `0`. |
| `comm_round`    | Number of communication rounds to use, default = `50`. |
| `partition`    | The partition way. Options:  `noniid-labeldir`. Default = `noniid-labeldir` |
| `beta` | The concentration parameter of the Dirichlet distribution for heterogeneous partition, default = `0.5`. |
| `device` | Specify the device to run the program, default = `cuda:0`. |
| `datadir` | The path of the dataset, default = `./data/`. |
| `logdir` | The path to store the logs, default = `./logs/`. |
| `noise` | Maximum variance of Gaussian noise we add to local party, default = `0`. |
| `sample` | Ratio of parties that participate in each communication round, default = `1`. |
| `init_seed` | The initial seed, default = `0`. |

## Cite
This project is developed based on Non-NllD, if you find this repository useful, please cite paper:
```
@inproceedings{li2022federated,
      title={Federated Learning on Non-IID Data Silos: An Experimental Study},
      author={Li, Qinbin and Diao, Yiqun and Chen, Quan and He, Bingsheng},
      booktitle={IEEE International Conference on Data Engineering},
      year={2022}
}
```

## Developor

Jia Shang (1243099164@qq.com)

Xiaolu Xu (lu.xu@lnnu.edu.cn)

School of Computer and Artificial Intelligence 

Liaoning Normal University