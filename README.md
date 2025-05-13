# SIRGCN

The source codes and datasets for **SIRGCN - Susceptible Infectious Recovered Graph Neural Network**.

The code using two datasets from Japan and United States, which are also used in Colagnn and EpiGNN. 

---

## 1. Quick Start

All programs are implemented using:
- Python 3.8.5
- PyTorch 1.9.1 with CUDA 11.1 (1.9.1 cu111)

### Install Pytorch and Dependencies

```bash
pip install -r requirements.txt
```

### Download the dataset


### Run the US-Region Dataset as an Example

```bash
cd SIRGCN
bash run_sir.sh
```

### 3.1 Parameters

horizon=1
gpu=0
dataset=japan or state360: dataset
sim_mat=japan-adj or state-adj-49: adj matrix
seed=420: random seed

---


