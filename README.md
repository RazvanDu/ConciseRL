# ConciseRL: Conciseness‑Guided Reinforcement Learning for Efficient Reasoning Models

The paper is available at: https://arxiv.org/abs/2505.17250.

This repository contains the official implementation for our paper: **“ConciseRL: Conciseness‑Guided Reinforcement Learning for Efficient Reasoning Models”.**  

The codebase is **forked from** [Zanette‑Labs/efficient-reasoning](https://github.com/Zanette-Labs/efficient-reasoning) and extends it with a new reward family that explicitly favours *concise yet correct* chain‑of‑thought (CoT) traces.

We release:

* **Training & Evaluation Scripts** supporting three reward modes (`sigmoid`, `GPT_Score3`, `GPT_Separated`) with plug‑and‑play toggles;
* **Pre‑configured Slurm Launchers** that accept `OPENAI_API_KEY`, `REWARD_TYPE`, and `MODEL_API` as environment variables;

The implementation has been validated on 4xA100 GPUs with **Python 3.10.12**, **CUDA 12.4**, and **PyTorch 2.5.1**.

---

## 1. Installation

```bash
# Clone repository (SSH or HTTPS)
git clone git@github.com:RazvanDu/ConciseRL.git
cd ConciseRL

# Create and activate conda env
conda create -n concise_rl python=3.10.12 -y
conda activate concise_rl

# Install latex -> sympy converter (needed for Math reward evaluation)
cd utils/latex2sympy
pip install -e .
cd ../../

# Install dependencies
pip install wheel packaging
pip install torch==2.5.1
pip install flash_attn==2.7.0.post2 --no-build-isolation
pip install -r requirements.txt
```

---

## 2. Dataset

Download the *compression* dataset used in the paper:

```bash
huggingface-cli download daman1209arora/compression_dataset \
  --repo-type dataset \
  --local-dir datasets/compression_dataset
```

The split merges items from **MATH**, **CN‑K12**, **AIME**, **AoPS**, and **Olympiad** subsets of *Numina Math*, mirroring the upstream project.

---

## 3. Model Weights

We will soon release model weights for the key models evaluated in the paper.

---

## 4. Reward Modes

| REWARD_TYPE     | Description                                                                                                                                      |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `GPT_Score3`    | **ConciseRL (gated).** Three‑component mixture that grants no credit if the answer is incorrect, and it uses our score if the answer is correct. |
| `GPT_Separated` | **ConciseRL (ungated).** Linear penalty without the hard gate at the answer token.                                                               |
| `sigmoid`       | Original score from the baseline repository.                                                                                                     |

---

## 5. Quick Start

### 5.1. Training

We provide Slurm launchers that forward the OpenAI key, reward type, and model type to the trainer.

```bash
# Single‑node 4xGH200 example
export WANDB_KEY="<your‑wandb‑key>"
export OPENAI_API_KEY="<your‑openai‑key>"
export REWARD_TYPE="GPT_Score3"                      # GPT_Score3 | GPT_Separated | sigmoid
export MODEL_API="gpt-4.1-mini-2025-04-14"           # Specify API model to use (e.g. gpt-4o, gpt-4.1-mini)

sbatch run_rloo_1.5B.sh
```

If you do not use Slurm, translate the `#SBATCH` headers into an `accelerate launch` or `torchrun` command.

### 5.2. Evaluation

```bash
python evaluate_model.py \
  --model_path runs/scale:1.5B_alpha:0.1_reward:${REWARD_TYPE} \
  --dataset    openai/gsm8k \
  --scale      1.5B
```

The script reports **accuracy** and **mean generated tokens** side‑by‑side.

---

## 6. Citation

If you use this code, please cite our paper.

```
Dumitru, R.-G., Peteleaza, D., Yadav, V., & Pan, L. (2025). ConciseRL: Conciseness-Guided Reinforcement Learning for Efficient Reasoning Models. arXiv preprint arXiv:2505.17250. https://arxiv.org/abs/2505.17250
```

### BibTeX

```
@misc{dumitru2025conciserlconcisenessguidedreinforcementlearning,
      title={ConciseRL: Conciseness-Guided Reinforcement Learning for Efficient Reasoning Models}, 
      author={Razvan-Gabriel Dumitru and Darius Peteleaza and Vikas Yadav and Liangming Pan},
      year={2025},
      eprint={2505.17250},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.17250}, 
}
```

### RIS

```
TY  - MISC
AU  - Dumitru, Razvan-Gabriel
AU  - Peteleaza, Darius
AU  - Yadav, Vikas
AU  - Pan, Liangming
TI  - ConciseRL: Conciseness-Guided Reinforcement Learning for Efficient Reasoning Models
PY  - 2025
UR  - https://arxiv.org/abs/2505.17250
DO  - 10.48550/arXiv.2505.17250
ID  - dumitru2025conciserlconcisenessguidedreinforcementlearning
ER  - 
```

---

## 6. License

This project is licensed under **Apache 2.0**, identical to the upstream codebase.

---

*Happy concise reasoning!*
