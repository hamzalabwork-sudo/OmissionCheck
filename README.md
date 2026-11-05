# OmissionCheck

**Omission-evident interbank audit evidence through counterparty co-commitment on blockchains: protocol, smart
contract, experiments and results.**

Anchoring commitments to financial records on a blockchain proves that the records were not changed afterwards. It
cannot prove that nothing was left out. Every interbank transfer, however, has two independent witnesses: the paying
and the receiving bank. *Counterparty co-commitment* turns this redundancy into an omission check that needs no bank to
disclose its books.

In each epoch every bank of a consortium publishes on a shared ledger one fixed-size, masked, invertible sketch (an
invertible Bloom lookup table) of the transfers in its books, counted +1 when it paid and −1 when it received.
Pairwise masks hide each sketch, and the sum of all sketches cancels every transfer that both witnesses recorded
identically. Anyone can therefore read from the ledger exactly the omitted, fabricated or altered transfers, and the
sign shows whose books lack them.

The repository also studies omission as an evasion attack on AI-based anti-money-laundering detection: insiders delete
laundering records before a graph-learning monitor (Multi-GNN) reads them, and the records recovered by the protocol
are put back.

---

## Results at a glance

All numbers are regenerated from `results/paper2/` by `python -m omission.report` (see `paper2/generated/`).

| | |
|---|---|
| Data | IBM AML HI-Small: 5,078,345 synthetic transfers |
| Two-witness coverage | the 1,000 busiest banks cover 84.5 % of laundering legs |
| Detection | 382 hourly epochs, no false alarm; all 1,023 laundering legs deleted by single-bank insiders located |
| Baselines | per-pair reconciliation totals caught 0 % of substitutions; the protocol located 100 % |
| Collusion | coalitions of ten banks could hide 4.5 % of covered laundering legs |
| Ledgers | Hyperledger Besu QBFT, Clique, Ethereum proof-of-stake devnet, live PureChain: 120 of 120 on-chain epochs decoded exactly |
| AML evasion | 20 insider banks cut the PR-AUC of a pooled Multi-GNN monitor by 47 %; DropEdge and omission-aware adversarial training did not reduce the loss |
| Recovery | 73.4 % of deleted records recovered with the correct witness, none wrongly |

---

## Repository map

```
omission/                 the protocol and all experiments
  iblt.py                 signed invertible Bloom lookup tables (vectorised)
  protocol.py             keys, books, masked per-bank sketches, aggregation, decoding, attribution
  data.py                 IBM AML loading, consortium membership, covered transfers and their keys
  run.py                  X1 coverage, X2 detection and localisation, X3 sketch sizing, X4 collusion
  bench.py                X7 per-bank computation micro-benchmarks
  chain.py                X6 end-to-end runs of the OmissionOracle contract on four ledgers
  aml.py, x_aml.py        X8-X10 Multi-GNN monitor: training, insider attacks, protocol recovery, continuous verification
  report.py               tables, figures and LaTeX macros -> paper2/generated, paper2/figures
  queue.sh, pipeline.sh,  job queue, retrying pipeline and memory watchdog used on a shared GPU server
  watchdog.sh
contracts/OmissionOracle.sol   accumulate / publish / commit modes (solc 0.8.19)
finverity/, infra/        shared ledger client (fork-aware RPC, accounts) and Docker launchers for the local ledgers
results/paper2/           raw results of every experiment
paper2/generated/         generated LaTeX tables and macros
paper2/figures/           generated figures and the system design figure (TikZ source F1_design.tex)
scripts/get_data.py       dataset download with checksum verification
```

---

## Reproducing the study

### 1. Environment

Python 3.13 for the protocol and ledger experiments (`requirements.txt`), Docker for the local ledgers, and for the AML
experiments a CUDA machine with the packages in `requirements-ml.txt` (run with Python 3.10, PyTorch 2.5.1, CUDA 12.4).

```bash
python -m venv .venv
pip install -r requirements.txt
pip install --no-deps web3==6.20.4
```

Live PureChain runs need `.env` with `PURECHAIN_RPC`, `PURECHAIN_CHAIN_ID` and `PURECHAIN_PRIVATE_KEY` (zero gas: the
account needs no funds). Never commit `.env`.

### 2. Data

```bash
python scripts/get_data.py ibm_aml  # IBM AML HI-Small -> data/raw/ibm_aml/HI-Small_Trans.csv (Kaggle API)
```

The dataset is available from its authors (Kaggle `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`) and is
not redistributed here.

### 3. Protocol experiments

```bash
python -m omission.run x1 x2 x3 x4                     # coverage, detection, sizing, collusion
python -m omission.bench                               # per-bank computation
python -m omission.chain --backend besu --mode accumulate --epochs 10
#   backends: besu, poa2_local (Clique), ethpos (PoS devnet), purechain (live); modes: accumulate, publish, commit
python -m infra.besu up 4                              # local ledgers: infra/besu.py, infra/poa2.py, infra/ethpos.py
```

On Clique-style chains (PureChain and its emulation) the on-chain sum is read seven blocks deep, the depth that covers
the reorganisations observed on PureChain; a read at the chain head can come from a node that has not yet applied
every publication (`results/paper2/chain/headread/`).

### 4. AML evasion experiments

The detector is IBM's reference Multi-GNN (Egressy et al., AAAI 2024). Clone it and point `MULTIGNN` at it; its model
classes are imported unchanged.

```bash
git clone https://github.com/IBM/Multi-GNN && export MULTIGNN=$PWD/Multi-GNN   # results used commit 252b025
pip install -r requirements-ml.txt
python -m omission.x_aml train  --setting views  --model gin --variant standard --seed 0 --gpu 0 --epochs 50
python -m omission.x_aml attack --setting views  --model gin --variant standard --seed 0 --gpu 0
python -m omission.x_aml stream --setting views  --model gin --variant standard --seed 0 --gpu 0 --every 1
#   settings: global (pooled monitor), views (per-bank monitors); variants: standard, dropedge, omission
```

`AML_WORKERS=0` loads graphs in-process (lower memory). Training checkpoints every epoch and resumes after an
interruption; `omission/pipeline.sh GPU EPOCHS "setting model variant seed" ...` runs train and attack jobs with
retries.

### 5. Tables and figures

```bash
python -m omission.report            # -> paper2/generated/*.tex, paper2/generated/macros.tex, paper2/figures/
```

---

## Citation

See `CITATION.cff`.

## License

Code: MIT (see `LICENSE`). The IBM AML dataset is not redistributed and remains under its original license.
