# OmissionCheck

**Making omissions in interbank books evident on a blockchain without disclosing the books: protocol, smart contract,
zero-knowledge binding proof, experiments and results.**

Anchoring commitments to financial records on a blockchain proves that the records were not changed afterwards. It
cannot prove that nothing was left out. Every interbank transfer, however, has two independent witnesses: the paying
and the receiving bank. *Counterparty co-commitment* turns this redundancy into an omission check that needs no bank to
disclose its books.

In each epoch every bank of a consortium first commits on a shared ledger to the Merkle root of its books. Only then is
an epoch seed drawn from the ledger; it keys a fixed-size invertible sketch (an invertible Bloom lookup table) into
which the bank inserts each transfer with weight +1 as payer and −1 as payee. Pairwise masks hide each sketch, sketches
are published by commit-then-reveal, and their sum cancels every transfer that both witnesses recorded identically.
Anyone can therefore read from the ledger exactly the omitted, fabricated or altered transfers, and the sign shows
whose books lack them. Each bank can attach a Groth16 proof that its sketch was computed from the books it committed.

The repository also includes a case study on graph-learning anti-money-laundering monitors: insiders delete laundering
records before the monitor reads them, and the records recovered by the protocol are put back.

---

## Results at a glance

All numbers are regenerated from `results/paper2/` by `python -m omission.report` (see `paper2/generated/`) and, for
the binding proof, by `python -m omission.zk.run` (see `paper2/zk/`).

| | |
|---|---|
| Data | IBM AML HI-Small: 5,078,345 synthetic transfers |
| Two-witness coverage | the 1,000 busiest banks cover 84.5 % of laundering legs; 10 to 50 banks only 1.5 % to 10.1 % |
| Detection | 382 hourly epochs, no false alarm; all 1,023 laundering legs deleted by single-bank insiders located |
| Baselines | per-pair reconciliation totals caught 0 % of substitutions; the protocol located 100 % |
| Adaptive insider | against fixed public hash functions one insider hid an omission behind 157 fabricated entries; with books committed before the seed, the same books were flagged under 100 % of 1,000 fresh seeds |
| Benign discrepancies | decoded alerts mixed with an insider's deletions had a precision of 4.1 % when epochs closed 15 minutes after their end and 99.6 % after 60 minutes |
| Collusion | the optimal ten-bank coalition hides 4.8 % of covered laundering legs |
| Ledgers | Hyperledger Besu QBFT, Clique, Ethereum proof-of-stake devnet, live PureChain: 59 of 59 on-chain sums decoded exactly; commit-then-reveal adds 114,749 gas per bank and epoch |
| Binding proof | Groth16 (circom, snarkjs): books of 256 entries, which hold 99.4 % of bank-epochs, proved in 57 s with 6.4 GB; a 256-byte proof verified on chain for 407,917 gas |
| AML case study | 20 insider banks cut pooled F1 by 19.8 ± 0.8 points (Multi-GIN+EU, 5 seeds) and 22.9 ± 1.6 (Multi-PNA+EU, 3 seeds); DropEdge (17.4 ± 0.3) and omission-aware training (20.3 ± 0.6) lost about as much |
| Where the loss lies | laundering left in the books was flagged as often after deletion as before (recall moved by at most 3.7 points): deletion removes evidence, it does not mislead the model |
| Recovery | 73.5 % of deleted records recovered with the correct witness, none wrongly |

---

## Repository map

```
omission/                 the protocol and all experiments
  iblt.py                 signed invertible Bloom lookup tables (vectorised)
  protocol.py             keys, books, masked per-bank sketches, aggregation, decoding, attribution
  data.py                 IBM AML loading, consortium membership, covered transfers and their keys
  run.py                  X1 coverage, X2 detection, X3 sketch sizing, X4 collusion (greedy and exact),
                          X5 benign discrepancies, X12 scale
  attacks.py              X9 adaptive attacks: decoys against the naive design and the protocol, last mover, dropout
  bench.py                X7 per-bank computation micro-benchmarks
  chain.py                X6 end-to-end runs of the OmissionOracle contract on four ledgers
  aml.py, x_aml.py        X8-X11 Multi-GNN monitors: training, insider attacks, protocol recovery, validation-optimal
                          threshold, spill-over onto the laundering left in the books, continuous verification
  zk/                     zero-knowledge binding of sketches to committed books: field sketch and Poseidon reference
                          (field.py, fsketch.py), constraint model and relation checks (circuit.py), circom circuit
                          (circuits/sketch_binding.circom), proving (prove.py), verifier gas (gas.py), macros (run.py)
  report.py               tables, figures and LaTeX macros -> paper2/generated, paper2/figures
  queue.sh, lane.sh,      job queue, memory-gated GPU lanes, retrying pipeline and memory watchdog used on a shared
  pipeline.sh,            GPU server; revise*.sh are the job lists that produced the published AML results
  watchdog.sh
contracts/OmissionOracle.sol   accumulate / publish / commit modes (solc 0.8.19)
finverity/, infra/        shared ledger client (fork-aware RPC, accounts) and Docker launchers for the local ledgers
results/paper2/           raw results of every experiment (results/paper2/aml: Multi-GNN runs)
paper2/generated/         generated LaTeX tables and macros
paper2/figures/           generated figures, the system design figure (F1_design.tex) and the protocol message
                          sequence (F4_protocol.tex)
paper2/zk/                constraint counts, proving and verifier-gas measurements of the binding circuit
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
python -m omission.run x1 x2 x3 x4 x4exact x5 x12     # coverage, detection, sizing, collusion, benign, scale
python -m omission.attacks decoy lastmover dropout     # adaptive attacks
python -m omission.bench                               # per-bank computation
python -m omission.chain --backend besu --mode accumulate --epochs 10
#   backends: besu, poa2_local (Clique), ethpos (PoS devnet), purechain (live); modes: accumulate, publish, commit
python -m infra.besu up 4                              # local ledgers: infra/besu.py, infra/poa2.py, infra/ethpos.py
```

On Clique-style chains (PureChain and its emulation) the on-chain sum is read seven blocks deep, the depth that covers
the reorganisations observed on PureChain; a read at the chain head can come from a node that has not yet applied
every publication (`results/paper2/chain/headread/`).

`infra/purechain_local.py` runs the released PureChain client locally. It needs `purechain-geth-v1.0.1-linux-amd64`
and `autoMine-v2.js` from the v1.0.1 release of `github.com/gieworld/purechain_testnet` in `build/purechain/`; the
binary's SHA-256 is checked before use. Neither file is redistributed here.

### 4. Zero-knowledge binding proof

```bash
python -m omission.zk.run                              # constraint model, relation checks, macros
python -m omission.zk.prove --tools <dir> --ptau <powers-of-tau file> --sizes 64 256
#   <dir> holds bin/circom (2.2.2) and node_modules with snarkjs 0.7.5, circomlib 2.0.5 and circomlibjs 0.1.7;
#   POSEIDON_CONSTANTS may point to circomlibjs/src/poseidon_constants.json
python -m omission.zk.gas --real <build dir of a proved size>   # verifier gas on a local ledger
```

The published timings used a local single-contributor powers of tau (2^20): it serves for timing only and is not a
trusted setup.

### 5. AML case study

The detectors are IBM's reference Multi-GNN models (Egressy et al., AAAI 2024). Clone it and point `MULTIGNN` at it;
its model classes are imported unchanged.

```bash
git clone https://github.com/IBM/Multi-GNN && export MULTIGNN=$PWD/Multi-GNN   # results used commit 252b025
pip install -r requirements-ml.txt
python -m omission.x_aml train  --setting views  --model gin --variant standard --seed 0 --gpu 0 --epochs 50
python -m omission.x_aml attack --setting views  --model gin --variant standard --seed 0 --gpu 0
python -m omission.x_aml spill  --setting views  --model gin --variant standard --seed 0 --gpu 0
python -m omission.x_aml stream --setting views  --model gin --variant standard --seed 0 --gpu 0 --every 1
#   settings: global (pooled monitor), views (per-bank monitors); models: gin, pna;
#   variants: standard, dropedge, omission
```

`AML_WORKERS=0` loads graphs in-process (lower memory). Training checkpoints every epoch and an attack saves every
finished scenario, so both resume after an interruption; `omission/lane.sh GPU GATE EPOCHS "kind setting model variant
seed" ...` runs train and attack jobs on one GPU, starting each only when enough memory is free.

### 6. Tables and figures

```bash
python -m omission.report            # -> paper2/generated/*.tex, paper2/generated/macros.tex, paper2/figures/
```

---

## Citation

See `CITATION.cff`.

## License

Code: MIT (see `LICENSE`). The IBM AML dataset and the PureChain client are not redistributed and remain under their
original licenses.
