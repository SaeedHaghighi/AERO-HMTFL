# Hierarchical Multi-Task Federated Learning Simulation

This repository contains a Python implementation of a hierarchical multi-task federated learning simulation for vehicular networks. The system models vehicles, cluster heads, and an EPC layer, with Kafka used for inter-node communication.

## Repository structure

```text
.
├── mftl/
│   ├── data/
│   │   └── data_loader.py
│   ├── kafka/
│   │   ├── kafka_config.py
│   │   ├── kafka_consumer.py
│   │   ├── kafka_producer.py
│   │   └── kafka_topics.py
│   ├── models/
│   │   ├── autoencoder.py
│   │   ├── cluster_head.py
│   │   ├── cnn_models.py
│   │   ├── epc.py
│   │   └── vehicle.py
│   ├── simulation/
│   │   └── simulator.py
│   ├── utils/
│   │   └── helpers.py
│   └── config.py
├── .env.example
├── .gitignore
├── requirements.txt
└── run_simulation.py
```

## Requirements

- Python 3.10 or newer is recommended.
- A reachable Apache Kafka broker is required for Kafka-based communication.
- The first run may download CIFAR-10 and GTSRB through `torchvision`.

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Kafka configuration

The Kafka broker address and consumer group can be configured through environment variables. Copy `.env.example` values into your shell or environment manager and update the broker address when needed.

Linux/macOS example:

```bash
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
export KAFKA_GROUP_ID=mtfl-fl-group
```

PowerShell example:

```powershell
$env:KAFKA_BOOTSTRAP_SERVERS="localhost:9092"
$env:KAFKA_GROUP_ID="mtfl-fl-group"
```

## Running the simulation

Start Kafka first, then run:

```bash
python run_simulation.py
```

The simulator can also be configured directly in Python:

```python
from mftl import Simulator

simulator = Simulator(
    num_vehicles=40,
    cluster_size=5,
    attacker_fraction=0.1,
    rounds=100,
)

ch_accuracy, cm_accuracy, epc_accuracy = simulator.run()
```

## Main components

- `Vehicle`: local training, task models, mobility state, reliability state, and Kafka communication.
- `ClusterHead`: cluster-level collection and aggregation of vehicle updates.
- `EPC`: higher-level aggregation and global model redistribution.
- `Autoencoder`: representation used by the model aggregation workflow.
- `Simulator`: vehicle initialization, clustering, dataset preparation, communication rounds, and evaluation.

## Datasets

The implementation uses scikit-learn digits data and includes loaders for CIFAR-10 and GTSRB through `torchvision`. Dataset files are ignored by Git and are downloaded locally when required.

## Reproducibility

Default simulation parameters are defined in `mftl/config.py`. Kafka parameters are defined in `mftl/kafka/kafka_config.py` and can be overridden through environment variables.
