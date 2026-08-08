



import random
import threading
import time
from typing import List, Tuple, Dict, Optional, Any, Set
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import SGDClassifier

from mftl.utils.helpers import (
    TwoTaskWeights,
    split_tasks_2way,
    get_classes_for_task,
    _sanitize_vector
)
from mftl.models.cnn_models import LightCNN
from mftl.models.autoencoder import Autoencoder
from mftl.kafka import (
    KafkaProducerWrapper,
    KafkaConsumerWrapper,
    VehicleInfoCollector,
    TOPIC_CM_TO_CH,
    TOPIC_CH_TO_CM,
    TOPIC_EPC_TO_CH,
    TOPIC_CLUSTER_UPDATE,
    TOPIC_RELIABILITY,
    HELLO_PACKET_INTERVAL
)






class SharedEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent_dim), nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SharedDecoder(nn.Module):
    def __init__(self, latent_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class TaskHead(nn.Module):
    def __init__(self, latent_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


class MTFLModel(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 128,
                 task_names: List[str] = None, num_classes: int = 10):
        super().__init__()
        self.encoder = SharedEncoder(input_dim, latent_dim)
        self.decoder = SharedDecoder(latent_dim, input_dim)
        self.task_names = task_names or ['task_a', 'task_b']
        self.task_heads = nn.ModuleDict({
            name: TaskHead(latent_dim, num_classes) for name in self.task_names
        })
        self.latent_dim = latent_dim
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, task_name: str):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        y_hat = self.task_heads[task_name](z)
        return z, x_hat, y_hat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward_task(self, z: torch.Tensor, task_name: str) -> torch.Tensor:
        return self.task_heads[task_name](z)

    def get_ae_params_flat(self) -> np.ndarray:
        params = []
        for name, param in self.named_parameters():
            if 'encoder' in name or 'decoder' in name:
                params.append(param.detach().cpu().numpy().ravel())
        return np.concatenate(params) if params else np.array([])

    def set_ae_params_from_flat(self, flat_params: np.ndarray) -> bool:
        try:
            flat_params = _sanitize_vector(flat_params)
            idx = 0
            for name, param in self.named_parameters():
                if 'encoder' in name or 'decoder' in name:
                    size = param.numel()
                    if idx + size > len(flat_params):
                        return False
                    param.data = torch.from_numpy(
                        flat_params[idx:idx+size].reshape(param.shape)
                    ).float().to(param.device)
                    idx += size
            return True
        except Exception:
            return False






class Vehicle:


    def __init__(
        self,
        source_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        is_attacker: bool = False,
        cnn_device: Optional[torch.device] = None,
        cifar10_train_loader: Optional[DataLoader] = None,
        cifar10_test_loader: Optional[DataLoader] = None,
        gtsrb_train_loader: Optional[DataLoader] = None,
        gtsrb_test_loader: Optional[DataLoader] = None,
        gtsrb_num_classes: int = 43,
        use_mtfl: bool = True,
        latent_dim: int = 128,
        public_buffer: Optional[Dict[str, DataLoader]] = None,
        task_names: Optional[List[str]] = None,
        mtfl_num_classes: int = 10,
        transmission_range: float = 100.0
    ) -> None:

        self.source_id = source_id
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.X_test = np.asarray(X_test)
        self.y_test = np.asarray(y_test)

        self.is_attacker = is_attacker
        self.transmission_range = transmission_range


        self.role = "SE"
        self.cluster_head_id: Optional[int] = None


        self.speed = random.uniform(8, 15)
        self.direction = random.uniform(0, 2 * np.pi)
        self.position_x = 100 + (source_id % 10) * 80
        self.position_y = 100 + (source_id // 10) * 80


        (self.Xa_train, self.ya_train), (self.Xb_train, self.yb_train) = split_tasks_2way(
            self.X_train, self.y_train
        )
        (self.Xa_test, self.ya_test), (self.Xb_test, self.yb_test) = split_tasks_2way(
            self.X_test, self.y_test
        )


        self.model_a = SGDClassifier(random_state=42, max_iter=100)
        self.model_b = SGDClassifier(random_state=42, max_iter=100)


        self.weights_two: Optional[TwoTaskWeights] = None
        self.latest_accuracy_a: float = 0.0
        self.latest_accuracy_b: float = 0.0
        self.autoencoder: Optional[Autoencoder] = None


        self.use_mtfl = use_mtfl
        self.latent_dim = latent_dim
        self.public_buffer = public_buffer or {}
        self.task_names = task_names or ['task_a', 'task_b']
        self.mtfl_num_classes = mtfl_num_classes

        self.mtfl_model: Optional[MTFLModel] = None
        self.mtfl_optimizer: Optional[optim.Optimizer] = None


        self.historical_accuracy: float = 0.0
        self.participation_count: int = 0
        self.total_rounds: int = 0
        self.encoder_params: Optional[np.ndarray] = None



        self.cnn_device = cnn_device
        self.cifar10_train_loader = cifar10_train_loader
        self.cifar10_test_loader = cifar10_test_loader
        self.cnn_b: Optional[LightCNN] = None
        self.cnn_b_optimizer: Optional[optim.SGD] = None
        self.cnn_b_shapes: Optional[List[Tuple[int, ...]]] = None
        self.cnn_b_sizes: Optional[List[int]] = None

        if self.cnn_device is not None and self.cifar10_train_loader is not None:
            try:
                self.cnn_b = LightCNN().to(self.cnn_device)
                self.cnn_b_optimizer = optim.SGD(self.cnn_b.parameters(), lr=0.01, momentum=0.9)
            except Exception:
                self.cnn_b = None


        self.gtsrb_train_loader = gtsrb_train_loader
        self.gtsrb_test_loader = gtsrb_test_loader
        self.gtsrb_num_classes = int(gtsrb_num_classes)
        self.cnn_c: Optional[LightCNN] = None
        self.cnn_c_optimizer: Optional[optim.SGD] = None
        self.cnn_c_shapes: Optional[List[Tuple[int, ...]]] = None
        self.cnn_c_sizes: Optional[List[int]] = None

        if self.cnn_device is not None and self.gtsrb_train_loader is not None:
            try:
                self.cnn_c = LightCNN(num_classes=self.gtsrb_num_classes).to(self.cnn_device)
                self.cnn_c_optimizer = optim.SGD(self.cnn_c.parameters(), lr=0.01, momentum=0.9)
            except Exception:
                self.cnn_c = None

        self.training_rounds = 0


        if self.use_mtfl:
            self._init_mtfl_model()


        self.producer = KafkaProducerWrapper(client_id=f'vehicle-{source_id}')

        self.subscribed_topics = [
            TOPIC_CH_TO_CM,
            TOPIC_EPC_TO_CH,
            TOPIC_CLUSTER_UPDATE
        ]
        self.consumer = KafkaConsumerWrapper(
            consumer_id=f'vehicle-{source_id}',
            topics=self.subscribed_topics
        )

        self.info_collector = VehicleInfoCollector(consumer_id=f'collector-{source_id}')


        self.received_ae: Optional[np.ndarray] = None


        self._running = False
        self._hello_thread: Optional[threading.Thread] = None
        self._start_hello_publisher()



    def _start_hello_publisher(self) -> None:
        self._running = True
        self._hello_thread = threading.Thread(target=self._publish_hello_loop, daemon=True)
        self._hello_thread.start()

    def _publish_hello_loop(self) -> None:
        while self._running:
            try:
                self._publish_hello_packet()
                time.sleep(HELLO_PACKET_INTERVAL)
            except Exception as e:
                print(f"Vehicle {self.source_id}: Hello packet error: {e}")

    def _publish_hello_packet(self) -> None:
        vib_info = {
            'vehicle_id': self.source_id,
            'position_x': self.position_x,
            'position_y': self.position_y,
            'speed': self.speed,
            'direction': self.direction,
            'role': self.role,
            'timestamp': time.time()
        }
        self.producer.publish_hello_packet(self.source_id, vib_info)



    def update_position(self, time_delta: float = 1.0) -> None:
        center_x, center_y = 500, 500
        distance_to_center = np.sqrt(
            (self.position_x - center_x)**2 + (self.position_y - center_y)**2
        )
        if distance_to_center > 200:
            direction_to_center = np.arctan2(center_y - self.position_y, center_x - self.position_x)
            self.direction = 0.7 * self.direction + 0.3 * direction_to_center
        else:
            self.direction += random.uniform(-0.1, 0.1)
        self.position_x += self.speed * time_delta * np.cos(self.direction)
        self.position_y += self.speed * time_delta * np.sin(self.direction)
        self.position_x = max(0, min(1000, self.position_x))
        self.position_y = max(0, min(1000, self.position_y))
        self.speed = max(8, min(15, self.speed + random.uniform(-0.5, 0.5)))



    def _init_mtfl_model(self) -> None:
        input_dim = self.X_train.shape[1] if self.X_train.size > 0 else 64
        self.mtfl_model = MTFLModel(
            input_dim=input_dim,
            latent_dim=self.latent_dim,
            task_names=self.task_names,
            num_classes=self.mtfl_num_classes
        )
        if self.cnn_device is not None:
            self.mtfl_model = self.mtfl_model.to(self.cnn_device)
        self.mtfl_optimizer = optim.SGD(self.mtfl_model.parameters(), lr=0.01, momentum=0.9)
        self.encoder_params = self.mtfl_model.get_ae_params_flat()

    def train_mtfl_local(self, epochs: int = 1, epsilon: float = 1e-4,
                         mu: float = 0.1, max_batches: int = 20) -> Tuple[float, Dict[str, float]]:

        if not self.use_mtfl or self.mtfl_model is None:
            return 0.0, {}

        self.mtfl_model.train()
        self.training_rounds += 1

        lr = 0.01 * (0.95 ** (self.training_rounds // 10))
        for param_group in self.mtfl_optimizer.param_groups:
            param_group['lr'] = lr

        prev_loss = float('inf')
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            for task_name in self.task_names:
                if task_name == 'task_a' and self.Xa_train.size > 0:
                    X_data, y_data = self.Xa_train, self.ya_train
                elif task_name == 'task_b' and self.Xb_train.size > 0:
                    X_data, y_data = self.Xb_train, self.yb_train
                else:
                    continue
                indices = np.random.permutation(len(X_data))
                for start in range(0, len(indices), 32):
                    if max_batches and batch_count >= max_batches:
                        break
                    batch_indices = indices[start:start+32]
                    x = torch.tensor(X_data[batch_indices], dtype=torch.float32)
                    y = torch.tensor(y_data[batch_indices], dtype=torch.long)
                    if self.cnn_device is not None:
                        x, y = x.to(self.cnn_device), y.to(self.cnn_device)
                    z, x_hat, y_hat = self.mtfl_model(x, task_name)
                    task_loss = nn.CrossEntropyLoss()(y_hat, y)
                    rec_loss = nn.MSELoss()(x_hat, x)
                    total_loss = task_loss + mu * rec_loss
                    self.mtfl_optimizer.zero_grad()
                    total_loss.backward()
                    self.mtfl_optimizer.step()
                    epoch_loss += total_loss.item()
                    batch_count += 1
            if abs(prev_loss - epoch_loss) <= epsilon:
                break
            prev_loss = epoch_loss

        validation_score, per_task_acc = self.evaluate_mtfl_on_public_buffer()
        self.encoder_params = self.mtfl_model.get_ae_params_flat()
        return validation_score, per_task_acc

    def evaluate_mtfl_on_public_buffer(self) -> Tuple[float, Dict[str, float]]:

        if not self.use_mtfl or self.mtfl_model is None:
            return 0.0, {}
        self.mtfl_model.eval()
        per_task_acc = {}
        total_weighted_acc = 0.0
        total_weight = 0.0
        with torch.no_grad():
            for task_name in self.task_names:
                if task_name in self.public_buffer and self.public_buffer[task_name] is not None:
                    loader = self.public_buffer[task_name]
                elif task_name == 'task_a' and self.Xa_test.size > 0:
                    loader = DataLoader(TensorDataset(
                        torch.tensor(self.Xa_test, dtype=torch.float32),
                        torch.tensor(self.ya_test, dtype=torch.long)
                    ), batch_size=64, shuffle=False)
                elif task_name == 'task_b' and self.Xb_test.size > 0:
                    loader = DataLoader(TensorDataset(
                        torch.tensor(self.Xb_test, dtype=torch.float32),
                        torch.tensor(self.yb_test, dtype=torch.long)
                    ), batch_size=64, shuffle=False)
                else:
                    continue
                correct, total = 0, 0
                for x, y in loader:
                    if self.cnn_device is not None:
                        x, y = x.to(self.cnn_device), y.to(self.cnn_device)
                    z = self.mtfl_model.encode(x)
                    y_hat = self.mtfl_model.forward_task(z, task_name)
                    _, predicted = torch.max(y_hat, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                acc = correct / total if total > 0 else 0.0
                per_task_acc[task_name] = acc
                total_weighted_acc += acc
                total_weight += 1
        validation_score = total_weighted_acc / total_weight if total_weight > 0 else 0.0
        return validation_score, per_task_acc

    def get_mtfl_ae_params(self) -> np.ndarray:
        if self.use_mtfl and self.mtfl_model is not None:
            return self.mtfl_model.get_ae_params_flat()
        return self.get_flat_weights()

    def set_mtfl_ae_params(self, flat_params: np.ndarray) -> bool:
        if self.use_mtfl and self.mtfl_model is not None:
            success = self.mtfl_model.set_ae_params_from_flat(_sanitize_vector(flat_params))
            if success:
                self.encoder_params = flat_params
            return success
        return False

    def update_reliability(self, validation_score: float) -> None:
        self.total_rounds += 1
        self.historical_accuracy += validation_score
        self.participation_count += 1

    def get_reliability_score(self, lambda_acc: float = 0.6, lambda_freq: float = 0.4) -> float:
        freq = self.participation_count / self.total_rounds if self.total_rounds > 0 else 0.0
        return lambda_acc * self.historical_accuracy + lambda_freq * freq

    def get_encoder_similarity(self, other: 'Vehicle') -> float:
        if not self.use_mtfl or not other.use_mtfl:
            return self._weights_cosine_similarity(other)
        enc1 = self.encoder_params if self.encoder_params is not None else self.get_mtfl_ae_params()
        enc2 = other.encoder_params if other.encoder_params is not None else other.get_mtfl_ae_params()
        enc1, enc2 = _sanitize_vector(enc1), _sanitize_vector(enc2)
        if np.linalg.norm(enc1) == 0 or np.linalg.norm(enc2) == 0:
            return 0.0
        return np.dot(enc1, enc2) / (np.linalg.norm(enc1) * np.linalg.norm(enc2) + 1e-6)

    def _weights_cosine_similarity(self, other: 'Vehicle') -> float:
        try:
            w1, w2 = _sanitize_vector(self.get_flat_weights()), _sanitize_vector(other.get_flat_weights())
            if np.linalg.norm(w1) == 0 or np.linalg.norm(w2) == 0:
                return 0.5
            return np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2) + 1e-6)
        except Exception:
            return 0.5

    def get_task_similarity(self, other: 'Vehicle') -> float:
        tasks1 = set(self.task_names) if self.use_mtfl else {'task_a', 'task_b'}
        tasks2 = set(other.task_names) if other.use_mtfl else {'task_a', 'task_b'}
        if not tasks1 and not tasks2:
            return 1.0
        if not tasks1 or not tasks2:
            return 0.0
        return len(tasks1 & tasks2) / (len(tasks1 | tasks2) + 1e-6)



    def reset_training_rounds(self) -> None:
        self.training_rounds = 0

    def train_local_two_tasks(self) -> None:
        if self.use_mtfl:
            self.train_mtfl_local()
            return
        self.training_rounds += 1
        lr = 0.01 * (0.95 ** (self.training_rounds // 10))
        self._train_task_a(lr)
        self._train_task_b(lr)
        self._train_task_c(lr)
        self._update_weights_bundle()

    def _train_task_a(self, lr: float) -> None:
        if self.Xa_train.size == 0:
            return
        classes_a = get_classes_for_task(self.ya_train)
        try:
            if hasattr(self.model_a, "coef_") and self.model_a.coef_ is not None:
                existing = set(range(self.model_a.coef_.shape[0]))
                new = set(classes_a)
                if not new.issubset(existing):
                    self.model_a = SGDClassifier(learning_rate='adaptive', eta0=lr, random_state=42)
                    self.model_a.partial_fit(self.Xa_train, self.ya_train, classes=sorted(existing | new))
                else:
                    if hasattr(self.model_a, 'eta0'):
                        self.model_a.eta0 = lr
                    self.model_a.partial_fit(self.Xa_train, self.ya_train, classes=classes_a)
            else:
                self.model_a = SGDClassifier(learning_rate='adaptive', eta0=lr, random_state=42)
                self.model_a.partial_fit(self.Xa_train, self.ya_train, classes=classes_a)
        except Exception as e:
            print(f"Error in MNIST training: {e}")
            self.model_a = SGDClassifier(learning_rate='adaptive', eta0=lr, random_state=42)
            self.model_a.partial_fit(self.Xa_train, self.ya_train, classes=classes_a)

    def _train_task_b(self, lr: float) -> None:
        used_cnn = False
        if self.cnn_b is not None and self.cifar10_train_loader is not None:
            try:
                for pg in self.cnn_b_optimizer.param_groups:
                    pg['lr'] = lr
                self._train_cnn_b_one_epoch()
                used_cnn = True
            except Exception:
                used_cnn = False
        if not used_cnn and self.Xb_train.size > 0:
            classes_b = get_classes_for_task(self.yb_train)
            try:
                if hasattr(self.model_b, "coef_") and self.model_b.coef_ is not None:
                    existing = set(range(self.model_b.coef_.shape[0]))
                    new = set(classes_b)
                    if not new.issubset(existing):
                        self.model_b = SGDClassifier(learning_rate='adaptive', eta0=lr, random_state=42)
                        self.model_b.partial_fit(self.Xb_train, self.yb_train, classes=sorted(existing | new))
                    else:
                        if hasattr(self.model_b, 'eta0'):
                            self.model_b.eta0 = lr
                        self.model_b.partial_fit(self.Xb_train, self.yb_train, classes=classes_b)
                else:
                    self.model_b = SGDClassifier(learning_rate='adaptive', eta0=lr, random_state=42)
                    self.model_b.partial_fit(self.Xb_train, self.yb_train, classes=classes_b)
            except Exception:
                pass

    def _train_task_c(self, lr: float) -> None:
        if self.cnn_c is not None and self.gtsrb_train_loader is not None:
            try:
                for pg in self.cnn_c_optimizer.param_groups:
                    pg['lr'] = lr
                self._train_cnn_c_one_epoch()
            except Exception:
                pass

    def _train_cnn_b_one_epoch(self, max_batches: int = 20) -> None:
        if self.cnn_b is None or self.cifar10_train_loader is None:
            return
        self.cnn_b.train()
        criterion = nn.CrossEntropyLoss()
        batches = 0
        for images, labels in self.cifar10_train_loader:
            images, labels = images.to(self.cnn_device), labels.to(self.cnn_device)
            self.cnn_b_optimizer.zero_grad()
            loss = criterion(self.cnn_b(images), labels)
            loss.backward()
            self.cnn_b_optimizer.step()
            batches += 1
            if batches >= max_batches:
                break

    def _train_cnn_c_one_epoch(self, max_batches: int = 20) -> None:
        if self.cnn_c is None or self.gtsrb_train_loader is None:
            return
        self.cnn_c.train()
        criterion = nn.CrossEntropyLoss()
        batches = 0
        for images, labels in self.gtsrb_train_loader:
            images, labels = images.to(self.cnn_device), labels.to(self.cnn_device)
            self.cnn_c_optimizer.zero_grad()
            loss = criterion(self.cnn_c(images), labels)
            loss.backward()
            self.cnn_c_optimizer.step()
            batches += 1
            if batches >= max_batches:
                break

    def _update_weights_bundle(self) -> None:
        coeff_a = getattr(self.model_a, "coef_", np.zeros((10, self.X_train.shape[1])))
        inter_a = getattr(self.model_a, "intercept_", np.zeros(10))
        coeff_b = getattr(self.model_b, "coef_", np.zeros((10, self.X_train.shape[1])))
        inter_b = getattr(self.model_b, "intercept_", np.zeros(10))
        self.weights_two = TwoTaskWeights(
            coeff_a=np.asarray(coeff_a), inter_a=np.asarray(inter_a),
            coeff_b=np.asarray(coeff_b), inter_b=np.asarray(inter_b)
        )

    def evaluate_two_tasks(self) -> Tuple[float, float]:
        if self.use_mtfl:
            score, per_task = self.evaluate_mtfl_on_public_buffer()
            acc_a, acc_b = per_task.get('task_a', 0.0), per_task.get('task_b', 0.0)
            self.latest_accuracy_a, self.latest_accuracy_b = acc_a, acc_b
            return acc_a, acc_b
        acc_a, acc_b = 0.0, 0.0
        if self.Xa_test.size > 0 and hasattr(self.model_a, "coef_"):
            try:
                acc_a = float(self.model_a.score(self.Xa_test, self.ya_test))
            except Exception:
                pass
        if self.cnn_b is not None and self.cifar10_test_loader is not None:
            try:
                acc_b = self._evaluate_cnn_b()
            except Exception:
                pass
        elif self.Xb_test.size > 0 and hasattr(self.model_b, "coef_"):
            try:
                acc_b = float(self.model_b.score(self.Xb_test, self.yb_test))
            except Exception:
                pass
        self.latest_accuracy_a, self.latest_accuracy_b = acc_a, acc_b
        return acc_a, acc_b

    def _evaluate_cnn_b(self) -> float:
        if self.cnn_b is None or self.cifar10_test_loader is None:
            return 0.0
        self.cnn_b.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.cifar10_test_loader:
                images, labels = images.to(self.cnn_device), labels.to(self.cnn_device)
                outputs = self.cnn_b(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total if total > 0 else 0.0

    def evaluate_gtsrb(self) -> float:
        if self.cnn_c is None or self.gtsrb_test_loader is None:
            return 0.0
        self.cnn_c.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.gtsrb_test_loader:
                images, labels = images.to(self.cnn_device), labels.to(self.cnn_device)
                outputs = self.cnn_c(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total if total > 0 else 0.0

    def get_flat_weights(self) -> np.ndarray:
        if self.weights_two is None:
            self.train_local_two_tasks()
        parts = [self.weights_two.coeff_a.ravel(), self.weights_two.inter_a.ravel(),
                 self.weights_two.coeff_b.ravel(), self.weights_two.inter_b.ravel()]
        try:
            if self.cnn_b is not None:
                fb, sb, shb = self._state_to_flat(self.cnn_b)
                self.cnn_b_sizes, self.cnn_b_shapes = sb, shb
                parts.append(fb)
            if self.cnn_c is not None:
                fc, sc, shc = self._state_to_flat(self.cnn_c)
                self.cnn_c_sizes, self.cnn_c_shapes = sc, shc
                parts.append(fc)
        except Exception:
            pass
        return np.concatenate(parts)

    def set_weights_from_flat(self, flat: np.ndarray) -> None:
        if self.weights_two is None:
            self.train_local_two_tasks()
        flat = _sanitize_vector(flat)
        tpl = self.weights_two
        sizes = [tpl.coeff_a.size, tpl.inter_a.size, tpl.coeff_b.size, tpl.inter_b.size]
        total = sum(sizes)
        f = np.asarray(flat, dtype=float).ravel()
        if f.size < total:
            f = np.pad(f, (0, total - f.size))
        elif f.size > total:
            f = f[:total]
        try:
            p0 = 0
            coeff_a = f[p0:p0+sizes[0]].reshape(tpl.coeff_a.shape)
            p0 += sizes[0]
            inter_a = f[p0:p0+sizes[1]].reshape(tpl.inter_a.shape)
            p0 += sizes[1]
            coeff_b = f[p0:p0+sizes[2]].reshape(tpl.coeff_b.shape)
            p0 += sizes[2]
            inter_b = f[p0:p0+sizes[3]].reshape(tpl.inter_b.shape)
            self.set_two_task_weights(TwoTaskWeights(coeff_a, inter_a, coeff_b, inter_b))
            self._load_cnn_weights_from_flat(f[total:])
        except Exception:
            self.train_local_two_tasks()

    def set_two_task_weights(self, weights: TwoTaskWeights) -> None:
        if weights is None:
            return
        self.model_a.coef_ = np.asarray(weights.coeff_a)
        self.model_a.intercept_ = np.asarray(weights.inter_a)
        self.model_b.coef_ = np.asarray(weights.coeff_b)
        self.model_b.intercept_ = np.asarray(weights.inter_b)
        self.weights_two = weights

    def _load_cnn_weights_from_flat(self, remaining: np.ndarray) -> None:
        try:
            if remaining.size > 0 and self.cnn_b is not None:
                if self.cnn_b_sizes is None:
                    fb, sb, shb = self._state_to_flat(self.cnn_b)
                    self.cnn_b_sizes, self.cnn_b_shapes = sb, shb
                if self.cnn_b_sizes is not None:
                    numel_b = int(np.sum(self.cnn_b_sizes))
                    self._flat_to_state(self.cnn_b, remaining[:numel_b], self.cnn_b_sizes, self.cnn_b_shapes)
                    remaining = remaining[numel_b:]
            if remaining.size > 0 and self.cnn_c is not None:
                if self.cnn_c_sizes is None:
                    fc, sc, shc = self._state_to_flat(self.cnn_c)
                    self.cnn_c_sizes, self.cnn_c_shapes = sc, shc
                if self.cnn_c_sizes is not None:
                    numel_c = int(np.sum(self.cnn_c_sizes))
                    self._flat_to_state(self.cnn_c, remaining[:numel_c], self.cnn_c_sizes, self.cnn_c_shapes)
        except Exception:
            pass

    def _state_to_flat(self, model: nn.Module):
        sd = model.state_dict()
        flats, sizes, shapes = [], [], []
        for key in sd:
            t = sd[key].detach().cpu().float().contiguous()
            arr = t.view(-1).numpy()
            flats.append(arr)
            sizes.append(arr.size)
            shapes.append(tuple(t.shape))
        return (np.concatenate(flats) if flats else np.array([], dtype=np.float32)), sizes, shapes

    def _flat_to_state(self, model: nn.Module, flat: np.ndarray, sizes: List[int], shapes: List[Tuple[int, ...]]) -> None:
        try:
            sd = model.state_dict()
            keys = list(sd.keys())
            idx = 0
            tensor_list = []
            for s, shape in zip(sizes, shapes):
                tensor_list.append(torch.from_numpy(flat[idx:idx+s]).view(*shape))
                idx += s
            new_sd = {key: tensor.type_as(sd[key]) for key, tensor in zip(keys, tensor_list)}
            model.load_state_dict(new_sd, strict=False)
        except Exception:
            pass

    def _ensure_autoencoder(self) -> None:
        if self.autoencoder is None:
            if self.weights_two is None:
                self.train_local_two_tasks()
            flat_len = self.get_flat_weights().size
            self.autoencoder = Autoencoder(input_dim=flat_len, latent_dim=min(256, flat_len // 2), learning_rate=1e-3)

    def message_to_ch_flat(self) -> np.ndarray:
        if self.use_mtfl:
            ae_params = self.get_mtfl_ae_params()
            if self.is_attacker:
                ae_params = ae_params + np.random.normal(0, 0.01, size=ae_params.shape)
            return _sanitize_vector(ae_params)
        self._ensure_autoencoder()
        flat = self.get_flat_weights()
        if self.autoencoder.input_dim != flat.size:
            self.autoencoder = Autoencoder(input_dim=flat.size, latent_dim=min(256, flat.size // 2), learning_rate=1e-3)
        if np.any(np.isnan(flat)) or np.any(np.isinf(flat)):
            flat = _sanitize_vector(flat)
        self.autoencoder.partial_fit(flat, epochs=5)
        z = self.autoencoder.encode_flat(_sanitize_vector(flat))
        flat_rec = self.autoencoder.decode_flat(z)
        flat_rec = _sanitize_vector(flat_rec)
        if not self._validate_weight_reconstruction(flat, flat_rec):
            flat_rec = flat.copy()
        if self.is_attacker:
            flat_rec += np.random.normal(0, 0.01, size=flat_rec.shape)
        return flat_rec

    def _validate_weight_reconstruction(self, original: np.ndarray, reconstructed: np.ndarray) -> bool:
        try:
            if np.any(np.isnan(reconstructed)) or np.any(np.isinf(reconstructed)):
                return False
            if original.size != reconstructed.size:
                return False
            if np.all(reconstructed == 0):
                return False
            if np.max(np.abs(reconstructed)) > 1e6:
                return False
            correlation = np.corrcoef(original.ravel(), reconstructed.ravel())[0, 1]
            return correlation >= 0.3
        except Exception:
            return False



    def send_cm_to_ch(self, ch_id: int, ae_params: np.ndarray, score: float) -> None:

        if self.is_attacker:
            ae_params = ae_params + np.random.normal(0, 0.01, size=ae_params.shape)
        self.producer.publish_cm_to_ch(ch_id, self.source_id, _sanitize_vector(ae_params), score)
        self.producer.publish_reliability(self.source_id, self.get_reliability_score(), self.role)

    def receive_from_ch(self, timeout: float = 2.0) -> Optional[np.ndarray]:

        start_time = time.time()
        while time.time() - start_time < timeout:
            msg = self.consumer.consume_once()
            if msg is None:
                continue
            try:
                value = msg['value']
                if value.get('type') == 'ch_to_cm' and value.get('cm_id') == self.source_id:
                    ae_list = value.get('aggregated_ae', [])
                    if ae_list:
                        return np.array(ae_list)
                elif value.get('type') == 'epc_global':
                    ae_list = value.get('global_ae', [])
                    if ae_list:
                        return np.array(ae_list)
            except Exception:
                continue
            time.sleep(0.1)
        return None

    def receive_cluster_update(self, timeout: float = 2.0) -> Optional[int]:

        start_time = time.time()
        while time.time() - start_time < timeout:
            msg = self.consumer.consume_once()
            if msg is None:
                continue
            try:
                value = msg['value']
                if value.get('type') == 'cluster_update':
                    members = value.get('members', [])
                    if self.source_id in members:
                        return value.get('ch_id')
            except Exception:
                continue
            time.sleep(0.1)
        return None

    def get_neighbors_from_kafka(self) -> List[int]:

        self.info_collector.update_vehicle_info()
        neighbors = []
        my_info = self.info_collector.get_vehicle_info(self.source_id)
        if my_info is None:
            return neighbors
        my_x, my_y = my_info.get('position_x', 0), my_info.get('position_y', 0)
        for vid, info in self.info_collector.get_all_vehicles().items():
            if vid == self.source_id:
                continue
            dist = ((my_x - info.get('position_x', 0))**2 + (my_y - info.get('position_y', 0))**2)**0.5
            if dist <= self.transmission_range:
                neighbors.append(vid)
        return neighbors

    def get_all_vehicle_info(self) -> Dict[int, Dict[str, Any]]:
        self.info_collector.update_vehicle_info()
        return self.info_collector.get_all_vehicles()



    def shutdown(self) -> None:
        self._running = False
        if self._hello_thread is not None:
            self._hello_thread.join(timeout=2.0)
        self.consumer.stop()
        self.info_collector.stop()
        print(f"Vehicle {self.source_id} shutdown complete")
