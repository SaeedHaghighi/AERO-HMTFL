



from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.linear_model import SGDClassifier
import time

from mftl.utils.helpers import _sanitize_vector
from mftl.models.cnn_models import LightCNN
from mftl.models.autoencoder import Autoencoder
from mftl.kafka import (
    KafkaProducerWrapper,
    KafkaConsumerWrapper,
    EPCCollector,
    TOPIC_CH_TO_EPC,
    TOPIC_EPC_TO_CH
)


class EPC:





    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        cnn_device: Optional[torch.device] = None,
        cifar10_train_loader: Optional[DataLoader] = None,
        cifar10_test_loader: Optional[DataLoader] = None,
        gtsrb_train_loader: Optional[DataLoader] = None,
        gtsrb_test_loader: Optional[DataLoader] = None,
        gtsrb_num_classes: int = 43
    ) -> None:



        self.mnist_classifier = SGDClassifier(random_state=42, max_iter=100)
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.X_test = np.asarray(X_test)
        self.y_test = np.asarray(y_test)


        self.cnn_device = cnn_device
        self.cifar10_train_loader = cifar10_train_loader
        self.cifar10_test_loader = cifar10_test_loader
        self.cifar10_model: Optional[LightCNN] = None
        self.cifar10_optimizer: Optional[optim.SGD] = None

        if self.cnn_device is not None and self.cifar10_train_loader is not None:
            try:
                self.cifar10_model = LightCNN().to(self.cnn_device)
                self.cifar10_optimizer = optim.SGD(self.cifar10_model.parameters(), lr=0.01, momentum=0.9)
            except Exception:
                self.cifar10_model = None


        self.gtsrb_train_loader = gtsrb_train_loader
        self.gtsrb_test_loader = gtsrb_test_loader
        self.gtsrb_model: Optional[LightCNN] = None
        self.gtsrb_optimizer: Optional[optim.SGD] = None
        self.gtsrb_num_classes = int(gtsrb_num_classes)

        if self.cnn_device is not None and self.gtsrb_train_loader is not None:
            try:
                self.gtsrb_model = LightCNN(num_classes=self.gtsrb_num_classes).to(self.cnn_device)
                self.gtsrb_optimizer = optim.SGD(self.gtsrb_model.parameters(), lr=0.01, momentum=0.9)
            except Exception:
                self.gtsrb_model = None


        self.mnist_accuracy_history: List[float] = []
        self.cifar10_accuracy_history: List[float] = []
        self.gtsrb_accuracy_history: List[float] = []


        self.mnist_weight_size: Optional[int] = None
        self.cifar_weight_size: Optional[int] = None
        self.total_weight_size: Optional[int] = None


        self.autoencoder: Optional[Autoencoder] = None


        self.producer = KafkaProducerWrapper(client_id='epc-server')


        self.consumer = KafkaConsumerWrapper(
            consumer_id='epc-server',
            topics=[TOPIC_CH_TO_EPC]
        )


        self.ch_collector = EPCCollector(consumer_id='epc-collector')


        self.ch_updates: Dict[int, Dict[str, any]] = {}
        self.last_aggregated: Optional[np.ndarray] = None

    def _ensure_autoencoder(self) -> None:
        if self.autoencoder is None:
            self.autoencoder = Autoencoder(input_dim=1300, latent_dim=min(256, 1300 // 2))



    def collect_ch_updates(self, timeout: float = 5.0) -> Dict[int, Dict[str, any]]:

        self.ch_updates = self.ch_collector.collect_ch_updates(timeout)
        return self.ch_updates

    def broadcast_to_chs(self, ch_ids: List[int], global_ae: np.ndarray) -> None:

        for ch_id in ch_ids:
            self.producer.publish_epc_to_ch(ch_id, global_ae)



    def aggregate_from_ch(self, ch_weights: List[np.ndarray]) -> np.ndarray:

        if not ch_weights:
            return None

        print(f"EPC: Processing {len(ch_weights)} CH weight vectors")

        aggregated_weights = self._aggregate_vectors(ch_weights)
        print(f"EPC: Aggregated weights size: {aggregated_weights.size}")

        self._set_and_train_dual_models(aggregated_weights)

        mnist_acc = self._evaluate_mnist()
        cifar10_acc = self._evaluate_cifar10()
        gtsrb_acc = self._evaluate_gtsrb()

        self.mnist_accuracy_history.append(mnist_acc)
        self.cifar10_accuracy_history.append(cifar10_acc)
        self.gtsrb_accuracy_history.append(gtsrb_acc)

        print(f"EPC: MNIST: {mnist_acc:.4f}, CIFAR: {cifar10_acc:.4f}, GTSRB: {gtsrb_acc:.4f}")

        self.last_aggregated = aggregated_weights
        return aggregated_weights

    def _aggregate_vectors(self, vectors: List[np.ndarray]) -> np.ndarray:

        if not vectors:
            raise ValueError("No vectors to aggregate")
        if len(vectors) == 1:
            return _sanitize_vector(vectors[0])

        self._ensure_autoencoder()
        flat_vectors = [_sanitize_vector(v) for v in vectors]
        flat_vectors = [v for v in flat_vectors if np.linalg.norm(v) > 0]
        if not flat_vectors:
            return _sanitize_vector(vectors[0])

        if self.autoencoder.input_dim != flat_vectors[0].size:
            self.autoencoder = Autoencoder(
                input_dim=flat_vectors[0].size,
                latent_dim=min(256, flat_vectors[0].size // 2)
            )

        processed_vectors = []
        for vec in flat_vectors:
            try:
                encoded = self.autoencoder.encode_flat(vec)
                decoded = self.autoencoder.decode_flat(encoded)
                processed_vectors.append(decoded)
            except Exception:
                processed_vectors.append(vec)

        if len(processed_vectors) > 1:
            final_aggregated = 0.7 * np.mean(processed_vectors, axis=0) + 0.3 * np.median(processed_vectors, axis=0)
        else:
            final_aggregated = processed_vectors[0]

        return _sanitize_vector(final_aggregated)

    def _set_and_train_dual_models(self, aggregated_weights: np.ndarray) -> None:

        mnist_part, cifar_part = self._extract_task_weights(aggregated_weights)
        if mnist_part is not None:
            self._set_and_train_mnist(mnist_part)
        if cifar_part is not None:
            self._set_and_train_cifar(cifar_part)
        self._set_and_train_gtsrb(aggregated_weights)

    def _extract_task_weights(self, flat_weight: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:

        try:
            self._ensure_autoencoder()
            if self.autoencoder.input_dim != flat_weight.size:
                self.autoencoder = Autoencoder(
                    input_dim=flat_weight.size,
                    latent_dim=min(256, flat_weight.size // 2)
                )
            try:
                encoded = self.autoencoder.encode_flat(flat_weight)
                processed_flat = self.autoencoder.decode_flat(encoded)
            except Exception:
                processed_flat = flat_weight

            if self.mnist_weight_size is not None and self.total_weight_size is not None:
                if processed_flat.size == self.total_weight_size:
                    mnist_part = processed_flat[:self.mnist_weight_size]
                    cifar_part = processed_flat[self.mnist_weight_size:] if self.mnist_weight_size < processed_flat.size else None
                    return mnist_part, cifar_part

            if hasattr(self.mnist_classifier, 'coef_') and self.mnist_classifier.coef_ is not None:
                coef_shape = self.mnist_classifier.coef_.shape
                inter_shape = self.mnist_classifier.intercept_.shape
                mnist_size = coef_shape[0] * coef_shape[1] + inter_shape[0]
                if flat_weight.size >= mnist_size:
                    mnist_part = flat_weight[:mnist_size]
                    cifar_part = flat_weight[mnist_size:] if flat_weight.size > mnist_size else None
                    if self.mnist_weight_size is None:
                        self.mnist_weight_size = mnist_size
                        self.cifar_weight_size = flat_weight.size - mnist_size if cifar_part is not None else 0
                        self.total_weight_size = flat_weight.size
                    return mnist_part, cifar_part
                else:
                    if self.mnist_weight_size is None:
                        self.mnist_weight_size = flat_weight.size
                        self.cifar_weight_size = 0
                        self.total_weight_size = flat_weight.size
                    return flat_weight, None

            mid_point = flat_weight.size // 2
            if self.mnist_weight_size is None:
                self.mnist_weight_size = mid_point
                self.cifar_weight_size = flat_weight.size - mid_point
                self.total_weight_size = flat_weight.size
            return flat_weight[:mid_point], flat_weight[mid_point:]
        except Exception:
            mid_point = flat_weight.size // 2
            return flat_weight[:mid_point], flat_weight[mid_point:]

    def _set_and_train_mnist(self, aggregated_weights: Optional[np.ndarray]) -> None:
        if aggregated_weights is None:
            return
        try:
            if hasattr(self.mnist_classifier, 'coef_') and self.mnist_classifier.coef_ is not None:
                coef_shape = self.mnist_classifier.coef_.shape
                inter_shape = self.mnist_classifier.intercept_.shape
                total = coef_shape[0] * coef_shape[1] + inter_shape[0]
                if aggregated_weights.size >= total:
                    self.mnist_classifier.coef_ = aggregated_weights[:coef_shape[0] * coef_shape[1]].reshape(coef_shape)
                    self.mnist_classifier.intercept_ = aggregated_weights[coef_shape[0] * coef_shape[1]:total].reshape(inter_shape)
                    classes = np.unique(self.y_train)
                    if len(classes) > 0:
                        self.mnist_classifier.partial_fit(self.X_train, self.y_train, classes=classes)
                else:
                    self.mnist_classifier.fit(self.X_train, self.y_train)
            else:
                self.mnist_classifier.fit(self.X_train, self.y_train)
        except Exception:
            pass

    def _set_and_train_cifar(self, aggregated_weights: Optional[np.ndarray]) -> None:
        if aggregated_weights is None or self.cifar10_model is None:
            return
        try:
            self.cifar10_model.train()
            criterion = nn.CrossEntropyLoss()
            for epoch in range(3):
                for images, labels in self.cifar10_train_loader:
                    if images.size(0) == 0:
                        continue
                    images, labels = images.to(self.cnn_device), labels.to(self.cnn_device)
                    self.cifar10_optimizer.zero_grad()
                    loss = criterion(self.cifar10_model(images), labels)
                    loss.backward()
                    self.cifar10_optimizer.step()
        except Exception:
            pass

    def _set_and_train_gtsrb(self, aggregated_weights: Optional[np.ndarray]) -> None:
        if aggregated_weights is None or self.gtsrb_model is None:
            return
        try:
            self.gtsrb_model.train()
            criterion = nn.CrossEntropyLoss()
            for epoch in range(3):
                for images, labels in self.gtsrb_train_loader:
                    if images.size(0) == 0:
                        continue
                    images, labels = images.to(self.cnn_device), labels.to(self.cnn_device)
                    self.gtsrb_optimizer.zero_grad()
                    loss = criterion(self.gtsrb_model(images), labels)
                    loss.backward()
                    self.gtsrb_optimizer.step()
        except Exception:
            pass

    def _evaluate_mnist(self) -> float:
        try:
            if hasattr(self.mnist_classifier, 'coef_') and self.mnist_classifier.coef_ is not None:
                return float(self.mnist_classifier.score(self.X_test, self.y_test))
            return 0.0
        except Exception:
            return 0.0

    def _evaluate_cifar10(self) -> float:
        if self.cifar10_model is None or self.cifar10_test_loader is None:
            return 0.0
        self.cifar10_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.cifar10_test_loader:
                if images.size(0) == 0:
                    continue
                images, labels = images.to(self.cnn_device), labels.to(self.cnn_device)
                outputs = self.cifar10_model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total if total > 0 else 0.0

    def _evaluate_gtsrb(self) -> float:
        if self.gtsrb_model is None or self.gtsrb_test_loader is None:
            return 0.0
        self.gtsrb_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.gtsrb_test_loader:
                if images.size(0) == 0:
                    continue
                images, labels = images.to(self.cnn_device), labels.to(self.cnn_device)
                outputs = self.gtsrb_model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total if total > 0 else 0.0

    def get_latest_accuracies(self) -> Tuple[float, float, float]:
        mnist_acc = self.mnist_accuracy_history[-1] if self.mnist_accuracy_history else 0.0
        cifar10_acc = self.cifar10_accuracy_history[-1] if self.cifar10_accuracy_history else 0.0
        gtsrb_acc = self.gtsrb_accuracy_history[-1] if self.gtsrb_accuracy_history else 0.0
        return mnist_acc, cifar10_acc, gtsrb_acc

    def message_to_ch_flat(self, sample_weights: np.ndarray) -> np.ndarray:
        self._ensure_autoencoder()
        if self.autoencoder.input_dim != sample_weights.size:
            self.autoencoder = Autoencoder(
                input_dim=sample_weights.size,
                latent_dim=min(256, sample_weights.size // 2)
            )
        flat = sample_weights
        self.autoencoder.partial_fit(flat, epochs=1)
        z = self.autoencoder.encode_flat(_sanitize_vector(flat))
        flat_rec = self.autoencoder.decode_flat(z)
        flat_rec = _sanitize_vector(flat_rec)
        return flat_rec



    def shutdown(self) -> None:
        self.consumer.stop()
        self.ch_collector.stop()
        print("EPC shutdown complete")
