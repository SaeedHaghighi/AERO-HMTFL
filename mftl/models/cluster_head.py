



from typing import List, Optional, Dict
import numpy as np
import threading
import time

from mftl.models.vehicle import Vehicle
from mftl.models.autoencoder import Autoencoder
from mftl.utils.helpers import _sanitize_vector
from mftl.kafka import (
    KafkaProducerWrapper,
    KafkaConsumerWrapper,
    CHMessageCollector,
    TOPIC_CM_TO_CH,
    TOPIC_CH_TO_EPC,
    TOPIC_EPC_TO_CH,
    TOPIC_CH_TO_CM,
    TOPIC_CLUSTER_UPDATE,
    TOPIC_RELIABILITY
)


class ClusterHead:





    def __init__(
        self,
        vehicle: Vehicle,
        lambda_acc: float = 0.6,
        lambda_freq: float = 0.4
    ):

        self.vehicle = vehicle
        self.ch_id = vehicle.source_id
        self.lambda_acc = lambda_acc
        self.lambda_freq = lambda_freq

        self.member_ids: List[int] = []
        self.member_vehicles: Dict[int, Vehicle] = {}
        self.member_reliability: Dict[int, float] = {}

        self.aggregate_weights = None
        self.autoencoder: Optional[Autoencoder] = None
        self.incoming_transfers: List[np.ndarray] = []


        self.producer = KafkaProducerWrapper(client_id=f'ch-{self.ch_id}')


        self.subscribed_topics = [TOPIC_CM_TO_CH, TOPIC_EPC_TO_CH, TOPIC_RELIABILITY]
        self.consumer = KafkaConsumerWrapper(
            consumer_id=f'ch-{self.ch_id}',
            topics=self.subscribed_topics
        )


        self.cm_collector = CHMessageCollector(
            ch_id=self.ch_id,
            consumer_id=f'ch-{self.ch_id}-collector'
        )


        self.cm_updates: Dict[int, Dict[str, any]] = {}
        self.reliability_updates: Dict[int, float] = {}



    def collect_cm_updates(self, timeout: float = 5.0) -> Dict[int, Dict[str, any]]:

        self.cm_updates = self.cm_collector.collect_cm_updates(timeout)
        return self.cm_updates

    def broadcast_to_members(self, ae_params: np.ndarray) -> None:

        sanitized = _sanitize_vector(ae_params)


        self.vehicle.set_mtfl_ae_params(sanitized)


        for member_id in self.member_ids:
            self.producer.publish_ch_to_cm(self.ch_id, member_id, sanitized)

    def send_to_epc(self, ae_params: np.ndarray) -> None:

        self.producer.publish_ch_to_epc(self.ch_id, ae_params)

    def receive_from_epc(self, timeout: float = 5.0) -> Optional[np.ndarray]:

        start_time = time.time()
        while time.time() - start_time < timeout:
            msg = self.consumer.consume_once()
            if msg is None:
                continue
            try:
                value = msg['value']
                if value.get('type') == 'epc_global' and value.get('ch_id') == self.ch_id:
                    ae_list = value.get('global_ae', [])
                    if ae_list:
                        return np.array(ae_list)
            except Exception:
                continue
            time.sleep(0.1)
        return None

    def publish_cluster_update(self) -> None:

        self.producer.publish_cluster_update(self.ch_id, self.member_ids)



    def aggregate_members_mtfl(
        self,
        member_updates: Dict[int, np.ndarray],
        member_scores: Dict[int, float]
    ) -> np.ndarray:

        if not member_updates:
            return self.vehicle.get_mtfl_ae_params()


        all_updates = {self.vehicle.source_id: self.vehicle.get_mtfl_ae_params()}
        all_scores = {self.vehicle.source_id: self.vehicle.historical_accuracy}

        for member_id, update in member_updates.items():
            all_updates[member_id] = update
            if member_id in self.member_vehicles:
                vehicle = self.member_vehicles[member_id]
                rel = vehicle.get_reliability_score(self.lambda_acc, self.lambda_freq)
                all_scores[member_id] = rel
            elif member_id in self.member_reliability:
                all_scores[member_id] = self.member_reliability[member_id]
            else:
                all_scores[member_id] = member_scores.get(member_id, 0.0)


        total_reliability = sum(all_scores.values())
        if total_reliability == 0:
            weights = {k: 1.0 / len(all_updates) for k in all_updates}
        else:
            weights = {k: v / total_reliability for k, v in all_scores.items()}


        aggregated = np.zeros_like(next(iter(all_updates.values())))
        for member_id, update in all_updates.items():
            aggregated += weights.get(member_id, 0.0) * _sanitize_vector(update)

        self.aggregate_weights = _sanitize_vector(aggregated)
        return self.aggregate_weights

    def add_member(self, vehicle: Vehicle) -> None:

        self.member_ids.append(vehicle.source_id)
        self.member_vehicles[vehicle.source_id] = vehicle
        vehicle.cluster_head_id = self.ch_id

    def remove_member(self, vehicle_id: int) -> None:

        if vehicle_id in self.member_ids:
            self.member_ids.remove(vehicle_id)
            if vehicle_id in self.member_vehicles:
                del self.member_vehicles[vehicle_id]



    def _ensure_autoencoder(self) -> None:
        if self.autoencoder is None:
            sample_weights = self.vehicle.get_flat_weights()
            input_dim = sample_weights.size
            self.autoencoder = Autoencoder(
                input_dim=input_dim,
                latent_dim=min(256, input_dim // 2)
            )

    def aggregate_encoded(self, encoded_updates: List[np.ndarray]) -> np.ndarray:
        if not encoded_updates:
            return None
        return self._aggregate_vectors([_sanitize_vector(v) for v in encoded_updates])

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

    def message_to_epc_flat(self) -> np.ndarray:
        self._ensure_autoencoder()
        flat = self.vehicle.get_flat_weights()
        if self.autoencoder.input_dim != flat.size:
            self.autoencoder = Autoencoder(input_dim=flat.size, latent_dim=min(256, flat.size // 2))
        self.autoencoder.partial_fit(flat, epochs=1)
        z = self.autoencoder.encode_flat(_sanitize_vector(flat))
        flat_rec = self.autoencoder.decode_flat(z)
        flat_rec = _sanitize_vector(flat_rec)
        if self.vehicle.is_attacker:
            flat_rec += np.random.normal(0, 0.01, size=flat_rec.shape)
        return flat_rec

    def message_to_cm_flat(self) -> np.ndarray:
        self._ensure_autoencoder()
        flat = self.vehicle.get_flat_weights()
        if self.autoencoder.input_dim != flat.size:
            self.autoencoder = Autoencoder(input_dim=flat.size, latent_dim=min(256, flat.size // 2))
        self.autoencoder.partial_fit(flat, epochs=1)
        z = self.autoencoder.encode_flat(_sanitize_vector(flat))
        flat_rec = self.autoencoder.decode_flat(z)
        flat_rec = _sanitize_vector(flat_rec)
        if self.vehicle.is_attacker:
            flat_rec += np.random.normal(0, 0.01, size=flat_rec.shape)
        return flat_rec



    def shutdown(self) -> None:
        self.consumer.stop()
        self.cm_collector.stop()
        print(f"ClusterHead {self.ch_id} shutdown complete")
