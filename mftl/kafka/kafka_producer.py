



import json
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from confluent_kafka import Producer

from .kafka_config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_PRODUCER_ACKS,
    KAFKA_PRODUCER_RETRIES
)

logger = logging.getLogger(__name__)


class KafkaProducerWrapper:


    def __init__(self, client_id: Optional[str] = None):
        config = {
            'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
            'acks': KAFKA_PRODUCER_ACKS,
            'retries': KAFKA_PRODUCER_RETRIES,
            'enable.idempotence': True,
            'max.in.flight.requests.per.connection': 5,
            'compression.type': 'snappy'
        }
        if client_id:
            config['client.id'] = client_id
        self.producer = Producer(config)
        self.client_id = client_id

    def publish(self, topic: str, key: str, value: Dict[str, Any]) -> None:

        try:
            value_json = json.dumps(value).encode('utf-8')
            self.producer.produce(
                topic=topic,
                key=key.encode('utf-8'),
                value=value_json,
                callback=self._delivery_report
            )
            self.producer.flush()
        except Exception as e:
            logger.error(f"Failed to publish message to {topic}: {e}")
            raise

    def publish_hello_packet(self, vehicle_id: int, vehicle_info: Dict[str, Any]) -> None:

        from .kafka_topics import TOPIC_HELLO
        message = {
            'type': 'hello_packet',
            'source_id': vehicle_id,
            'timestamp': vehicle_info.get('timestamp'),
            'data': vehicle_info
        }
        self.publish(TOPIC_HELLO, f'vehicle-{vehicle_id}', message)

    def publish_cm_to_ch(self, ch_id: int, cm_id: int, ae_params: np.ndarray, score: float) -> None:







        from .kafka_topics import TOPIC_CM_TO_CH

        ae_list = ae_params.tolist() if ae_params.size > 0 else []
        message = {
            'type': 'cm_ae_params',
            'source_id': cm_id,
            'ch_id': ch_id,
            'ae_params': ae_list,
            'validation_score': score,
            'timestamp': __import__('time').time()
        }
        self.publish(TOPIC_CM_TO_CH, f'cm-{cm_id}-to-ch-{ch_id}', message)

    def publish_ch_to_epc(self, ch_id: int, aggregated_ae: np.ndarray) -> None:






        from .kafka_topics import TOPIC_CH_TO_EPC

        ae_list = aggregated_ae.tolist() if aggregated_ae.size > 0 else []
        message = {
            'type': 'ch_aggregated',
            'source_id': ch_id,
            'aggregated_ae': ae_list,
            'timestamp': __import__('time').time()
        }
        self.publish(TOPIC_CH_TO_EPC, f'ch-{ch_id}-to-epc', message)

    def publish_epc_to_ch(self, ch_id: int, global_ae: np.ndarray) -> None:






        from .kafka_topics import TOPIC_EPC_TO_CH

        ae_list = global_ae.tolist() if global_ae.size > 0 else []
        message = {
            'type': 'epc_global',
            'source_id': 'EPC',
            'ch_id': ch_id,
            'global_ae': ae_list,
            'timestamp': __import__('time').time()
        }
        self.publish(TOPIC_EPC_TO_CH, f'epc-to-ch-{ch_id}', message)

    def publish_ch_to_cm(self, ch_id: int, cm_id: int, aggregated_ae: np.ndarray) -> None:







        from .kafka_topics import TOPIC_CH_TO_CM

        ae_list = aggregated_ae.tolist() if aggregated_ae.size > 0 else []
        message = {
            'type': 'ch_to_cm',
            'source_id': ch_id,
            'cm_id': cm_id,
            'aggregated_ae': ae_list,
            'timestamp': __import__('time').time()
        }
        self.publish(TOPIC_CH_TO_CM, f'ch-{ch_id}-to-cm-{cm_id}', message)

    def publish_reliability(self, vehicle_id: int, reliability: float, role: str) -> None:

        from .kafka_topics import TOPIC_RELIABILITY

        message = {
            'type': 'reliability',
            'source_id': vehicle_id,
            'role': role,
            'reliability': reliability,
            'timestamp': __import__('time').time()
        }
        self.publish(TOPIC_RELIABILITY, f'reliability-{vehicle_id}', message)

    def publish_cluster_update(self, ch_id: int, members: List[int]) -> None:

        from .kafka_topics import TOPIC_CLUSTER_UPDATE

        message = {
            'type': 'cluster_update',
            'ch_id': ch_id,
            'members': members,
            'timestamp': __import__('time').time()
        }
        self.publish(TOPIC_CLUSTER_UPDATE, f'cluster-{ch_id}', message)

    def _delivery_report(self, err, msg) -> None:
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")
