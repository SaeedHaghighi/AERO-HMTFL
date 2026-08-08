



import json
import logging
import numpy as np
from typing import Dict, Any, Optional, Callable, List
from confluent_kafka import Consumer, KafkaError

from .kafka_config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_GROUP_ID,
    MAX_POLL_RECORDS
)

logger = logging.getLogger(__name__)


class KafkaConsumerWrapper:


    def __init__(self, consumer_id: str, topics: List[str]):
        config = {
            'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
            'group.id': f"{KAFKA_GROUP_ID}-{consumer_id}",
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True,
            'auto.commit.interval.ms': 5000,
            'max.poll.interval.ms': 300000,
            'max.poll.records': MAX_POLL_RECORDS
        }
        self.consumer = Consumer(config)
        self.consumer.subscribe(topics)
        self.consumer_id = consumer_id
        self.topics = topics
        self._running = False
        logger.info(f"Consumer {consumer_id} subscribed to: {topics}")

    def consume(self, callback: Callable[[str, str, Dict[str, Any]], None], timeout: float = 1.0) -> None:

        self._running = True
        while self._running:
            try:
                msg = self.consumer.poll(timeout)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"Consumer error: {msg.error()}")
                        break
                try:
                    key = msg.key().decode('utf-8') if msg.key() else None
                    value = json.loads(msg.value().decode('utf-8'))
                    callback(msg.topic(), key, value)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Consumer poll error: {e}")
                break

    def stop(self) -> None:
        self._running = False
        self.consumer.close()
        logger.info(f"Consumer {self.consumer_id} stopped")

    def consume_once(self) -> Optional[Dict[str, Any]]:

        msg = self.consumer.poll(0.1)
        if msg is None:
            return None
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                return None
            logger.error(f"Consumer error: {msg.error()}")
            return None
        try:
            return {
                'topic': msg.topic(),
                'key': msg.key().decode('utf-8') if msg.key() else None,
                'value': json.loads(msg.value().decode('utf-8'))
            }
        except Exception as e:
            logger.error(f"Error parsing message: {e}")
            return None


class VehicleInfoCollector:


    def __init__(self, consumer_id: str = 'info-collector'):
        from .kafka_topics import TOPIC_HELLO
        self.consumer = KafkaConsumerWrapper(consumer_id, [TOPIC_HELLO])
        self.vehicle_info: Dict[int, Dict[str, Any]] = {}

    def update_vehicle_info(self) -> None:

        messages = self._get_hello_packets(timeout=1.0)
        for msg in messages:
            vehicle_id = msg.get('source_id')
            data = msg.get('data', {})
            if vehicle_id is not None:
                self.vehicle_info[vehicle_id] = data

    def _get_hello_packets(self, timeout: float = 2.0) -> List[Dict[str, Any]]:
        messages = []
        start_time = __import__('time').time()
        while __import__('time').time() - start_time < timeout:
            msg = self.consumer.consume_once()
            if msg is None:
                continue
            try:
                value = msg['value']
                if value.get('type') == 'hello_packet':
                    messages.append(value)
            except Exception:
                continue
        return messages

    def get_vehicle_info(self, vehicle_id: int) -> Optional[Dict[str, Any]]:
        return self.vehicle_info.get(vehicle_id)

    def get_all_vehicles(self) -> Dict[int, Dict[str, Any]]:
        return self.vehicle_info.copy()

    def stop(self) -> None:
        self.consumer.stop()


class CHMessageCollector:


    def __init__(self, ch_id: int, consumer_id: Optional[str] = None):
        from .kafka_topics import TOPIC_CM_TO_CH
        self.ch_id = ch_id
        self.consumer_id = consumer_id or f'ch-{ch_id}-collector'
        self.consumer = KafkaConsumerWrapper(self.consumer_id, [TOPIC_CM_TO_CH])
        self.cm_updates: Dict[int, Dict[str, Any]] = {}

    def collect_cm_updates(self, timeout: float = 5.0) -> Dict[int, Dict[str, Any]]:

        start_time = __import__('time').time()
        while __import__('time').time() - start_time < timeout:
            msg = self.consumer.consume_once()
            if msg is None:
                continue
            try:
                value = msg['value']
                if value.get('type') == 'cm_ae_params':
                    cm_id = value.get('source_id')
                    ch_id = value.get('ch_id')
                    if ch_id == self.ch_id and cm_id is not None:
                        self.cm_updates[cm_id] = value
            except Exception:
                continue
        return self.cm_updates

    def clear(self) -> None:
        self.cm_updates = {}

    def stop(self) -> None:
        self.consumer.stop()


class EPCCollector:


    def __init__(self, consumer_id: str = 'epc-collector'):
        from .kafka_topics import TOPIC_CH_TO_EPC
        self.consumer = KafkaConsumerWrapper(consumer_id, [TOPIC_CH_TO_EPC])
        self.ch_updates: Dict[int, Dict[str, Any]] = {}

    def collect_ch_updates(self, timeout: float = 5.0) -> Dict[int, Dict[str, Any]]:

        start_time = __import__('time').time()
        while __import__('time').time() - start_time < timeout:
            msg = self.consumer.consume_once()
            if msg is None:
                continue
            try:
                value = msg['value']
                if value.get('type') == 'ch_aggregated':
                    ch_id = value.get('source_id')
                    if ch_id is not None:
                        self.ch_updates[ch_id] = value
            except Exception:
                continue
        return self.ch_updates

    def clear(self) -> None:
        self.ch_updates = {}

    def stop(self) -> None:
        self.consumer.stop()
