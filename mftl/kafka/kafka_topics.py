







TOPIC_HELLO = 'mtfl.hello'


TOPIC_VEHICLE_INFO = 'mtfl.vehicle.info'


TOPIC_CM_TO_CH = 'mtfl.cm.ch'


TOPIC_CH_TO_EPC = 'mtfl.ch.epc'


TOPIC_EPC_TO_CH = 'mtfl.epc.ch'


TOPIC_CH_TO_CM = 'mtfl.ch.cm'


TOPIC_RELIABILITY = 'mtfl.reliability'


TOPIC_CLUSTER_UPDATE = 'mtfl.cluster.update'





def get_vehicle_topic(vehicle_id: int, topic_type: str) -> str:

    return f'mtfl.vehicle.{vehicle_id}.{topic_type}'


def get_ch_topic(ch_id: int, topic_type: str) -> str:

    return f'mtfl.ch.{ch_id}.{topic_type}'


def get_all_mtfl_topics() -> list:

    return [
        TOPIC_HELLO,
        TOPIC_VEHICLE_INFO,
        TOPIC_CM_TO_CH,
        TOPIC_CH_TO_EPC,
        TOPIC_EPC_TO_CH,
        TOPIC_CH_TO_CM,
        TOPIC_RELIABILITY,
        TOPIC_CLUSTER_UPDATE
    ]
