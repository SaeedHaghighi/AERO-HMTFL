



import random
import time
from typing import List, Tuple, Dict, Set, Optional
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from mftl.models import Vehicle, ClusterHead, EPC
from mftl.data import load_data_for_vehicles
from mftl.utils.helpers import _sanitize_vector
from mftl.config import *


class Simulator:


    def __init__(
        self,
        num_vehicles: int = DEFAULT_NUM_VEHICLES,
        cluster_size: int = DEFAULT_CLUSTER_SIZE,
        attacker_fraction: float = DEFAULT_ATTACKER_FRACTION,
        rounds: int = DEFAULT_ROUNDS,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA,
        transmission_range: float = DEFAULT_TRANSMISSION_RANGE,
        speed_min: float = DEFAULT_SPEED_MIN,
        speed_max: float = DEFAULT_SPEED_MAX,
        cifar_k: Optional[int] = 10,
        use_mtfl: bool = True,
        lambda_acc: float = DEFAULT_LAMBDA_ACC,
        lambda_freq: float = DEFAULT_LAMBDA_FREQ,
        mu: float = DEFAULT_MU,
        latent_dim: int = DEFAULT_LATENT_DIM,
        epsilon: float = DEFAULT_EPSILON
    ) -> None:

        self.num_vehicles = num_vehicles
        self.cluster_size = max(2, cluster_size)
        self.attacker_fraction = max(0.0, min(1.0, attacker_fraction))
        self.rounds = max(10, rounds)

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.TRANSMISSION_RANGE = float(transmission_range)
        self.SPEED_MIN = float(speed_min)
        self.SPEED_MAX = float(speed_max)

        self.use_mtfl = use_mtfl
        self.lambda_acc = lambda_acc
        self.lambda_freq = lambda_freq
        self.mu = mu
        self.latent_dim = latent_dim
        self.epsilon = epsilon


        list_X, list_y, all_X, all_y = load_data_for_vehicles(num_vehicles)
        X_train, X_test, y_train, y_test = train_test_split(
            all_X, all_y, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_SEED
        )


        try:
            self.cnn_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        except Exception:
            self.cnn_device = None


        self.cifar10_train_loaders: Dict[int, DataLoader] = {}
        self.cifar10_test_loaders: Dict[int, DataLoader] = {}
        self._initialize_cifar_loaders(num_vehicles, cifar_k)


        self.gtsrb_train_loaders: Dict[int, DataLoader] = {}
        self.gtsrb_test_loaders: Dict[int, DataLoader] = {}
        self.gtsrb_num_classes = 43
        self._initialize_gtsrb_loaders(num_vehicles)


        self.public_buffers: Dict[int, Dict[str, DataLoader]] = {}
        self._initialize_public_buffers(num_vehicles, X_train, y_train)


        self.vehicles: Dict[int, Vehicle] = {}
        self._initialize_vehicles(num_vehicles, list_X, list_y, X_test, y_test)


        self.epc = EPC(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            cnn_device=self.cnn_device,
            cifar10_train_loader=self.cifar10_train_loaders.get(1),
            cifar10_test_loader=self.cifar10_test_loaders.get(1),
            gtsrb_train_loader=self.gtsrb_train_loaders.get(1),
            gtsrb_test_loader=self.gtsrb_test_loaders.get(1),
            gtsrb_num_classes=self.gtsrb_num_classes
        )


        self.cluster_heads: Dict[int, ClusterHead] = {}
        self._assign_clusters_initial()


        self.vehicle_last_ch_encoded: Dict[int, np.ndarray] = {}

    def _initialize_public_buffers(self, num_vehicles: int, X_train: np.ndarray, y_train: np.ndarray) -> None:
        for vid in range(1, num_vehicles + 1):
            self.public_buffers[vid] = {}
            if X_train.shape[0] > 0:
                buffer_size = min(100, len(X_train))
                indices = np.random.choice(len(X_train), buffer_size, replace=False)
                X_buffer, y_buffer = X_train[indices], y_train[indices]
                dataset = torch.utils.data.TensorDataset(
                    torch.tensor(X_buffer, dtype=torch.float32),
                    torch.tensor(y_buffer, dtype=torch.long)
                )
                self.public_buffers[vid]['task_a'] = DataLoader(dataset, batch_size=32, shuffle=True)
                self.public_buffers[vid]['task_b'] = DataLoader(dataset, batch_size=32, shuffle=True)

    def _initialize_vehicles(self, num_vehicles: int, list_X: List[np.ndarray],
                            list_y: List[np.ndarray], X_test: np.ndarray, y_test: np.ndarray) -> None:
        attacker_count = int(self.attacker_fraction * num_vehicles)
        attacker_ids = set(random.sample(range(1, num_vehicles + 1), attacker_count))

        for vid in range(1, num_vehicles + 1):
            veh = Vehicle(
                source_id=vid,
                X_train=list_X[vid], y_train=list_y[vid],
                X_test=X_test, y_test=y_test,
                is_attacker=(vid in attacker_ids),
                cnn_device=self.cnn_device,
                cifar10_train_loader=self.cifar10_train_loaders.get(vid),
                cifar10_test_loader=self.cifar10_test_loaders.get(vid),
                gtsrb_train_loader=self.gtsrb_train_loaders.get(vid),
                gtsrb_test_loader=self.gtsrb_test_loaders.get(vid),
                gtsrb_num_classes=self.gtsrb_num_classes,
                use_mtfl=self.use_mtfl,
                latent_dim=self.latent_dim,
                public_buffer=self.public_buffers.get(vid, {}),
                task_names=['task_a', 'task_b'],
                transmission_range=self.TRANSMISSION_RANGE
            )
            veh.speed = random.uniform(self.SPEED_MIN, self.SPEED_MAX)
            self.vehicles[vid] = veh

    def _initialize_cifar_loaders(self, num_vehicles: int, cifar_k: Optional[int]) -> None:
        try:
            cifar_transform_train = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            cifar_transform_test = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            cifar_train = torchvision.datasets.CIFAR10(
                root="./data", train=True, download=True, transform=cifar_transform_train
            )
            cifar_test = torchvision.datasets.CIFAR10(
                root="./data", train=False, download=True, transform=cifar_transform_test
            )


            desired_class_names = ["airplane", "automobile", "bird", "cat", "deer",
                                   "dog", "frog", "horse", "ship", "truck"]
            name_to_idx = {name: idx for idx, name in enumerate(cifar_train.classes)}
            selected_classes = np.array([name_to_idx[name] for name in desired_class_names], dtype=int)

            train_targets = np.array(cifar_train.targets)
            test_targets = np.array(cifar_test.targets)
            train_idx = np.where(np.isin(train_targets, selected_classes))[0]
            test_idx = np.where(np.isin(test_targets, selected_classes))[0]

            rng = np.random.RandomState(42)
            rng.shuffle(train_idx)
            rng.shuffle(test_idx)

            train_splits = np.array_split(train_idx, num_vehicles)
            test_splits = np.array_split(test_idx, num_vehicles)

            pin_mem = bool(self.cnn_device is not None and self.cnn_device.type == 'cuda')
            for vid in range(1, num_vehicles + 1):
                train_subset = Subset(cifar_train, train_splits[vid - 1])
                test_subset = Subset(cifar_test, test_splits[vid - 1])
                self.cifar10_train_loaders[vid] = DataLoader(
                    train_subset, batch_size=64, shuffle=True, num_workers=0, pin_memory=pin_mem
                )
                self.cifar10_test_loaders[vid] = DataLoader(
                    test_subset, batch_size=128, shuffle=False, num_workers=0, pin_memory=pin_mem
                )
        except Exception as e:
            print(f"Error initializing CIFAR-10 loaders: {e}")
            self.cifar10_train_loaders = {}
            self.cifar10_test_loaders = {}

    def _initialize_gtsrb_loaders(self, num_vehicles: int) -> None:
        try:
            gtsrb_transform_train = T.Compose([
                T.Resize((32, 32)), T.RandomRotation(10),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                T.ToTensor(),
                T.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
            ])
            gtsrb_transform_test = T.Compose([
                T.Resize((32, 32)), T.ToTensor(),
                T.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
            ])

            gtsrb_train = torchvision.datasets.GTSRB(
                root="./data", split='train', download=True, transform=gtsrb_transform_train
            )
            gtsrb_test = torchvision.datasets.GTSRB(
                root="./data", split='test', download=True, transform=gtsrb_transform_test
            )

            rng = np.random.RandomState(123)
            train_idx = np.arange(len(gtsrb_train))
            test_idx = np.arange(len(gtsrb_test))
            rng.shuffle(train_idx)
            rng.shuffle(test_idx)

            train_splits = np.array_split(train_idx, num_vehicles)
            test_splits = np.array_split(test_idx, num_vehicles)

            pin_mem = bool(self.cnn_device is not None and self.cnn_device.type == 'cuda')
            for vid in range(1, num_vehicles + 1):
                train_subset = Subset(gtsrb_train, train_splits[vid - 1])
                test_subset = Subset(gtsrb_test, test_splits[vid - 1])
                self.gtsrb_train_loaders[vid] = DataLoader(
                    train_subset, batch_size=64, shuffle=True, num_workers=0, pin_memory=pin_mem
                )
                self.gtsrb_test_loaders[vid] = DataLoader(
                    test_subset, batch_size=128, shuffle=False, num_workers=0, pin_memory=pin_mem
                )
        except Exception as e:
            print(f"Error initializing GTSRB loaders: {e}")
            self.gtsrb_train_loaders = {}
            self.gtsrb_test_loaders = {}



    def _calculate_avg_cosim(self, v1: Vehicle, v2: Vehicle) -> float:

        speed_diff = abs(v1.speed - v2.speed)
        avg_speed = (v1.speed + v2.speed) / 2
        delta_v = min(1.0, speed_diff / (avg_speed + 1e-6))

        if self.use_mtfl:
            enc_sim = v1.get_encoder_similarity(v2)
        else:
            enc_sim = self._weights_cosine_similarity(v1, v2)
        enc_dissim = 1 - max(0, min(1, enc_sim))

        if self.use_mtfl:
            task_sim = v1.get_task_similarity(v2)
        else:
            task_sim = self._jaccard_tasks(v1, v2)
        task_dissim = 1 - max(0, min(1, task_sim))

        return self.alpha * delta_v + self.beta * enc_dissim + self.gamma * task_dissim

    def _get_neighbors_mtfl(self, vid: int) -> List[int]:
        neighbors = []
        v1 = self.vehicles[vid]
        for other_id, v2 in self.vehicles.items():
            if vid == other_id:
                continue
            distance = np.sqrt((v1.position_x - v2.position_x)**2 + (v1.position_y - v2.position_y)**2)
            if distance <= self.TRANSMISSION_RANGE:
                if self._calculate_avg_cosim(v1, v2) < 1.0:
                    neighbors.append(other_id)
        return neighbors

    def _assign_clusters_dynamic(self) -> None:

        prev_ch_ids = set(self.cluster_heads.keys())


        for v in self.vehicles.values():
            v.role = ROLE_SE
            v.cluster_head_id = None
        self.cluster_heads.clear()


        vehicle_ids = list(self.vehicles.keys())
        adjacency: Dict[int, List[int]] = {vid: [] for vid in vehicle_ids}

        for vid in vehicle_ids:
            v1 = self.vehicles[vid]
            if self.use_mtfl:
                neighbors = self._get_neighbors_mtfl(vid)
            else:
                neighbors = self._get_neighbors(vid)

            for nid in neighbors:
                if nid == vid:
                    continue
                v2 = self.vehicles[nid]
                if self.use_mtfl:
                    if self._calculate_avg_cosim(v1, v2) < 1.0:
                        adjacency[vid].append(nid)
                        adjacency[nid].append(vid)
                else:
                    cos_ok = self._weights_cosine_similarity(v1, v2) >= self.alpha
                    speed_ok = self._speed_similarity_val(v1, v2) >= self.beta
                    j_ok = self._jaccard_tasks(v1, v2) >= self.gamma
                    if cos_ok and speed_ok and j_ok:
                        adjacency[vid].append(nid)
                        adjacency[nid].append(vid)


        visited = set()
        clusters: List[List[int]] = []
        for vid in vehicle_ids:
            if vid in visited:
                continue
            stack = [vid]
            comp = []
            visited.add(vid)
            while stack:
                curr = stack.pop()
                comp.append(curr)
                for nb in adjacency[curr]:
                    if nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
            clusters.append(comp)


        for comp in clusters:
            if not comp:
                continue
            degrees = {node: len(adjacency[node]) for node in comp}
            ch_id = max(degrees, key=degrees.get)

            self.vehicles[ch_id].role = ROLE_CH
            self.vehicles[ch_id].cluster_head_id = ch_id

            ch = ClusterHead(
                self.vehicles[ch_id],
                lambda_acc=self.lambda_acc if self.use_mtfl else 0.6,
                lambda_freq=self.lambda_freq if self.use_mtfl else 0.4
            )

            for node in comp:
                if node == ch_id:
                    continue
                if len(ch.member_ids) < 10:
                    self.vehicles[node].role = ROLE_CM
                    self.vehicles[node].cluster_head_id = ch_id
                    ch.add_member(self.vehicles[node])

            self.cluster_heads[ch_id] = ch


            ch.publish_cluster_update()


        empty_chs = [ch_id for ch_id, ch in self.cluster_heads.items() if len(ch.member_ids) == 0]
        for ch_id in empty_chs:
            v = self.vehicles.get(ch_id)
            if v is not None:
                v.role = ROLE_SE
                v.cluster_head_id = None
            del self.cluster_heads[ch_id]


        for prev_ch in prev_ch_ids:
            v = self.vehicles.get(prev_ch)
            if v is None:
                continue
            if v.role == ROLE_CM and prev_ch in self.vehicle_last_ch_encoded:
                new_ch = v.cluster_head_id
                if new_ch in self.cluster_heads:
                    vec = _sanitize_vector(self.vehicle_last_ch_encoded[prev_ch])
                    incoming = getattr(self.cluster_heads[new_ch], "incoming_transfers", [])
                    incoming.append(vec)
                    self.cluster_heads[new_ch].incoming_transfers = incoming

        if not self.cluster_heads:
            print("No clusters formed, attempting fallback clustering...")
            self._fallback_clustering()



    def _round_once(self) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, Tuple[float, float]], float, float, float]:


        for v in self.vehicles.values():
            v.reset_training_rounds()


        for _ in range(3):
            for v in self.vehicles.values():
                v.update_position(time_delta=0.5)


        self._assign_clusters_dynamic()

        if not self.cluster_heads:
            print("No clusters formed. Skipping this round.")
            return {}, {}, 0.0, 0.0, 0.0

        print(f"\nClusters: {len(self.cluster_heads)} CHs")
        for ch_id, ch in self.cluster_heads.items():
            print(f"  CH {ch_id}: {len(ch.member_ids)} members")

        ch_accuracy: Dict[int, Tuple[float, float]] = {}
        cm_accuracy: Dict[int, Tuple[float, float]] = {}
        ch_weights_for_epc: List[np.ndarray] = []

        if self.use_mtfl:

            member_updates: Dict[int, Dict[int, np.ndarray]] = {}
            member_scores: Dict[int, Dict[int, float]] = {}


            for ch_id, ch in self.cluster_heads.items():
                member_updates[ch_id] = {}
                member_scores[ch_id] = {}


                ch_score, _ = ch.vehicle.train_mtfl_local(mu=self.mu, epsilon=self.epsilon)
                ch.vehicle.update_reliability(ch_score)


                for mid in ch.member_ids:
                    member = ch.member_vehicles.get(mid)
                    if member:
                        score, _ = member.train_mtfl_local(mu=self.mu, epsilon=self.epsilon)
                        member.update_reliability(score)
                        ae_params = member.get_mtfl_ae_params()
                        member.send_cm_to_ch(ch_id, ae_params, score)


                time.sleep(1.0)
                cm_updates = ch.collect_cm_updates(timeout=3.0)


                for cm_id, data in cm_updates.items():
                    ae_list = data.get('ae_params', [])
                    score = data.get('validation_score', 0.0)
                    if ae_list:
                        member_updates[ch_id][cm_id] = np.array(ae_list)
                        member_scores[ch_id][cm_id] = score


                member_updates[ch_id][ch.vehicle.source_id] = ch.vehicle.get_mtfl_ae_params()
                member_scores[ch_id][ch.vehicle.source_id] = ch_score


                incoming = getattr(ch, "incoming_transfers", [])
                for vec in incoming:
                    member_updates[ch_id][-1] = _sanitize_vector(vec)
                ch.incoming_transfers = []


            for ch_id, ch in self.cluster_heads.items():
                if member_updates[ch_id]:
                    aggregated = ch.aggregate_members_mtfl(
                        member_updates[ch_id],
                        member_scores[ch_id]
                    )
                    self.vehicle_last_ch_encoded[ch_id] = aggregated
                else:
                    aggregated = ch.vehicle.get_mtfl_ae_params()
                    self.vehicle_last_ch_encoded[ch_id] = aggregated


                ch.broadcast_to_members(aggregated)


                ch.send_to_epc(aggregated)
                ch_weights_for_epc.append(aggregated)


                acc_a, acc_b = ch.vehicle.evaluate_two_tasks()
                ch_accuracy[ch_id] = (acc_a, acc_b)


                for mid in ch.member_ids:
                    member = ch.member_vehicles.get(mid)
                    if member:
                        acc_ma, acc_mb = member.evaluate_two_tasks()
                        cm_accuracy[mid] = (acc_ma, acc_mb)


            time.sleep(1.0)
            ch_updates = self.epc.collect_ch_updates(timeout=3.0)


            ch_weights_for_epc = []
            for ch_id, data in ch_updates.items():
                ae_list = data.get('aggregated_ae', [])
                if ae_list:
                    ch_weights_for_epc.append(np.array(ae_list))


            epc_vec = self.epc.aggregate_from_ch(ch_weights_for_epc)
            epc_mnist_acc, epc_cifar10_acc, epc_gtsrb_acc = self.epc.get_latest_accuracies()


            if epc_vec is not None:
                ch_ids = list(self.cluster_heads.keys())
                self.epc.broadcast_to_chs(ch_ids, epc_vec)


                for ch_id, ch in self.cluster_heads.items():
                    received = ch.receive_from_epc(timeout=2.0)
                    if received is not None:
                        ch.vehicle.set_mtfl_ae_params(received)
                        ch.vehicle.train_mtfl_local(mu=self.mu, epsilon=self.epsilon)
                        acc_a, acc_b = ch.vehicle.evaluate_two_tasks()
                        ch_accuracy[ch_id] = (acc_a, acc_b)


                        for mid in ch.member_ids:
                            member = ch.member_vehicles.get(mid)
                            if member:

                                received_cm = member.receive_from_ch(timeout=1.0)
                                if received_cm is not None:
                                    member.set_mtfl_ae_params(received_cm)
                                    member.train_mtfl_local(mu=self.mu, epsilon=self.epsilon)
                                    acc_ma, acc_mb = member.evaluate_two_tasks()
                                    cm_accuracy[mid] = (acc_ma, acc_mb)

            return ch_accuracy, cm_accuracy, epc_mnist_acc, epc_cifar10_acc, epc_gtsrb_acc

        else:

            ch_to_member_updates: Dict[int, List[np.ndarray]] = {ch_id: [] for ch_id in self.cluster_heads}

            for ch_id, ch in self.cluster_heads.items():
                ch.vehicle.train_local_two_tasks()
                ch_to_member_updates[ch_id].append(ch.vehicle.message_to_ch_flat())

                for mid in ch.member_ids:
                    member = self.vehicles[mid]
                    member.train_local_two_tasks()
                    ch_to_member_updates[ch_id].append(member.message_to_ch_flat())

                incoming = getattr(ch, "incoming_transfers", [])
                for vec in incoming:
                    ch_to_member_updates[ch_id].append(_sanitize_vector(vec))
                ch.incoming_transfers = []

            for ch_id, ch in self.cluster_heads.items():
                if ch_to_member_updates[ch_id]:
                    agg_flat = ch.aggregate_encoded(ch_to_member_updates[ch_id])
                    self.vehicle_last_ch_encoded[ch_id] = agg_flat
                else:
                    flat_ch = ch.vehicle.message_to_ch_flat()
                    self.vehicle_last_ch_encoded[ch_id] = flat_ch

                ch_to_cm_initial = ch.message_to_cm_flat()
                for mid in ch.member_ids:
                    member = self.vehicles[mid]
                    member.set_weights_from_flat(ch_to_cm_initial)
                    member.train_local_two_tasks()

                ch.vehicle.train_local_two_tasks()
                ch_epc_message = ch.message_to_epc_flat()
                ch_weights_for_epc.append(ch_epc_message)

                acc_a, acc_b = ch.vehicle.evaluate_two_tasks()
                ch_accuracy[ch_id] = (acc_a, acc_b)

                for mid in ch.member_ids:
                    member = self.vehicles[mid]
                    acc_ma, acc_mb = member.evaluate_two_tasks()
                    cm_accuracy[mid] = (acc_ma, acc_mb)

            epc_vec = self.epc.aggregate_from_ch(ch_weights_for_epc)
            epc_mnist_acc, epc_cifar10_acc, epc_gtsrb_acc = self.epc.get_latest_accuracies()

            if epc_vec is not None:
                for ch_id, ch in self.cluster_heads.items():
                    epc_to_ch_message = self.epc.message_to_ch_flat(epc_vec)
                    ch.vehicle.set_weights_from_flat(epc_to_ch_message)
                    ch.vehicle.train_local_two_tasks()
                    acc_a, acc_b = ch.vehicle.evaluate_two_tasks()
                    ch_accuracy[ch_id] = (acc_a, acc_b)

                    ch_to_cm_message = ch.message_to_cm_flat()
                    for mid in ch.member_ids:
                        member = self.vehicles[mid]
                        member.set_weights_from_flat(ch_to_cm_message)
                        member.train_local_two_tasks()
                        acc_ma, acc_mb = member.evaluate_two_tasks()
                        cm_accuracy[mid] = (acc_ma, acc_mb)

            return ch_accuracy, cm_accuracy, epc_mnist_acc, epc_cifar10_acc, epc_gtsrb_acc



    def _get_neighbors(self, vid: int) -> List[int]:
        neighbors = []
        v1 = self.vehicles[vid]
        for other_id, v2 in self.vehicles.items():
            if vid == other_id:
                continue
            distance = np.sqrt((v1.position_x - v2.position_x)**2 + (v1.position_y - v2.position_y)**2)
            if distance <= self.TRANSMISSION_RANGE:
                neighbors.append(other_id)
        return neighbors

    def _speed_similarity_val(self, v1: Vehicle, v2: Vehicle) -> float:
        speed_diff = abs(v1.speed - v2.speed)
        max_speed = max(max(v1.speed, v2.speed), 1.0)
        return 1 - (speed_diff / max_speed)

    def _weights_cosine_similarity(self, v1: Vehicle, v2: Vehicle) -> float:
        if getattr(v1, 'weights_two', None) is None:
            v1.train_local_two_tasks()
        if getattr(v2, 'weights_two', None) is None:
            v2.train_local_two_tasks()
        try:
            v1_vec = np.concatenate([_sanitize_vector(v1.weights_two.coeff_a),
                                     _sanitize_vector(v1.weights_two.coeff_b)]).ravel()
            v2_vec = np.concatenate([_sanitize_vector(v2.weights_two.coeff_a),
                                     _sanitize_vector(v2.weights_two.coeff_b)]).ravel()
            if np.linalg.norm(v1_vec) == 0 or np.linalg.norm(v2_vec) == 0:
                return 0.0
            from sklearn.metrics.pairwise import cosine_similarity
            return float(cosine_similarity(v1_vec.reshape(1, -1), v2_vec.reshape(1, -1))[0][0])
        except Exception:
            return 0.0

    def _vehicle_task_set(self, v: Vehicle) -> Set[str]:
        tasks = set()
        if getattr(v, 'ya_train', None) is None or getattr(v, 'yb_train', None) is None:
            v.train_local_two_tasks()
        if v.ya_train.size > 0:
            tasks.add('A')
        if v.yb_train.size > 0:
            tasks.add('B')
        return tasks

    def _jaccard_tasks(self, v1: Vehicle, v2: Vehicle) -> float:
        s1 = self._vehicle_task_set(v1)
        s2 = self._vehicle_task_set(v2)
        if not s1 and not s2:
            return 1.0
        union = len(s1 | s2)
        inter = len(s1 & s2)
        return inter / union if union > 0 else 0.0



    def _fallback_clustering(self) -> None:
        print("Warning: No clusters formed, attempting fallback clustering...")
        for v in self.vehicles.values():
            v.role = ROLE_SE
        self.cluster_heads = {}

        strategies = [
            ("Distance + Speed", self._cluster_by_distance_and_speed),
            ("Distance + Weights", self._cluster_by_distance_and_weights),
            ("Distance + Tasks", self._cluster_by_distance_and_tasks),
            ("Distance Only", self._cluster_by_distance_only),
            ("Position Based", self._cluster_by_position)
        ]
        for strategy_name, strategy_func in strategies:
            print(f"Trying {strategy_name} clustering...")
            strategy_func()
            if self.cluster_heads:
                print(f"Successfully formed {len(self.cluster_heads)} clusters using {strategy_name}")
                return
        print("All fallback clustering strategies failed!")

    def _cluster_by_distance_and_speed(self) -> None:
        unassigned = list(self.vehicles.keys())
        while len(unassigned) >= 2:
            best_ch, best_score = None, -1
            for vid in unassigned:
                score = self._calculate_ch_score_relaxed(vid, unassigned)
                if score > best_score:
                    best_score, best_ch = score, vid
            if best_ch is None or best_score < 0.1:
                break
            ch_vehicle = self.vehicles[best_ch]
            ch_vehicle.role = ROLE_CH
            ch = ClusterHead(ch_vehicle, self.lambda_acc, self.lambda_freq)
            ch.member_ids = [best_ch]
            unassigned.remove(best_ch)
            for mid in unassigned[:]:
                member = self.vehicles[mid]
                distance = np.sqrt((ch_vehicle.position_x - member.position_x)**2 + (ch_vehicle.position_y - member.position_y)**2)
                speed_diff = abs(ch_vehicle.speed - member.speed)
                if distance < 300 and speed_diff < 10:
                    member.role = ROLE_CM
                    member.cluster_head_id = best_ch
                    ch.add_member(member)
                    unassigned.remove(mid)
            self.cluster_heads[best_ch] = ch

    def _cluster_by_distance_and_weights(self) -> None:
        unassigned = list(self.vehicles.keys())
        while len(unassigned) >= 2:
            best_ch = unassigned[0]
            ch_vehicle = self.vehicles[best_ch]
            ch_vehicle.role = ROLE_CH
            ch = ClusterHead(ch_vehicle, self.lambda_acc, self.lambda_freq)
            ch.member_ids = [best_ch]
            unassigned.remove(best_ch)
            for mid in unassigned[:]:
                member = self.vehicles[mid]
                distance = np.sqrt((ch_vehicle.position_x - member.position_x)**2 + (ch_vehicle.position_y - member.position_y)**2)
                if distance < 400:
                    member.role = ROLE_CM
                    member.cluster_head_id = best_ch
                    ch.add_member(member)
                    unassigned.remove(mid)
            self.cluster_heads[best_ch] = ch

    def _cluster_by_distance_and_tasks(self) -> None:
        unassigned = list(self.vehicles.keys())
        while len(unassigned) >= 2:
            best_ch = unassigned[0]
            ch_vehicle = self.vehicles[best_ch]
            ch_vehicle.role = ROLE_CH
            ch = ClusterHead(ch_vehicle, self.lambda_acc, self.lambda_freq)
            ch.member_ids = [best_ch]
            unassigned.remove(best_ch)
            for mid in unassigned[:]:
                member = self.vehicles[mid]
                distance = np.sqrt((ch_vehicle.position_x - member.position_x)**2 + (ch_vehicle.position_y - member.position_y)**2)
                if distance < 500:
                    member.role = ROLE_CM
                    member.cluster_head_id = best_ch
                    ch.add_member(member)
                    unassigned.remove(mid)
            self.cluster_heads[best_ch] = ch

    def _cluster_by_distance_only(self) -> None:
        unassigned = list(self.vehicles.keys())
        while len(unassigned) >= 2:
            best_ch = unassigned[0]
            ch_vehicle = self.vehicles[best_ch]
            ch_vehicle.role = ROLE_CH
            ch = ClusterHead(ch_vehicle, self.lambda_acc, self.lambda_freq)
            ch.member_ids = [best_ch]
            unassigned.remove(best_ch)
            for mid in unassigned[:]:
                member = self.vehicles[mid]
                distance = np.sqrt((ch_vehicle.position_x - member.position_x)**2 + (ch_vehicle.position_y - member.position_y)**2)
                if distance < 600:
                    member.role = ROLE_CM
                    member.cluster_head_id = best_ch
                    ch.add_member(member)
                    unassigned.remove(mid)
            self.cluster_heads[best_ch] = ch

    def _cluster_by_position(self) -> None:
        unassigned = list(self.vehicles.keys())
        regions = [(0, 500, 0, 500), (500, 1000, 0, 500), (0, 500, 500, 1000), (500, 1000, 500, 1000)]
        for region_x_min, region_x_max, region_y_min, region_y_max in regions:
            region_vehicles = []
            for vid in unassigned[:]:
                v = self.vehicles[vid]
                if region_x_min <= v.position_x < region_x_max and region_y_min <= v.position_y < region_y_max:
                    region_vehicles.append(vid)
                    unassigned.remove(vid)
            if len(region_vehicles) >= 2:
                ch_id = region_vehicles[0]
                ch_vehicle = self.vehicles[ch_id]
                ch_vehicle.role = ROLE_CH
                ch = ClusterHead(ch_vehicle, self.lambda_acc, self.lambda_freq)
                ch.member_ids = [ch_id]
                for mid in region_vehicles[1:]:
                    member = self.vehicles[mid]
                    member.role = ROLE_CM
                    member.cluster_head_id = ch_id
                    ch.add_member(member)
                self.cluster_heads[ch_id] = ch

    def _calculate_ch_score_relaxed(self, vid: int, unassigned: List[int]) -> float:
        v = self.vehicles[vid]
        neighbors = []
        for other_id in unassigned:
            if other_id == vid:
                continue
            other = self.vehicles[other_id]
            distance = np.sqrt((v.position_x - other.position_x)**2 + (v.position_y - other.position_y)**2)
            speed_diff = abs(v.speed - other.speed)
            if distance < 300 and speed_diff < 10:
                neighbors.append(other_id)
        if not neighbors:
            return 0.0
        similarities = []
        for nid in neighbors:
            neighbor = self.vehicles[nid]
            distance = np.sqrt((v.position_x - neighbor.position_x)**2 + (v.position_y - neighbor.position_y)**2)
            speed_diff = abs(v.speed - neighbor.speed)
            similarity = 1.0 / (1.0 + distance/100 + speed_diff/5)
            similarities.append(similarity)
        return sum(similarities) / len(similarities) * len(neighbors)

    def _assign_clusters_initial(self) -> None:
        sorted_ids = sorted(self.vehicles.keys())
        for i in range(0, len(sorted_ids), self.cluster_size):
            block = sorted_ids[i:i + self.cluster_size]
            if not block:
                continue
            ch_id = block[0]
            self.vehicles[ch_id].role = ROLE_CH
            self.vehicles[ch_id].cluster_head_id = ch_id
            ch = ClusterHead(self.vehicles[ch_id], self.lambda_acc, self.lambda_freq)
            members = block[1:]
            ch.member_ids = members
            for mid in members:
                self.vehicles[mid].role = ROLE_CM
                self.vehicles[mid].cluster_head_id = ch_id
                ch.add_member(self.vehicles[mid])
            self.cluster_heads[ch_id] = ch



    def run(self) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], List[Dict[str, float]]]:

        ch_accuracies, cm_accuracies, epc_accuracies = [], [], []
        skipped_rounds = 0

        for round_num in range(self.rounds):
            print(f"\n=== Round {round_num + 1}/{self.rounds} (Kafka) ===")
            ch_acc, cm_acc, epc_mnist_acc, epc_cifar10_acc, epc_gtsrb_acc = self._round_once()

            if not ch_acc and not cm_acc and epc_mnist_acc == 0.0 and epc_cifar10_acc == 0.0 and epc_gtsrb_acc == 0.0:
                skipped_rounds += 1
                print(f"Round skipped. Total skipped: {skipped_rounds}")
                continue

            avg_ch_a = sum(a for a, b in ch_acc.values()) / len(ch_acc) if ch_acc else 0.0
            avg_ch_b = sum(b for a, b in ch_acc.values()) / len(ch_acc) if ch_acc else 0.0
            avg_cm_a = sum(a for a, b in cm_acc.values()) / len(cm_acc) if cm_acc else 0.0
            avg_cm_b = sum(b for a, b in cm_acc.values()) / len(cm_acc) if cm_acc else 0.0

            ch_accuracies.append({"A": avg_ch_a, "B": avg_ch_b})
            cm_accuracies.append({"A": avg_cm_a, "B": avg_cm_b})
            epc_accuracies.append({
                "A": epc_mnist_acc, "B": epc_cifar10_acc, "C": epc_gtsrb_acc
            })

            print(f"  CH: A={avg_ch_a:.4f}, B={avg_ch_b:.4f}")
            print(f"  CM: A={avg_cm_a:.4f}, B={avg_cm_b:.4f}")
            print(f"  EPC: A={epc_mnist_acc:.4f}, B={epc_cifar10_acc:.4f}, C={epc_gtsrb_acc:.4f}")

        print(f"\nSummary: {skipped_rounds} rounds skipped, {len(ch_accuracies)} successful rounds")


        for ch in self.cluster_heads.values():
            ch.shutdown()
        for v in self.vehicles.values():
            v.shutdown()
        self.epc.shutdown()

        return ch_accuracies, cm_accuracies, epc_accuracies
