from dataclasses import dataclass, field, fields, is_dataclass
from typing import Optional, List, Tuple, Union, get_origin, get_args


@dataclass
class BackboneConfig:
    hidden_dim: int = 256
    in_channels: int = 3
    conv1_channels: int = 32
    conv2_channels: int = 64


@dataclass
class DecoderLayerConfig:
    d_model: int = 256
    nhead: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"
    ttt_steps: int = 10
    ttt_lr: float = 0.1
    ttt_momentum: float = 0.8


@dataclass
class DecoderConfig:
    num_layers: int = 3
    num_queries: int = 20


@dataclass
class HeadConfig:
    num_classes: int = 3
    hidden_dim: int = 1024
    sig_dim: int = 64


@dataclass
class ModelConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    decoder_layer: DecoderLayerConfig = field(default_factory=DecoderLayerConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    heads: HeadConfig = field(default_factory=HeadConfig)

    learned_alpha: bool = False
    alpha_focal: float = 1.0
    compact_margin: float = 0.8
    w_proto_ttt: float = 1.0
    spatial_hw: int = 64


@dataclass
class LossConfig:
    w_mask_bce: float = 2.0
    w_mask_iou: float = 5.0
    w_sim: float = 2.0
    w_margin: float = 2.0
    w_inter: float = 10.0
    w_proto_ttt: float = 1.0
    inter_margin: float = 0.0
    topk_per_gt: int = 3


@dataclass
class SeedFilterConfig:
    quality_threshold: float = 0.17
    topk: Optional[int] = None
    min_num_seeds: int = 1
    exclude_background: bool = True
    min_foreground_prob: float = 0.22
    max_influence: Optional[float] = None
    use_foreground_in_score: bool = False
    foreground_score_power: float = 1.0


@dataclass
class ClusterConfig:
    method: str = "cc"   # ["dbscan", "hdbscan", "cc", "louvain", "leiden"]
    cluster_per_class: bool = True
    promote_noise_to_singletons: bool = True

    # DBSCAN
    dbscan_eps: float = 0.15
    dbscan_min_samples: int = 1
    dbscan_use_sample_weight: bool = True

    # HDBSCAN
    hdbscan_min_cluster_size: int = 2
    hdbscan_min_samples: Optional[int] = None
    hdbscan_cluster_selection_epsilon: float = 0.0

    # Graph methods
    graph_affinity_threshold: float = 0.76
    graph_min_edge_weight: float = 0.01

    # Louvain / Leiden
    louvain_resolution: float = 1.0
    leiden_resolution: float = 1.0
    random_seed: int = 42


@dataclass
class SoftAssignmentConfig:
    use_all_queries: bool = True
    refinement_steps: int = 1

    use_alpha_focal: bool = True
    similarity_floor: float = 0.0

    use_layer_weights: bool = True
    use_query_quality: bool = True
    query_quality_power: float = 0.6

    use_foreground_prob: bool = False
    foreground_prob_power: float = 1.0

    class_compat_power: float = 0.0   # 0 disables class-aware assignment
    normalize_over_queries: bool = True


@dataclass
class OverlapResolutionConfig:
    remove_background: bool = True
    min_prototype_score: float = 0.04
    min_area: int = 20

    mask_threshold: float = 0.42
    pixel_score_threshold: float = 0.28

    use_class_confidence: bool = True
    use_foreground_confidence: bool = True
    use_assignment_strength: bool = True
    assignment_strength_power: float = 0.5


@dataclass
class PrototypeInferenceConfig:
    ttt_steps: Optional[int] = None
    seed: SeedFilterConfig = field(default_factory=SeedFilterConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    assign: SoftAssignmentConfig = field(default_factory=SoftAssignmentConfig)
    overlap: OverlapResolutionConfig = field(default_factory=OverlapResolutionConfig)


@dataclass
class PanopticSystemConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    inference: PrototypeInferenceConfig = field(default_factory=PrototypeInferenceConfig)


def _convert_value(tp, value):
    if value is None:
        return None

    origin = get_origin(tp)

    if is_dataclass(tp):
        return dataclass_from_dict(tp, value)

    if origin is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return _convert_value(args[0], value)
        return value

    if origin in (list, List):
        arg = get_args(tp)[0]
        return [_convert_value(arg, x) for x in value]

    if origin in (tuple, Tuple):
        args = get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert_value(args[0], x) for x in value)
        return tuple(_convert_value(a, x) for a, x in zip(args, value))

    return value


def dataclass_from_dict(cls, data: dict):
    kwargs = {}
    for f in fields(cls):
        if f.name in data:
            kwargs[f.name] = _convert_value(f.type, data[f.name])
    return cls(**kwargs)
