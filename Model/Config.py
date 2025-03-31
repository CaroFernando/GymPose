from dataclasses import dataclass

@dataclass
class ALIGNNConfig:
    node_input_features: int
    hidden_features: int
    expantion_type: str
    edge_angle_input_features: int
    edge_input_features: int
    embedding_features: int
    triplet_input_features: int
    alignn_layers: int
    gcn_layers: int
    epochs: int
    lr: float
    weight_decay: float

BaseConfig = ALIGNNConfig(
    node_input_features=17,
    hidden_features=128,
    expantion_type='rbfe',
    edge_angle_input_features=40,
    edge_input_features=2,
    embedding_features=256,
    triplet_input_features=40,
    alignn_layers=4,
    gcn_layers=4,
    epochs=200,
    lr = 1e-4,
    weight_decay = 9e-2 #for elbows and 3d projection
    #weight_decay = 9e-2 #for knees and 3d projection
)

BaseConfigMLP = ALIGNNConfig(
    node_input_features=17,
    hidden_features=128,
    expantion_type='mlp',
    edge_angle_input_features=40,
    edge_input_features=2,
    embedding_features=256,
    triplet_input_features=40,
    alignn_layers=4,
    gcn_layers=4,
    epochs=200,
    lr = 1e-4,
    weight_decay = 9e-2 #for elbows and 3d projection
    #weight_decay = 9e-2 #for knees and 3d projection
)

SmallConfig = ALIGNNConfig(
    node_input_features=17,
    hidden_features=64,
    expantion_type='rbfe',
    edge_angle_input_features=40,
    edge_input_features=2,
    embedding_features=128,
    triplet_input_features=40,
    alignn_layers=2,
    gcn_layers=2,
    epochs=200,
    lr = 1e-4,
    weight_decay = 9e-2 #for elbows and 3d projection
    #weight_decay = 9e-2 #for knees and 3d projection
)

SmallConfigMLP = ALIGNNConfig(
    node_input_features=17,
    hidden_features=64,
    expantion_type='mlp',
    edge_angle_input_features=40,
    edge_input_features=2,
    embedding_features=128,
    triplet_input_features=40,
    alignn_layers=2,
    gcn_layers=2,
    epochs=200,
    lr = 1e-4,
    weight_decay = 9e-2 #for elbows and 3d projection
    #weight_decay = 9e-2 #for knees and 3d projection
)