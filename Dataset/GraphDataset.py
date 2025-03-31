import numpy as np
import pandas as pd
import torch
import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import pickle
import os

# load bar
from tqdm import tqdm

class GymPoseDataset(DGLDataset):
    """
    Skeleton dataset class
    Args:
        dataset (np.ndarray): 3d keypoints array of shape (frames, points, xyz)
        name (str): Name of the dataset
        path (str): Path to save the processed dataset
    """
    def __init__(self, df : pd.DataFrame, points, name : str, path : str = None):
        self.df = df
        self.points = points
        self.graphs = []
        self.line_graphs = []
        self.targets = []
        self.path = path
        super().__init__(name=name)

    def compute_edge_lengths(self, g : dgl.DGLGraph, norm_lenght) -> dgl.DGLGraph:
        """ 
        Computes the edge lengths of a graph
        Args:
            g (dgl.DGLGraph): Graph of the skeleton
        Returns:
            dgl.DGLGraph: Graph of the skeleton with edge lengths
        """
        # get the node coordinates
        node_coords = g.ndata['coords']
        # get the edge connections
        edge_connections = g.edges()
        # compute the edge lengths
        edge_lengths = node_coords[edge_connections[0]] - node_coords[edge_connections[1]]
        # add the edge lengths to the graph
        #edge_lengths /= norm_lenght
        g.edata['r'] = edge_lengths
        return g

    def build_graph_from_data(self, frames : np.ndarray) -> dgl.DGLGraph:
        """ 
        Builds a graph from a 3d keypoints array
        Args:
            points (np.ndarray): 3d keypoints array of shape (frames, points, xyz)
        Returns:
            dgl.DGLGraph: Graph of the skeleton
        """
        
        edges = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], 
                 [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16], 
                 [6, 3], [5, 2], [5, 11], [2, 14], [6, 11], [3, 14], [0,8]
                 # Extra edges for knees evaluation
                 #[6, 3], [5, 2], [5, 11], [2, 14], [6, 11], [3, 14], [0,8],
                 # Extra edges for elbows evaluation
                 #[8, 12], [8, 13], [8, 15], [8, 16], 
                 #[11, 14], [11, 15], [11, 16],
                 #[12, 14], [12, 15], [12, 16], 
                 #[13, 11], [13, 14], [13, 15], [13,16],
                 #[14,16]
                 ]        
        #edges = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16], [6, 3], [5, 2], [5, 11], [2, 14], [6, 11], [3, 14], [0,8], [13, 11], [14,16], [12, 15]]    
        
        #print(len(edges), frames.shape[1])
        #norm_distance = np.linalg.norm(frames[0][7] - frames[0][8])
        edges = np.array(edges)
        one_hot = np.eye(frames.shape[1])

        #important_nodes = [11, 12, 13, 14, 15, 16]
        #feats = np.array([1 if i in important_nodes else 0 for i in range(17)])
        #node_feats = np.concatenate((one_hot, feats[:, np.newaxis]), axis=1)

        node_coords = []
        node_class = np.repeat(one_hot, frames.shape[0], axis=0)
        #matrix_probabilidades = np.repeat(probabilities, 17, axis=0)
        #node_class*=matrix_probabilidades
        # repeat edges for each frame to get a shape of (frames, edges, 2)
        edge_connections = np.stack([edges] * frames.shape[0], axis=0)
        # print(f"edge connections: {edge_connections.shape}")
        # one class for spatial, one class for temporal, [1, 0] for spatial, [0, 1] for temporal
        edge_class = np.array([[1, 0]] * edges.shape[0] * frames.shape[0])
        
        for frame_idx in range(frames.shape[0]):
            node_coords.append(frames[frame_idx])

        # add the number of previous points to the edge connections so that the graph is connected
        for i in range(1, frames.shape[0]):
            edge_connections[i] += i * frames.shape[1]
            # print(f"edge connections {i}: {edge_connections[i]}")

        edge_connections = np.concatenate(edge_connections, axis=0)
        # append connections to ith node connected to the previous frame's ith node
        for i in range(0, frames.shape[0] - 1):
            # concatenate node 0 to node 17, node 1 to node 18, etc...
            # to_append = [[0, 17], [1, 18], [2, 19], [3, 20], [4, 21], [5, 22], [6, 23], [7, 24], [8, 25], [9, 26], [10, 27], [11, 28], [12, 29], [13, 30], [14, 31], [15, 32], [16, 33]
            to_append = np.array([[i * frames.shape[1] + j, (i + 1) * frames.shape[1] + j] for j in range(frames.shape[1])])
            edge_connections = np.concatenate((edge_connections, to_append), axis=0)
            edge_class = np.concatenate((edge_class, np.array([[0, 1]] * frames.shape[1])), axis=0)
        # flatten the node coords
        node_coords = np.array(node_coords)
        node_coords = node_coords.reshape(-1, node_coords.shape[-1])

        # build the graph
        g = dgl.DGLGraph()
        g.add_nodes(node_coords.shape[0])
        g.add_edges(edge_connections[:, 0], edge_connections[:, 1])

        # add the node features
        g.ndata['coords'] = torch.tensor(node_coords, dtype=torch.float32)
        g.ndata['node_features'] = torch.tensor(node_class, dtype=torch.float32)
        # add the edge features
        g.edata['edge_features'] = torch.tensor(edge_class, dtype=torch.float32)
        # compute the edge lengths
        g = self.compute_edge_lengths(g, norm_lenght=None)
        return g

    def compute_bond_cosines(self, edges):
        """Compute bond angle cosines from bond displacement vectors."""
        # line graph edge: (a, b), (b, c)
        # `a -> b -> c`
        # use law of cosines to compute angles cosines
        # negate src bond so displacements are like `a <- b -> c`
        # cos(theta) = ba \dot bc / (||ba|| ||bc||)
        
        r1 = -edges.src["r"]
        r2 = edges.dst["r"]
        # (x,y,z) : (x,y), (x,z), (y,z)
        bond_cosine_projection = torch.zeros([3, r1.shape[0]], dtype=torch.float32)
        columns = r1.shape[1]
        for i in range(columns):
            for j in range(i, columns):
                pj_r1 = r1[:, i:j+1]
                pj_r2 = r2[:, i:j+1]
        
                bond_cosine = torch.sum(pj_r1 * pj_r2, dim=1) / (
                    torch.norm(pj_r1, dim=1) * torch.norm(pj_r2, dim=1)
                )
                bond_cosine = torch.clamp(bond_cosine, -1, 1)
                bond_cosine = torch.nan_to_num(bond_cosine)
                bond_cosine_projection[i] = bond_cosine

        ##bond_cosine = torch.sum(r1 * r2, dim=1) / (
        ##    torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
        ##)
        ##bond_cosine = torch.clamp(bond_cosine, -1, 1)
        ### replace NaNs with 0s
        ##bond_cosine = torch.nan_to_num(bond_cosine)
        ###bond_cosine = torch.arccos((torch.clamp(bond_cosine, -1, 1)))
        return {"h": bond_cosine_projection.T}
        ##return {"h": bond_cosine}

    def process(self):
        self.graphs = []
        self.targets = []
        print(f"Processing {self.name}...")
        print(f"No. of samples: {len(self.df)}")
        for i in tqdm(range(len(self.df))):
            row = self.df.iloc[i]
            id = row['id']
            #points = self.points[id][0][:,:,:2]
            #probabilities = self.points[id][0][:,:,2]
            points = self.points[id]
            #ee = row['error_knees_forward']
            #ek = row['error_knees_inward']
            #el = row['lumbar_error_label']
            #ea = row['label']
            #es = row['label']
            target = torch.tensor([row["label"]], dtype=torch.float32)
            self.targets.append(target)

            g = self.build_graph_from_data(points)
            self.graphs.append(g)

            lg = g.line_graph(shared=True)
            lg.apply_edges(self.compute_bond_cosines)
            self.line_graphs.append(lg)


    def __getitem__(self, idx):
        return self.graphs[idx], self.line_graphs[idx], self.targets[idx]

    def __len__(self):
        return len(self.graphs)

    def save(self):
        filename = self.name + '.pkl'
        fullpath = os.path.join(self.path, filename)
        # create folder if it does not exist
        if not os.path.exists(self.path):
            print(f"Creating folder {self.path}")
            os.makedirs(self.path)
        else:
            print(f"Folder {self.path} already exists")

        with open(fullpath, 'wb') as f:
            pickle.dump((self.graphs, self.line_graphs, self.targets), f)

    def load(self):
        filename = self.name + '.pkl'
        fullpath = os.path.join(self.path, filename)
        with open(fullpath, 'rb') as f:
            self.graphs, self.line_graphs, self.targets = pickle.load(f)
            print(f"Loaded {self.name} from {fullpath}")
            print(f"Graphs: {len(self.graphs)}")
            print(f"Targets: {len(self.targets)}")

    def has_cache(self):
        filename = self.name + '.pkl'
        try:
            with open(self.path + "/" + filename, 'rb') as f:
                print(f"Found cache for {self.name}")
                return True
        except:
            return False