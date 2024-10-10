import os
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter_add
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset



class GraphDataset(Dataset):
    def __init__(self, home_dir, sub_dir, preprocess=True, hparams=None):
        self.base_path = os.path.join(home_dir, sub_dir)
        self.file_names = self._get_file_names()
        self.preprocess = preprocess
        self.hparams = hparams if hparams is not None else {}

    def _get_file_names(self):
        file_names = []
        for file_name in os.listdir(self.base_path):
            if file_name.endswith('.pyg'):  # Adjust this condition as needed
                file_names.append(file_name)
        return file_names

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = os.path.join(self.base_path, file_name)
        data = torch.load(file_path)
        print(f"Loaded data from {file_path}")

        # Remove 'scores' from data if it exists
        if 'scores' in data:
            del data['scores']

        
        if self.preprocess:
            edge_features = self.hparams["edge_features"]
            if "dr" in edge_features and not ("dr" in data.keys()):
                src, dst = data['edge_index']
                data.dr = data.r[dst] - data.r[src]
        
        nodefeatures = torch.stack([data[feature] for feature in self.hparams["node_features"]], dim=-1).float()
        mask = torch.logical_or(data['region'] == 2, data['region'] == 6).reshape(-1)
        nodefeatures[mask] = torch.cat([nodefeatures[mask, 0:4], nodefeatures[mask, 0:4], nodefeatures[mask, 0:4]], dim=1) #Why?
        if len(self.hparams.get("edge_features",[]))>0:
            edgefeatures = torch.stack([data[feature] for feature in self.hparams["edge_features"]], dim=-1).float()
        else:
            edgefeatures = None
        edge_index = data['edge_index']
        return nodefeatures, edgefeatures, edge_index[0], edge_index[1]

    def __len__(self):
        return len(self.file_names)

def make_mlp(
    input_size,
    sizes,
    hidden_activation="ReLU",
    output_activation=None,
    layer_norm=False,  # TODO : change name to hidden_layer_norm while ensuring backward compatibility
    output_layer_norm=False,
    batch_norm=False,  # TODO : change name to hidden_batch_norm while ensuring backward compatibility
    output_batch_norm=False,
    input_dropout=0,
    hidden_dropout=0,
    track_running_stats=False,
):
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # Hidden layers
    for i in range(n_layers - 1):
        if i == 0 and input_dropout > 0:
            layers.append(nn.Dropout(input_dropout))
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:  # hidden_layer_norm
            layers.append(nn.LayerNorm(sizes[i + 1], elementwise_affine=False))
        if batch_norm:  # hidden_batch_norm
            layers.append(
                nn.BatchNorm1d(
                    sizes[i + 1],
                    eps=6e-05,
                    track_running_stats=track_running_stats,
                    affine=True,
                )  # TODO : Set BatchNorm and LayerNorm parameters in config file ?
            )
        layers.append(hidden_activation())
        if hidden_dropout > 0:
            layers.append(nn.Dropout(hidden_dropout))
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if output_layer_norm:
            layers.append(nn.LayerNorm(sizes[-1], elementwise_affine=False))
        if output_batch_norm:
            layers.append(
                nn.BatchNorm1d(
                    sizes[-1],
                    eps=6e-05,
                    track_running_stats=track_running_stats,
                    affine=True,
                )  # TODO : Set BatchNorm and LayerNorm parameters in config file ?
            )
        layers.append(output_activation())
    return nn.Sequential(*layers)



class InteractionGNN2(LightningModule):
    """
    Interaction Network (L2IT version).
    Operates on directed graphs.
    Aggregate and reduce (sum) separately incomming and outcoming edges latents.
    """

    def __init__(self, hparams:dict, train_dataset=None, val_dataset=None, test_dataset=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        hparams["batchnorm"] = (
            False if "batchnorm" not in hparams else hparams["batchnorm"]
        )
        hparams["output_batch_norm"] = hparams.get("output_batch_norm", False)
        hparams["edge_output_transform_final_batch_norm"] = hparams.get(
            "edge_output_transform_final_batch_norm", False
        )
        hparams["edge_output_transform_final_batch_norm"] = hparams.get(
            "edge_output_transform_final_batch_norm", False
        )
        hparams["track_running_stats"] = (
            False
            if "track_running_stats" not in hparams
            else hparams["track_running_stats"]
        )

        # TODO: Add equivalent check and default values for other model parameters ?
        # TODO: Use get() method

        # Define the dataset to be used, if not using the default
        #self.save_hyperparameters(hparams)

        # self.setup_layer_sizes()

        if hparams["concat"]:
            if hparams["in_out_diff_agg"]:
                in_node_net = hparams["hidden"] * 4
            else:
                in_node_net = hparams["hidden"] * 3
            in_edge_net = hparams["hidden"] * 6
        else:
            if hparams["in_out_diff_agg"]:
                in_node_net = hparams["hidden"] * 3
            else:
                in_node_net = hparams["hidden"] * 2
            in_edge_net = hparams["hidden"] * 3
        # node encoder
        self.node_encoder = make_mlp(
            input_size=len(hparams["node_features"]),
            sizes=[hparams["hidden"]] * hparams["n_node_net_layers"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_batch_norm=hparams["output_batch_norm"],
            track_running_stats=hparams["track_running_stats"],
        )
     
        # edge encoder
        if "edge_features" in hparams and len(hparams["edge_features"]) != 0:
            self.edge_encoder = make_mlp(
                input_size=len(hparams["edge_features"]),
                sizes=[hparams["hidden"]] * hparams["n_edge_net_layers"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                output_batch_norm=hparams["output_batch_norm"],
                track_running_stats=hparams["track_running_stats"],
            )
        else:
            self.edge_encoder = make_mlp(
                input_size=2 * hparams["hidden"],
                sizes=[hparams["hidden"]] * hparams["n_edge_net_layers"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                output_batch_norm=hparams["output_batch_norm"],
                track_running_stats=hparams["track_running_stats"],
            )

        # edge network
        
        self.edge_network = make_mlp(
            input_size=in_edge_net,
            sizes=[hparams["hidden"]] * hparams["n_edge_net_layers"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_batch_norm=hparams["output_batch_norm"],
            track_running_stats=hparams["track_running_stats"],
        )
        # node network
        
        self.node_network = make_mlp(
            input_size=in_node_net,
            sizes=[hparams["hidden"]] * hparams["n_node_net_layers"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_batch_norm=hparams["output_batch_norm"],
            track_running_stats=hparams["track_running_stats"],
        )
        

        # edge decoder
        self.edge_decoder = make_mlp(
            input_size=hparams["hidden"],
            sizes=[hparams["hidden"]] * hparams["n_edge_decoder_layers"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_batch_norm=hparams["output_batch_norm"],
            track_running_stats=hparams["track_running_stats"],
        )
        # edge output transform layer
        self.edge_output_transform = make_mlp(
            input_size=hparams["hidden"],
            sizes=[hparams["hidden"], 1],
            output_activation=hparams["edge_output_transform_final_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_batch_norm=hparams["edge_output_transform_final_batch_norm"],
            track_running_stats=hparams["track_running_stats"],
        )

        # dropout layer
        self.dropout = nn.Dropout(p=0.1)
        # hyperparams

        self.node_features = hparams.get("node_features", [])
        self.edge_features = hparams.get("edge_features", [])
        self.checkpointing = hparams.get("checkpointing", True)
        self.isconcat = hparams.get("concat", True)
        self.n_graph_iters = hparams.get("n_graph_iters", 1)
        self.node_net_recurrent = hparams.get("node_net_recurrent", False)
        self.edge_net_recurrent = hparams.get("edge_net_recurrent", False)
        self.in_out_diff_agg = hparams.get("in_out_diff_agg", False)



    ###############################

    def training_step(self, batch, batch_idx):
        # Forward pass
        output = self(batch)
        
        # Dummy loss calculation
        loss = output.mean()
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)

        return loss
        
    def validation_step(self, batch, batch_idx):
        # Forward pass
        output = self(batch)
        
        # Dummy loss calculation
        loss = output.mean()
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        return loss
        
    def testing_step(self, batch, batch_idx):
        # Forward pass
        output = self(batch)
        
        # Dummy loss calculation
        loss = output.mean()

        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.get("lr", 0.001))
        return optimizer


    def train_dataloader(self):
        """
        Load the training set.
        """
        return DataLoader(self.train_dataset, batch_size=1, shuffle=True, num_workers=4)

    
    def val_dataloader(self):
        """
        Load the val set.
        """
        return DataLoader(self.val_dataset, batch_size=1, shuffle=True, num_workers=4)

    
    def test_dataloader(self):
        """
        Load the test set.
        """
        return DataLoader(self.test_dataset, batch_size=1, shuffle=True, num_workers=4)
    ###########################


    def forward(self, x, edge_attr, src:torch.Tensor, dst:torch.Tensor):
        # x = torch.stack(
        #     [batch[feature] for feature in self.node_features], dim=-1
        # ).float()

        # Same features on the 3 channels in the STRIP ENDCAP TODO: Process it in previous stage
        # mask = torch.logical_or(batch.region == 2, batch.region == 6).reshape(-1)
        # x[mask] = torch.cat([x[mask, 0:4], x[mask, 0:4], x[mask, 0:4]], dim=1)
        

        # if len(self.edge_features) > 0:
        #     edge_attr = torch.stack(
        #         [batch[feature] for feature in self.edge_features], dim=-1
        #     ).float()
        # else:
        #     edge_attr = None

        # x = x.detach().requires_grad_(True)
        # if edge_attr is not None:
        #     edge_attr = edge_attr.detach().requires_grad_(True)

        # Get src and dst
        # src, dst = batch.edge_index
        

####################
         # Call handle_edge_features function to handle edge features
        #handle_edge_features(batch, self.hparams.get("edge_features", []))
####################

        # Encode nodes and edges features into latent spaces
        x = self.node_encoder(x)
        print('node_embed')
        if edge_attr is not None:
            e = self.edge_encoder(edge_attr)
        else:
            e = self.edge_encoder(torch.cat([x[src], x[dst]], dim=-1))
        print('edge_embed')
        # Apply dropout
        # x = self.dropout(x)
        # e = self.dropout(e)

        # memorize initial encodings for concatenate in the gnn loop if request
        input_x = x
        input_e = e 

        # Loop over gnn layers
        for i in range(self.n_graph_iters):
            if self.isconcat:
                x = torch.cat([x, input_x], dim=-1)
                e = torch.cat([e, input_e], dim=-1)
            # x, e, out = self.message_step(x, e, src, dst, i)
            edge_inputs = torch.cat([e, x[src], x[dst]], dim=-1)  # order dst src x ?
            e_updated = self.edge_network(edge_inputs)
            # # Update nodes
            # edge_messages_from_src = scatter_add(e_updated, dst, dim=0, dim_size=x.shape[0])
            # edge_messages_from_dst = scatter_add(e_updated, src, dim=0, dim_size=x.shape[0])
            edge_messages_from_src = torch.zeros(x.shape[0], e_updated.shape[1], dtype=e_updated.dtype, device=e_updated.device)
            edge_messages_from_dst = torch.zeros(x.shape[0], e_updated.shape[1], dtype=e_updated.dtype, device=e_updated.device)
            edge_messages_from_src.scatter_add_(dim=0, index=dst.unsqueeze(-1).expand(-1, e_updated.shape[1]), src=e_updated)
            edge_messages_from_dst.scatter_add_(dim=0, index=src.unsqueeze(-1).expand(-1, e_updated.shape[1]), src=e_updated)
            if self.in_out_diff_agg:
                node_inputs = torch.cat(
                    [edge_messages_from_src, edge_messages_from_dst, x], dim=-1
                )  # to check : the order dst src  x ?
            else:
                edge_messages = edge_messages_from_src + edge_messages_from_dst
                node_inputs = torch.cat([edge_messages, x], dim=-1)
            
            x_updated = self.node_network(node_inputs)
            x = x_updated
            e = e_updated
        
        out = self.edge_output_transform(self.edge_decoder(e))
        return out.squeeze(-1)
    
    

    def concat(self, x, y):
        return torch.cat([x, y], dim=-1)
    