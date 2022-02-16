import torch
import numpy as np
from torch import nn
import math


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        # Alternative:
        # headst = heads.transpose(0, 1)  # swap the dimensions for batch and heads to align it for the matmul
        # # proj_h = torch.einsum('bhni,hij->bhnj', headst, self.W_out)
        # projected_heads = torch.matmul(headst, self.W_out)
        # out = torch.sum(projected_heads, dim=1)  # sum across heads

        # Or:
        # out = torch.einsum('hbni,hij->bnj', heads, self.W_out)

        return out


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )


class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    def forward(self, x, mask=None):

        assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        h = self.layers(h)

        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )

class CCN3(nn.Module):

    def __init__(
            self,
            node_dim = 2,
            embed_dim = 128,
            n_layers = 2,
            n_neighbors = 6
    ):
        super(CCN3, self).__init__()
        self.init_embed = nn.Linear(node_dim, embed_dim)
        self.n_neighbors = n_neighbors
        self.init_neighbour_embed = nn.Linear(node_dim, embed_dim)
        self.neighbour_encode = nn.Linear(embed_dim, embed_dim)
        self.neighbour_encode_2 = nn.Linear(embed_dim, embed_dim)
        self.init_embed_depot = nn.Linear(2, embed_dim)
        self.final_embedding = nn.Linear(embed_dim, embed_dim)
        self.final_embedding_2 = nn.Linear(embed_dim, embed_dim)
        self.activ = nn.LeakyReLU()

    def forward(self, X, mask=None):
        x = torch.cat((X['loc'], X['demand'][:, :, None]), 2)
        # x2 = x[:, :, 0:3]

        # F0_embedding_2d = self.init_embed_2d(x2)
        F0_embedding_3d = self.init_embed(x)
        # F0_embedding.reshape([1])

        dist_mat = (x[:, None] - x[:, :, None]).norm(dim=-1, p=2)  ## device to cuda to be added
        neighbors = dist_mat.sort().indices[:, :, :self.n_neighbors]  # for 6 neighbours
        # neighbour = x[:, neighbors][0]
        # neighbour_delta = neighbour - x[:, :, None, :]
        neighbour_delta_embedding = self.activ(self.neighbour_encode(F0_embedding_3d[:, neighbors][0] - F0_embedding_3d[:, :, None, :]))
        concat = torch.cat((F0_embedding_3d[:, :, None, :], neighbour_delta_embedding), 2)
        F_embed_final = self.final_embedding(concat).sum(2)

        neighbour_delta_embedding_2 = self.activ(
            self.neighbour_encode_2(F_embed_final[:, neighbors][0] - F_embed_final[:, :, None, :]))
        concat_2 = torch.cat((F_embed_final[:, :, None, :], neighbour_delta_embedding_2), 2)
        F_embed_final_2 = self.final_embedding_2(concat_2).sum(2)
        #h2_neighbor = F_embed_final[:, neighbors][0]
        #F_embed_final_2 = self.neighbour_encode_2(h2_neighbor).sum(dim=2)
        init_depot_embed = self.init_embed_depot(X['depot'])
        h = self.activ(torch.cat((init_depot_embed[:,None], F_embed_final_2), -2))
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )

class CCN(nn.Module):

    def __init__(
            self,
            node_dim = 2,
            embed_dim = 128,
            n_layers = 2,
    ):
        super(CCN, self).__init__()
        self.init_embed = nn.Linear(node_dim, embed_dim)
        self.init_neighbour_embed = nn.Linear(node_dim, embed_dim)
        self.neighbour_encode = nn.Linear(node_dim, embed_dim)
        self.neighbour_encode_2 = nn.Linear(embed_dim, embed_dim)
        self.init_embed_depot = nn.Linear(2, embed_dim)
        self.final_embedding = nn.Linear(embed_dim, embed_dim)

        self.test_layer_1 = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LeakyReLU())
        self.test_layer_2 = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LeakyReLU())

    def forward(self, X, mask=None):
        x = torch.cat((X['loc'], X['demand'][:, :, None]), 2)
        x2 = x[:, :, 0:2]
        activ = nn.LeakyReLU()
        # F0_embedding_2d = self.init_embed_2d(x2)
        F0_embedding_3d = self.init_embed(x)
        # F0_embedding.reshape([1])

        dist_mat = (x2[:, None] - x2[:, :, None]).norm(dim=-1, p=2)  ## device to cuda to be added
        neighbors = dist_mat.sort().indices[:, :, :10]  # for 6 neighbours
        neighbour = x[:, neighbors][0]
        neighbour_delta = neighbour - x[:, :, None, :]
        neighbour_delta_embedding = self.neighbour_encode(neighbour_delta)
        concat = torch.cat((F0_embedding_3d[:, :, None, :], neighbour_delta_embedding), 2)

        F_embed_final = self.test_layer_1(concat).sum(dim=2)
        h2_neighbor = F_embed_final[:, neighbors][0]
        F_embed_final_2 = self.test_layer_2(h2_neighbor).sum(dim=2)
        init_depot_embed = self.init_embed_depot(X['depot'])
        h = activ(torch.cat((init_depot_embed[:,None], F_embed_final_2), -2))
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )


class GCN_K1_L_1(nn.Module):

    def __init__(self,
                 n_layers = 3,
                 n_dim = 128,
                 n_K = 1,
                 node_dim = 3
                 ):
        super(GCN_K1_L_1, self).__init__()
        self.n_layers = n_layers
        self.n_dim = n_dim
        self.n_K = n_K
        self.node_dim = node_dim
        self.init_embed = nn.Linear(node_dim, n_dim)
        self.init_embed_depot = nn.Linear(2, n_dim)
        self.W1 = nn.Linear(n_dim*(n_K+1), n_dim)
        self.W2 = nn.Linear(n_dim * (n_K+1), n_dim)

        self.normalization_1 = Normalization(n_dim)

        self.activ = nn.LeakyReLU()

    def forward(self, data, mask=None):
        X = torch.cat((data['loc'], data['demand'][:, :, None]), 2)
        X = torch.cat((X[:, :, 0:2], (X[:, :, 2] / X[:, :, 2].max())[:, :, None]), -1)
        X_loc = X
        distance_matrix = (((X_loc[:, :, None] - X_loc[:, None]) ** 2).sum(-1)) ** .5
        num_samples, num_locations, _ = X.size()
        A = ((1 / distance_matrix) * (torch.eye(num_locations, device=distance_matrix.device).expand(
            (num_samples, num_locations, num_locations)) - 1).to(torch.bool).to(torch.float))
        A[A != A] = 0
        A = A / A.max()
        D = torch.mul(
            torch.eye(num_locations, device=distance_matrix.device).expand((num_samples, num_locations, num_locations)),
            (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))
        L = D - A
        F0 = self.init_embed(X)

        g1 = torch.cat((F0[:,:,:,None],
                        torch.matmul(L,F0)[:,:,:,None]
                        ),
                       -1).reshape((num_samples, num_locations, -1))

        F1 =   self.activ(self.W1(g1)) + F0
        F1 = self.normalization_1(F1)

        init_depot_embed = self.init_embed_depot(data['depot'])[:, None]
        h = torch.cat((init_depot_embed, F1), 1)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )



class GCN_K2_L_1(nn.Module):

    def __init__(self,
                 n_layers = 3,
                 n_dim = 128,
                 n_K = 2,
                 node_dim = 3
                 ):
        super(GCN_K2_L_1, self).__init__()
        self.n_layers = n_layers
        self.n_dim = n_dim
        self.n_K = n_K
        self.node_dim = node_dim
        self.init_embed = nn.Linear(node_dim, n_dim)
        self.init_embed_depot = nn.Linear(2, n_dim)
        self.W1 = nn.Linear(n_dim*(n_K+1), n_dim)
        self.W2 = nn.Linear(n_dim * (n_K+1), n_dim)

        self.normalization_1 = Normalization(n_dim)

        self.activ = nn.LeakyReLU()

    def forward(self, data, mask=None):
        X = torch.cat((data['loc'], data['demand'][:, :, None]), 2)
        X = torch.cat((X[:, :, 0:2], (X[:, :, 2] / X[:, :, 2].max())[:, :, None]), -1)
        X_loc = X
        distance_matrix = (((X_loc[:, :, None] - X_loc[:, None]) ** 2).sum(-1)) ** .5
        num_samples, num_locations, _ = X.size()
        A = ((1 / distance_matrix) * (torch.eye(num_locations, device=distance_matrix.device).expand(
            (num_samples, num_locations, num_locations)) - 1).to(torch.bool).to(torch.float))
        A[A != A] = 0
        A = A / A.max()
        D = torch.mul(
            torch.eye(num_locations, device=distance_matrix.device).expand((num_samples, num_locations, num_locations)),
            (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))
        L = D - A
        L_squared = torch.matmul(L, L)
        F0 = self.init_embed(X)

        g1 = torch.cat((F0[:,:,:,None],
                        torch.matmul(L,F0)[:,:,:,None],
                        torch.matmul(L_squared, F0)[:,:,:,None]
                        ),
                       -1).reshape((num_samples, num_locations, -1))

        F1 =   self.activ(self.W1(g1)) + F0
        F1 = self.normalization_1(F1)

        init_depot_embed = self.init_embed_depot(data['depot'])[:, None]
        h = torch.cat((init_depot_embed, F1), 1)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )



class GCN_K3_L_1(nn.Module):

    def __init__(self,
                 n_layers = 3,
                 n_dim = 128,
                 n_K = 3,
                 node_dim = 3
                 ):
        super(GCN_K3_L_1, self).__init__()
        self.n_layers = n_layers
        self.n_dim = n_dim
        self.n_K = n_K
        self.node_dim = node_dim
        self.init_embed = nn.Linear(node_dim, n_dim)
        self.init_embed_depot = nn.Linear(2, n_dim)
        self.W1 = nn.Linear(n_dim*(n_K+1), n_dim)
        self.W2 = nn.Linear(n_dim * (n_K+1), n_dim)

        self.normalization_1 = Normalization(n_dim)

        self.activ = nn.LeakyReLU()

    def forward(self, data, mask=None):
        X = torch.cat((data['loc'], data['demand'][:, :, None]), 2)
        X = torch.cat((X[:, :, 0:2], (X[:, :, 2] / X[:, :, 2].max())[:, :, None]), -1)
        X_loc = X
        distance_matrix = (((X_loc[:, :, None] - X_loc[:, None]) ** 2).sum(-1)) ** .5
        num_samples, num_locations, _ = X.size()
        A = ((1 / distance_matrix) * (torch.eye(num_locations, device=distance_matrix.device).expand(
            (num_samples, num_locations, num_locations)) - 1).to(torch.bool).to(torch.float))
        A[A != A] = 0
        A = A / A.max()
        D = torch.mul(
            torch.eye(num_locations, device=distance_matrix.device).expand((num_samples, num_locations, num_locations)),
            (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))
        L = D - A
        L_squared = torch.matmul(L, L)
        L_cube = torch.matmul(L, L_squared)
        F0 = self.init_embed(X)

        g1 = torch.cat((F0[:,:,:,None],
                        torch.matmul(L,F0)[:,:,:,None],
                        torch.matmul(L_squared, F0)[:,:,:,None],
                        torch.matmul(L_cube, F0)[:, :, :, None]
                        ),
                       -1).reshape((num_samples, num_locations, -1))

        F1 =   self.activ(self.W1(g1)) + F0
        F1 = self.normalization_1(F1)

        init_depot_embed = self.init_embed_depot(data['depot'])[:, None]
        h = torch.cat((init_depot_embed, F1), 1)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )


class GCN_K3_L_2(nn.Module):

    def __init__(self,
                 n_layers = 3,
                 n_dim = 128,
                 n_K = 3,
                 node_dim = 3
                 ):
        super(GCN_K3_L_2, self).__init__()
        self.n_layers = n_layers
        self.n_dim = n_dim
        self.n_K = n_K
        self.node_dim = node_dim
        self.init_embed = nn.Linear(node_dim, n_dim)
        self.init_embed_depot = nn.Linear(2, n_dim)
        self.W1 = nn.Linear(n_dim*(n_K+1), n_dim)
        self.W2 = nn.Linear(n_dim * (n_K+1), n_dim)

        self.normalization_1 = Normalization(n_dim)
        self.normalization_2 = Normalization(n_dim)

        self.activ = nn.LeakyReLU()

    def forward(self, data, mask=None):
        X = torch.cat((data['loc'], data['demand'][:, :, None]), 2)
        X = torch.cat((X[:, :, 0:2], (X[:, :, 2] / X[:, :, 2].max())[:, :, None]), -1)
        X_loc = X
        distance_matrix = (((X_loc[:, :, None] - X_loc[:, None]) ** 2).sum(-1)) ** .5
        num_samples, num_locations, _ = X.size()
        A = ((1 / distance_matrix) * (torch.eye(num_locations, device=distance_matrix.device).expand(
            (num_samples, num_locations, num_locations)) - 1).to(torch.bool).to(torch.float))
        A[A != A] = 0
        A = A / A.max()
        D = torch.mul(
            torch.eye(num_locations, device=distance_matrix.device).expand((num_samples, num_locations, num_locations)),
            (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))
        L = D - A
        L_squared = torch.matmul(L, L)
        L_cube = torch.matmul(L, L_squared)
        F0 = self.init_embed(X)

        g1 = torch.cat((F0[:,:,:,None],
                        torch.matmul(L,F0)[:,:,:,None],
                        torch.matmul(L_squared, F0)[:,:,:,None],
                        torch.matmul(L_cube, F0)[:, :, :, None]
                        ),
                       -1).reshape((num_samples, num_locations, -1))

        F1 =   self.activ(self.W1(g1)) + F0
        F1 = self.normalization_1(F1)
        g2 = torch.cat((F1[:,:,:,None],
                        torch.matmul(L,F1)[:,:,:,None],
                        torch.matmul(L_squared, F1)[:,:,:,None],
                        torch.matmul(L_cube, F1)[:, :, :, None]
                        ),
                       -1).reshape((num_samples, num_locations, -1))
        F2 = self.activ(self.W2(g2)) + F1
        F2 = self.normalization_2(F2)

        init_depot_embed = self.init_embed_depot(data['depot'])[:, None]
        h = torch.cat((init_depot_embed, F2), 1)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )


class GCN_K3_L_3(nn.Module):

    def __init__(self,
                 n_layers = 3,
                 n_dim = 128,
                 n_K = 3,
                 node_dim = 3
                 ):
        super(GCN_K3_L_3, self).__init__()
        self.n_layers = n_layers
        self.n_dim = n_dim
        self.n_K = n_K
        self.node_dim = node_dim
        self.init_embed = nn.Linear(node_dim, n_dim)
        self.init_embed_depot = nn.Linear(2, n_dim)
        self.W1 = nn.Linear(n_dim*(n_K+1), n_dim)
        self.W2 = nn.Linear(n_dim * (n_K+1), n_dim)
        self.W3 = nn.Linear(n_dim * (n_K + 1), n_dim)

        self.normalization_1 = Normalization(n_dim)
        self.normalization_2 = Normalization(n_dim)
        self.normalization_3 = Normalization(n_dim)

        self.activ = nn.LeakyReLU()

    def forward(self, data, mask=None):
        X = torch.cat((data['loc'], data['demand'][:, :, None]), 2)
        X = torch.cat((X[:, :, 0:2], (X[:, :, 2] / X[:, :, 2].max())[:, :, None]), -1)
        X_loc = X
        distance_matrix = (((X_loc[:, :, None] - X_loc[:, None]) ** 2).sum(-1)) ** .5
        num_samples, num_locations, _ = X.size()
        A = ((1 / distance_matrix) * (torch.eye(num_locations, device=distance_matrix.device).expand(
            (num_samples, num_locations, num_locations)) - 1).to(torch.bool).to(torch.float))
        A[A != A] = 0
        A = A / A.max()
        D = torch.mul(
            torch.eye(num_locations, device=distance_matrix.device).expand((num_samples, num_locations, num_locations)),
            (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))
        L = D - A
        L_squared = torch.matmul(L, L)
        L_cube = torch.matmul(L, L_squared)
        F0 = self.init_embed(X)

        g1 = torch.cat((F0[:,:,:,None],
                        torch.matmul(L,F0)[:,:,:,None],
                        torch.matmul(L_squared, F0)[:,:,:,None],
                        torch.matmul(L_cube, F0)[:, :, :, None]
                        ),
                       -1).reshape((num_samples, num_locations, -1))

        F1 =   self.activ(self.W1(g1)) + F0
        F1 = self.normalization_1(F1)
        g2 = torch.cat((F1[:,:,:,None],
                        torch.matmul(L,F1)[:,:,:,None],
                        torch.matmul(L_squared, F1)[:,:,:,None],
                        torch.matmul(L_cube, F1)[:, :, :, None]
                        ),
                       -1).reshape((num_samples, num_locations, -1))
        F2 = self.activ(self.W2(g2)) + F1
        F2 = self.normalization_2(F2)

        g3 = torch.cat((F2[:, :, :, None],
                        torch.matmul(L, F2)[:, :, :, None],
                        torch.matmul(L_squared, F2)[:, :, :, None],
                        torch.matmul(L_cube, F2)[:, :, :, None]
                        ),
                       -1).reshape((num_samples, num_locations, -1))
        F3 = self.activ(self.W2(g3)) + F2
        F3 = self.normalization_3(F3)

        init_depot_embed = self.init_embed_depot(data['depot'])[:, None]
        h = torch.cat((init_depot_embed, F3), 1)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )


class GCAPCN(nn.Module):

    def __init__(self,
                 n_layers = 3,
                 n_hops = 5,
                 n_dim = 128,
                 n_p = 3,
                 node_dim = 3
                 ):
        super(GCAPCN, self).__init__()
        self.n_layers = n_layers
        self.n_hops = n_hops
        self.n_dim = n_dim
        self.n_p = n_p
        self.node_dim = node_dim
        self.init_embed = nn.Linear(node_dim, n_dim)
        self.W1 = nn.Linear(n_dim*n_p, n_dim)
        self.W2 = nn.Linear(n_dim * n_p, n_dim)
        self.activ = nn.LeakyReLU()
        self.normalization_1 = nn.BatchNorm1d(n_dim * n_p)
        self.normalization_2 = nn.BatchNorm1d(n_dim)
        self.normalization_3 = nn.BatchNorm1d(n_dim * n_p)
        self.normalization_4 = nn.BatchNorm1d(n_dim)
        self.init_embed_depot = nn.Linear(2, n_dim)

    def forward(self, data, mask=None):
        X = torch.cat((data['loc'], data['demand'][:, :, None]), 2)
        # X = torch.cat((data['loc'], data['deadline']), -1)
        X_loc = X[:, :, 0:2]
        distance_matrix = (((X_loc[:, :, None] - X_loc[:, None]) ** 2).sum(-1)) ** .5
        num_samples, num_locations, _ = X.size()
        A = ((1 / distance_matrix) * (torch.eye(num_locations, device=distance_matrix.device).expand(
            (num_samples, num_locations, num_locations)) - 1).to(torch.bool).to(torch.float))
        A[A != A] = 0
        D = torch.mul(torch.eye(num_locations, device=distance_matrix.device).expand((num_samples, num_locations, num_locations)),
                      (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))
        L = D - A
        F0 = self.init_embed(X)

        L_squared = torch.matmul(L, L)
        L_cube = torch.matmul(L, L_squared)
        g1_1 = torch.cat((F0[:, :, :, None],
                          torch.matmul(L, F0)[:, :, :, None],
                          torch.matmul(L_squared, F0)[:, :, :, None],
                          torch.matmul(L_cube, F0)[:, :, :, None]
                          ),
                         -1).sum(-1)
        F0_squared = torch.mul(F0[:, :, :], F0[:, :, :])
        F0_cube = torch.mul(F0[:, :, :], F0_squared[:, :, :])
        g1_2 = torch.cat((F0_squared[:, :, :, None],
                          torch.matmul(L, F0_squared)[:, :, :, None],
                          torch.matmul(L_squared, F0_squared)[:, :, :, None],
                          torch.matmul(L_cube, F0_squared)[:, :, :, None]
                          ),
                         -1).sum(-1)
        g1_3 = torch.cat((F0_cube[:, :, :, None],
                          torch.matmul(L, F0_cube)[:, :, :, None],
                          torch.matmul(L_squared, F0_cube)[:, :, :, None],
                          torch.matmul(L_cube, F0_cube)[:, :, :, None]
                          ),
                         -1).sum(-1)
        g1_cat = torch.cat((g1_1, g1_2, g1_3), -1)
        inter_1 = self.normalization_1(g1_cat.view(-1, g1_cat.size(-1))).view(*g1_cat.size())
        F1 =   self.activ(self.W1(inter_1)) + F0 # F0 for skip connection
        F1_norm = self.normalization_2(F1)

        F1_norm_squared = torch.mul(F1_norm[:, :, :], F1_norm[:, :, :])
        F1_norm_cube = torch.mul(F1_norm[:, :, :], F1_norm_squared[:, :, :])
        g2_1 = torch.cat((F1_norm[:, :, :, None],
                          torch.matmul(L, F1_norm)[:, :, :, None],
                          torch.matmul(L_squared, F1_norm)[:, :, :, None],
                          torch.matmul(L_cube, F1_norm)[:, :, :, None]
                          ),
                         -1).sum(-1)

        g2_2 = torch.cat((F1_norm_squared[:, :, :, None],
                          torch.matmul(L, F1_norm_squared)[:, :, :, None],
                          torch.matmul(L_squared, F1_norm_squared)[:, :, :, None],
                          torch.matmul(L_cube, F1_norm_squared)[:, :, :, None]
                          ),
                         -1).sum(-1)

        g2_3 = torch.cat((F1_norm_squared[:, :, :, None],
                          torch.matmul(L, F1_norm_cube)[:, :, :, None],
                          torch.matmul(L_squared, F1_norm_cube)[:, :, :, None],
                          torch.matmul(L_cube, F1_norm_cube)[:, :, :, None]
                          ),
                         -1).sum(-1)

        g2_cat = torch.cat((g2_1, g2_2, g2_3), -1)

        inter_2 = self.normalization_3(g2_cat.view(-1, g2_cat.size(-1))).view(*g2_cat.size())

        F2 = self.activ(self.W2(inter_2)) + F1 # F11 for skip connection

        F2_norm = self.normalization_3(F2)

        init_depot_embed = self.init_embed_depot(data['depot'])[:, None]
        h = torch.cat((init_depot_embed, F2_norm), 1)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )


def batch_norm(F):
    return ((F.view(-1, F.size(-1)) - F.view(-1, F.size(-1)).mean(-2)) / (
                F.view(-1, F.size(-1)).std(-2) + torch.tensor([0.00001], device=F.device))).view(*F.size())

class GCAPCN_K_1_P_1_L_1(nn.Module):

    def __init__(self,
                 n_layers=2,
                 n_dim=128,
                 n_p=1,
                 node_dim=3,
                 n_K=1
                 ):
        super(GCAPCN_K_1_P_1_L_1, self).__init__()
        self.n_layers = n_layers
        self.n_dim = n_dim/n_p
        self.n_p = n_p
        self.n_K = n_K
        self.node_dim = node_dim
        self.init_embed = nn.Linear(node_dim, n_dim * n_p)
        self.init_embed_depot = nn.Linear(2, n_dim)

        self.W_L_1_G1 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)

        self.normalization_1 = Normalization(n_dim * n_p)

        self.W_F = nn.Linear(n_dim * n_p, n_dim)

        self.activ = nn.LeakyReLU()

    def forward(self, data, mask=None):
        X = torch.cat((data['loc'], data['demand'][:, :, None]), 2)
        X = torch.cat((X[:, :, 0:2], (X[:, :, 2] / X[:, :, 2].max())[:, :, None]), -1)
        X_loc = X
        distance_matrix = (((X_loc[:, :, None] - X_loc[:, None]) ** 2).sum(-1)) ** .5
        num_samples, num_locations, _ = X.size()
        A = ((1 / distance_matrix) * (torch.eye(num_locations, device=distance_matrix.device).expand(
            (num_samples, num_locations, num_locations)) - 1).to(torch.bool).to(torch.float))
        A[A != A] = 0
        A = A / A.max()
        D = torch.mul(
            torch.eye(num_locations, device=distance_matrix.device).expand((num_samples, num_locations, num_locations)),
            (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))

        # Layer 1

        # p = 1
        F0 = self.init_embed(X)

        # K = 2
        L = D - A

        g_L1_1 = self.W_L_1_G1(torch.cat((F0[:, :, :],
                                          torch.matmul(L, F0)[:, :, :]),
                                         -1))


        F1 = g_L1_1#torch.cat((g_L1_1), -1)
        F1 = self.activ(F1) + F0
        F1 = self.normalization_1(F1)

        F_final = self.activ(self.W_F(F1))

        init_depot_embed = self.init_embed_depot(data['depot'])[:, None]
        h = torch.cat((init_depot_embed, F_final), 1)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )



class GCAPCN_K_2_P_1_L_1(nn.Module):

    def __init__(self,
                 n_layers=2,
                 n_dim=128,
                 n_p=1,
                 node_dim=3,
                 n_K=2
                 ):
        super(GCAPCN_K_2_P_1_L_1, self).__init__()
        self.n_layers = n_layers
        self.n_dim = n_dim
        self.n_p = n_p
        self.n_K = n_K
        self.node_dim = node_dim
        self.init_embed = nn.Linear(node_dim, n_dim * n_p)
        self.init_embed_depot = nn.Linear(2, n_dim)

        self.W_L_1_G1 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)

        self.normalization_1 = Normalization(n_dim * n_p)

        self.W_F = nn.Linear(n_dim * n_p, n_dim)

        self.activ = nn.LeakyReLU()

    def forward(self, data, mask=None):
        X = torch.cat((data['loc'], data['demand'][:, :, None]), 2)
        X = torch.cat((X[:, :, 0:2], (X[:, :, 2] / X[:, :, 2].max())[:, :, None]), -1)
        X_loc = X
        distance_matrix = (((X_loc[:, :, None] - X_loc[:, None]) ** 2).sum(-1)) ** .5
        num_samples, num_locations, _ = X.size()
        A = ((1 / distance_matrix) * (torch.eye(num_locations, device=distance_matrix.device).expand(
            (num_samples, num_locations, num_locations)) - 1).to(torch.bool).to(torch.float))
        A[A != A] = 0
        A = A / A.max()
        D = torch.mul(
            torch.eye(num_locations, device=distance_matrix.device).expand((num_samples, num_locations, num_locations)),
            (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))

        # Layer 1

        # p = 1
        F0 = self.init_embed(X)

        # K = 2
        L = D - A
        L_squared = torch.matmul(L, L)

        g_L1_1 = self.W_L_1_G1(torch.cat((F0[:, :, :],
                                          torch.matmul(L, F0)[:, :, :],
                                          torch.matmul(L_squared, F0)[:, :, :]),
                                         -1))


        F1 = g_L1_1#torch.cat((g_L1_1), -1)
        F1 = self.activ(F1) + F0
        F1 = self.normalization_1(F1)

        F_final = self.activ(self.W_F(F1))

        init_depot_embed = self.init_embed_depot(data['depot'])[:, None]
        h = torch.cat((init_depot_embed, F_final), 1)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )


class GCAPCN_K_3_P_1_L_1(nn.Module):

    def __init__(self,
                 n_layers=2,
                 n_dim=128,
                 n_p=1,
                 node_dim=3,
                 n_K=3
                 ):
        super(GCAPCN_K_3_P_1_L_1, self).__init__()
        self.n_layers = n_layers
        self.n_dim = n_dim
        self.n_p = n_p
        self.n_K = n_K
        self.node_dim = node_dim
        self.init_embed = nn.Linear(node_dim, n_dim * n_p)
        self.init_embed_depot = nn.Linear(2, n_dim)

        self.W_L_1_G1 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)

        self.normalization_1 = Normalization(n_dim * n_p)

        self.W_F = nn.Linear(n_dim * n_p, n_dim)

        self.activ = nn.LeakyReLU()

    def forward(self, data, mask=None):
        X = torch.cat((data['loc'], data['demand'][:, :, None]), 2)
        X = torch.cat((X[:, :, 0:2], (X[:, :, 2] / X[:, :, 2].max())[:, :, None]), -1)
        X_loc = X
        distance_matrix = (((X_loc[:, :, None] - X_loc[:, None]) ** 2).sum(-1)) ** .5
        num_samples, num_locations, _ = X.size()
        A = ((1 / distance_matrix) * (torch.eye(num_locations, device=distance_matrix.device).expand(
            (num_samples, num_locations, num_locations)) - 1).to(torch.bool).to(torch.float))
        A[A != A] = 0
        A = A / A.max()
        D = torch.mul(
            torch.eye(num_locations, device=distance_matrix.device).expand((num_samples, num_locations, num_locations)),
            (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))

        # Layer 1

        # p = 3
        F0 = self.init_embed(X)

        # K = 3
        L = D - A
        L_squared = torch.matmul(L, L)
        L_cube = torch.matmul(L, L_squared)

        g_L1_1 = self.W_L_1_G1(torch.cat((F0[:, :, :],
                                          torch.matmul(L, F0)[:, :, :],
                                          torch.matmul(L_squared, F0)[:, :, :],
                                          torch.matmul(L_cube, F0)[:, :, :]
                                          ),
                                         -1))


        F1 = g_L1_1#torch.cat((g_L1_1), -1)
        F1 = self.activ(F1) + F0
        F1 = self.normalization_1(F1)

        F_final = self.activ(self.W_F(F1))

        init_depot_embed = self.init_embed_depot(data['depot'])[:, None]
        h = torch.cat((init_depot_embed, F_final), 1)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )



class GCAPCN_K_3_P_2_L_1(nn.Module):

    def __init__(self,
                 n_layers=2,
                 n_dim=128,
                 n_p=2,
                 node_dim=3,
                 n_K=3
                 ):
        super(GCAPCN_K_3_P_2_L_1, self).__init__()
        self.n_layers = n_layers
        self.n_dim = n_dim
        self.n_p = n_p
        self.n_K = n_K
        self.node_dim = node_dim
        self.init_embed = nn.Linear(node_dim, n_dim * n_p)
        self.init_embed_depot = nn.Linear(2, n_dim)

        self.W_L_1_G1 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_L_1_G2 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)


        self.normalization_1 = Normalization(n_dim * n_p)

        self.W_F = nn.Linear(n_dim * n_p, n_dim)

        self.activ = nn.LeakyReLU()

    def forward(self, data, mask=None):
        X = torch.cat((data['loc'], data['demand'][:, :, None]), 2)
        X = torch.cat((X[:, :, 0:2], (X[:, :, 2] / X[:, :, 2].max())[:, :, None]), -1)
        X_loc = X
        distance_matrix = (((X_loc[:, :, None] - X_loc[:, None]) ** 2).sum(-1)) ** .5
        num_samples, num_locations, _ = X.size()
        A = ((1 / distance_matrix) * (torch.eye(num_locations, device=distance_matrix.device).expand(
            (num_samples, num_locations, num_locations)) - 1).to(torch.bool).to(torch.float))
        A[A != A] = 0
        A = A / A.max()
        D = torch.mul(
            torch.eye(num_locations, device=distance_matrix.device).expand((num_samples, num_locations, num_locations)),
            (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))

        # Layer 1

        # p = 3
        F0 = self.init_embed(X)
        F0_squared = torch.mul(F0[:, :, :], F0[:, :, :])

        # K = 3
        L = D - A
        L_squared = torch.matmul(L, L)
        L_cube = torch.matmul(L, L_squared)

        g_L1_1 = self.W_L_1_G1(torch.cat((F0[:, :, :],
                                          torch.matmul(L, F0)[:, :, :],
                                          torch.matmul(L_squared, F0)[:, :, :],
                                          torch.matmul(L_cube, F0)[:, :, :]
                                          ),
                                         -1))
        g_L1_2 = self.W_L_1_G2(torch.cat((F0_squared[:, :, :],
                                          torch.matmul(L, F0_squared)[:, :, :],
                                          torch.matmul(L_squared, F0_squared)[:, :, :],
                                          torch.matmul(L_cube, F0_squared)[:, :, :]
                                          ),
                                         -1))


        F1 = torch.cat((g_L1_1, g_L1_2), -1)
        F1 = self.activ(F1) + F0
        F1 = self.normalization_1(F1)

        F_final = self.activ(self.W_F(F1))

        init_depot_embed = self.init_embed_depot(data['depot'])[:, None]
        h = torch.cat((init_depot_embed, F_final), 1)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )




class GCAPCN_K_3_P_3_L_1(nn.Module):

    def __init__(self,
                 n_layers=2,
                 n_dim=128,
                 n_p=3,
                 node_dim=3,
                 n_K=3
                 ):
        super(GCAPCN_K_3_P_3_L_1, self).__init__()
        self.n_layers = n_layers
        self.n_dim = n_dim
        self.n_p = n_p
        self.n_K = n_K
        self.node_dim = node_dim
        self.init_embed = nn.Linear(node_dim, n_dim * n_p)
        self.init_embed_depot = nn.Linear(2, n_dim)

        self.W_L_1_G1 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_L_1_G2 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_L_1_G3 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)

        self.normalization_1 = Normalization(n_dim * n_p)

        self.W_F = nn.Linear(n_dim * n_p, n_dim)

        self.activ = nn.LeakyReLU()

    def forward(self, data, mask=None):
        X = torch.cat((data['loc'], data['demand'][:, :, None]), 2)
        X = torch.cat((X[:, :, 0:2], (X[:, :, 2] / X[:, :, 2].max())[:, :, None]), -1)
        X_loc = X
        distance_matrix = (((X_loc[:, :, None] - X_loc[:, None]) ** 2).sum(-1)) ** .5
        num_samples, num_locations, _ = X.size()
        A = ((1 / distance_matrix) * (torch.eye(num_locations, device=distance_matrix.device).expand(
            (num_samples, num_locations, num_locations)) - 1).to(torch.bool).to(torch.float))
        A[A != A] = 0
        A = A / A.max()
        D = torch.mul(
            torch.eye(num_locations, device=distance_matrix.device).expand((num_samples, num_locations, num_locations)),
            (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))

        # Layer 1

        # p = 3
        F0 = self.init_embed(X)
        F0_squared = torch.mul(F0[:, :, :], F0[:, :, :])
        F0_cube = torch.mul(F0[:, :, :], F0_squared[:, :, :])

        # K = 3
        L = D - A
        L_squared = torch.matmul(L, L)
        L_cube = torch.matmul(L, L_squared)

        g_L1_1 = self.W_L_1_G1(torch.cat((F0[:, :, :],
                                          torch.matmul(L, F0)[:, :, :],
                                          torch.matmul(L_squared, F0)[:, :, :],
                                          torch.matmul(L_cube, F0)[:, :, :]
                                          ),
                                         -1))
        g_L1_2 = self.W_L_1_G2(torch.cat((F0_squared[:, :, :],
                                          torch.matmul(L, F0_squared)[:, :, :],
                                          torch.matmul(L_squared, F0_squared)[:, :, :],
                                          torch.matmul(L_cube, F0_squared)[:, :, :]
                                          ),
                                         -1))

        g_L1_3 = self.W_L_1_G3(torch.cat((F0_cube[:, :, :],
                                          torch.matmul(L, F0_cube)[:, :, :],
                                          torch.matmul(L_squared, F0_cube)[:, :, :],
                                          torch.matmul(L_cube, F0_cube)[:, :, :]
                                          ),
                                         -1))

        F1 = torch.cat((g_L1_1, g_L1_2, g_L1_3), -1)
        F1 = self.activ(F1) + F0
        F1 = self.normalization_1(F1)

        F_final = self.activ(self.W_F(F1))

        init_depot_embed = self.init_embed_depot(data['depot'])[:, None]
        h = torch.cat((init_depot_embed, F_final), 1)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )




class GCAPCN_K_3_P_4_L_1(nn.Module):

    def __init__(self,
                 n_layers=2,
                 n_dim=128,
                 n_p=4,
                 node_dim=4,
                 n_K=3
                 ):
        super(GCAPCN_K_3_P_4_L_1, self).__init__()
        self.n_layers = n_layers
        self.n_dim = n_dim
        self.n_p = n_p
        self.n_K = n_K
        self.node_dim = node_dim
        self.init_embed = nn.Linear(node_dim, n_dim * n_p)
        self.init_embed_depot = nn.Linear(2, n_dim)

        self.W_L_1_G1 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_L_1_G2 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_L_1_G3 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_L_1_G4 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)

        self.normalization_1 = Normalization(n_dim * n_p)

        self.W_F = nn.Linear(n_dim * n_p, n_dim)

        self.activ = nn.LeakyReLU()

    def forward(self, data, mask=None):
        X = torch.cat((data['loc'], data['demand'][:, :, None], data['std'][:, :, None]), 2)
        # X = torch.cat((X[:, :, 0:2], (X[:, :, 2] / X[:, :, 2].max())[:, :, None]), -1)
        X_loc = X
        distance_matrix = (((X_loc[:, :, None] - X_loc[:, None]) ** 2).sum(-1)) ** .5
        num_samples, num_locations, _ = X.size()
        A = ((1 / distance_matrix) * (torch.eye(num_locations, device=distance_matrix.device).expand(
            (num_samples, num_locations, num_locations)) - 1).to(torch.bool).to(torch.float))
        A[A != A] = 0
        A = A / A.max()
        D = torch.mul(
            torch.eye(num_locations, device=distance_matrix.device).expand((num_samples, num_locations, num_locations)),
            (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))

        # Layer 1

        # p = 3
        F0 = self.init_embed(X)
        F0_squared = torch.mul(F0[:, :, :], F0[:, :, :])
        F0_cube = torch.mul(F0[:, :, :], F0_squared[:, :, :])
        F0_quad = torch.mul(F0[:, :, :], F0_cube[:, :, :])

        # K = 3
        L = D - A
        L_squared = torch.matmul(L, L)
        L_cube = torch.matmul(L, L_squared)

        g_L1_1 = self.W_L_1_G1(torch.cat((F0[:, :, :],
                                          torch.matmul(L, F0)[:, :, :],
                                          torch.matmul(L_squared, F0)[:, :, :],
                                          torch.matmul(L_cube, F0)[:, :, :]
                                          ),
                                         -1))
        g_L1_2 = self.W_L_1_G2(torch.cat((F0_squared[:, :, :],
                                          torch.matmul(L, F0_squared)[:, :, :],
                                          torch.matmul(L_squared, F0_squared)[:, :, :],
                                          torch.matmul(L_cube, F0_squared)[:, :, :]
                                          ),
                                         -1))

        g_L1_3 = self.W_L_1_G3(torch.cat((F0_cube[:, :, :],
                                          torch.matmul(L, F0_cube)[:, :, :],
                                          torch.matmul(L_squared, F0_cube)[:, :, :],
                                          torch.matmul(L_cube, F0_cube)[:, :, :]
                                          ),
                                         -1))

        g_L1_4 = self.W_L_1_G4(torch.cat((F0_quad[:, :, :],
                                          torch.matmul(L, F0_quad)[:, :, :],
                                          torch.matmul(L_squared, F0_quad)[:, :, :],
                                          torch.matmul(L_cube, F0_quad)[:, :, :]
                                          ),
                                         -1))

        F1 = torch.cat((g_L1_1, g_L1_2, g_L1_3, g_L1_4), -1)
        F1 = self.activ(F1) + F0
        F1 = self.normalization_1(F1)

        F_final = self.activ(self.W_F(F1))

        init_depot_embed = self.init_embed_depot(data['depot'])[:, None]
        h = torch.cat((init_depot_embed, F_final), 1)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )



class GCAPCN_K_3_P_4_L_2(nn.Module):

    def __init__(self,
                 n_layers=2,
                 n_dim=128,
                 n_p=4,
                 node_dim=3,
                 n_K=3
                 ):
        super(GCAPCN_K_3_P_4_L_2, self).__init__()
        self.n_layers = n_layers
        self.n_dim = n_dim
        self.n_p = n_p
        self.n_K = n_K
        self.node_dim = node_dim
        self.init_embed = nn.Linear(node_dim, n_dim * n_p)
        self.init_embed_depot = nn.Linear(2, n_dim)

        self.W_L_1_G1 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_L_1_G2 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_L_1_G3 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_L_1_G4 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)

        self.W_L_2_G1 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_L_2_G2 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_L_2_G3 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_L_2_G4 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)

        self.normalization_1 = Normalization(n_dim * n_p)
        self.normalization_2 = Normalization(n_dim * n_p)

        self.W_F = nn.Linear(n_dim * n_p, n_dim)

        self.activ = nn.LeakyReLU()

    def forward(self, data, mask=None):
        X = torch.cat((data['loc'], data['demand'][:, :, None]), 2)
        X = torch.cat((X[:, :, 0:2], (X[:, :, 2] / X[:, :, 2].max())[:, :, None]), -1)
        X_loc = X
        distance_matrix = (((X_loc[:, :, None] - X_loc[:, None]) ** 2).sum(-1)) ** .5
        num_samples, num_locations, _ = X.size()
        A = ((1 / distance_matrix) * (torch.eye(num_locations, device=distance_matrix.device).expand(
            (num_samples, num_locations, num_locations)) - 1).to(torch.bool).to(torch.float))
        A[A != A] = 0
        A = A / A.max()
        D = torch.mul(
            torch.eye(num_locations, device=distance_matrix.device).expand((num_samples, num_locations, num_locations)),
            (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))

        # Layer 1

        # p = 3
        F0 = self.init_embed(X)
        F0_squared = torch.mul(F0[:, :, :], F0[:, :, :])
        F0_cube = torch.mul(F0[:, :, :], F0_squared[:, :, :])
        F0_quad = torch.mul(F0[:, :, :], F0_cube[:, :, :])

        # K = 3
        L = D - A
        L_squared = torch.matmul(L, L)
        L_cube = torch.matmul(L, L_squared)

        g_L1_1 = self.W_L_1_G1(torch.cat((F0[:, :, :],
                                          torch.matmul(L, F0)[:, :, :],
                                          torch.matmul(L_squared, F0)[:, :, :],
                                          torch.matmul(L_cube, F0)[:, :, :]
                                          ),
                                         -1))
        g_L1_2 = self.W_L_1_G2(torch.cat((F0_squared[:, :, :],
                                          torch.matmul(L, F0_squared)[:, :, :],
                                          torch.matmul(L_squared, F0_squared)[:, :, :],
                                          torch.matmul(L_cube, F0_squared)[:, :, :]
                                          ),
                                         -1))

        g_L1_3 = self.W_L_1_G3(torch.cat((F0_cube[:, :, :],
                                          torch.matmul(L, F0_cube)[:, :, :],
                                          torch.matmul(L_squared, F0_cube)[:, :, :],
                                          torch.matmul(L_cube, F0_cube)[:, :, :]
                                          ),
                                         -1))

        g_L1_4 = self.W_L_1_G4(torch.cat((F0_quad[:, :, :],
                                          torch.matmul(L, F0_quad)[:, :, :],
                                          torch.matmul(L_squared, F0_quad)[:, :, :],
                                          torch.matmul(L_cube, F0_quad)[:, :, :]
                                          ),
                                         -1))

        F1 = torch.cat((g_L1_1, g_L1_2, g_L1_3, g_L1_4), -1)
        F1 = self.activ(F1) + F0
        F1 = self.normalization_1(F1)


        # Layer 2

        F1_squared = torch.mul(F1[:, :, :], F1[:, :, :])
        F1_cube = torch.mul(F1[:, :, :], F1_squared[:, :, :])
        F1_quad = torch.mul(F1[:, :, :], F1_cube[:, :, :])
        g_L2_1 = self.W_L_2_G1(torch.cat((F1[:, :, :],
                                          torch.matmul(L, F1)[:, :, :],
                                          torch.matmul(L_squared, F1)[:, :, :],
                                          torch.matmul(L_cube, F1)[:, :, :]
                                          ),
                                         -1))
        g_L2_2 = self.W_L_2_G2(torch.cat((F1_squared[:, :, :],
                                          torch.matmul(L, F1_squared)[:, :, :],
                                          torch.matmul(L_squared, F1_squared)[:, :, :],
                                          torch.matmul(L_cube, F1_squared)[:, :, :]
                                          ),
                                         -1))

        g_L2_3 = self.W_L_2_G3(torch.cat((F1_cube[:, :, :],
                                          torch.matmul(L, F1_cube)[:, :, :],
                                          torch.matmul(L_squared, F1_cube)[:, :, :],
                                          torch.matmul(L_cube, F1_cube)[:, :, :]
                                          ),
                                         -1))

        g_L2_4 = self.W_L_2_G4(torch.cat((F1_quad[:, :, :],
                                          torch.matmul(L, F1_quad)[:, :, :],
                                          torch.matmul(L_squared, F1_quad)[:, :, :],
                                          torch.matmul(L_cube, F1_quad)[:, :, :]
                                          ),
                                         -1))


        F2 = self.activ(torch.cat((g_L2_1, g_L2_2, g_L2_3, g_L2_4), -1)) + F1
        F2 = self.normalization_2(F2)


        F_final = self.activ(self.W_F(F2))

        init_depot_embed = self.init_embed_depot(data['depot'])[:, None]
        h = torch.cat((init_depot_embed, F_final), 1)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )



class GCAPCN_K_3_P_4_L_3(nn.Module):

    def __init__(self,
                 n_layers=2,
                 n_dim=128,
                 n_p=4,
                 node_dim=3,
                 n_K=3
                 ):
        super(GCAPCN_K_3_P_4_L_3, self).__init__()
        self.n_layers = n_layers
        self.n_dim = n_dim
        self.n_p = n_p
        self.n_K = n_K
        self.node_dim = node_dim
        self.init_embed = nn.Linear(node_dim, n_dim * n_p)
        self.init_embed_depot = nn.Linear(2, n_dim)

        self.W_L_1_G1 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_L_1_G2 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_L_1_G3 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_L_1_G4 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)

        self.W_L_2_G1 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_L_2_G2 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_L_2_G3 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_L_2_G4 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)

        self.W_L_3_G1 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_L_3_G2 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_L_3_G3 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_L_3_G4 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)

        self.normalization_1 = Normalization(n_dim * n_p)
        self.normalization_2 = Normalization(n_dim * n_p)
        self.normalization_3 = Normalization(n_dim * n_p)

        self.W_F = nn.Linear(n_dim * n_p, n_dim)

        self.activ = nn.LeakyReLU()

    def forward(self, data, mask=None):
        X = torch.cat((data['loc'], data['demand'][:, :, None]), 2)
        X = torch.cat((X[:, :, 0:2], (X[:, :, 2] / X[:, :, 2].max())[:, :, None]), -1)
        X_loc = X
        distance_matrix = (((X_loc[:, :, None] - X_loc[:, None]) ** 2).sum(-1)) ** .5
        num_samples, num_locations, _ = X.size()
        A = ((1 / distance_matrix) * (torch.eye(num_locations, device=distance_matrix.device).expand(
            (num_samples, num_locations, num_locations)) - 1).to(torch.bool).to(torch.float))
        A[A != A] = 0
        A = A / A.max()
        D = torch.mul(
            torch.eye(num_locations, device=distance_matrix.device).expand((num_samples, num_locations, num_locations)),
            (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))

        L = D - A
        L_squared = torch.matmul(L, L)
        L_cube = torch.matmul(L, L_squared)

        # Layer 1

        # p = 3
        F0 = self.init_embed(X)
        F0_squared = torch.mul(F0[:, :, :], F0[:, :, :])
        F0_cube = torch.mul(F0[:, :, :], F0_squared[:, :, :])
        F0_quad = torch.mul(F0[:, :, :], F0_cube[:, :, :])

        # K = 3


        g_L1_1 = self.W_L_1_G1(torch.cat((F0[:, :, :],
                                          torch.matmul(L, F0)[:, :, :],
                                          torch.matmul(L_squared, F0)[:, :, :],
                                          torch.matmul(L_cube, F0)[:, :, :]
                                          ),
                                         -1))
        g_L1_2 = self.W_L_1_G2(torch.cat((F0_squared[:, :, :],
                                          torch.matmul(L, F0_squared)[:, :, :],
                                          torch.matmul(L_squared, F0_squared)[:, :, :],
                                          torch.matmul(L_cube, F0_squared)[:, :, :]
                                          ),
                                         -1))

        g_L1_3 = self.W_L_1_G3(torch.cat((F0_cube[:, :, :],
                                          torch.matmul(L, F0_cube)[:, :, :],
                                          torch.matmul(L_squared, F0_cube)[:, :, :],
                                          torch.matmul(L_cube, F0_cube)[:, :, :]
                                          ),
                                         -1))

        g_L1_4 = self.W_L_1_G4(torch.cat((F0_quad[:, :, :],
                                          torch.matmul(L, F0_quad)[:, :, :],
                                          torch.matmul(L_squared, F0_quad)[:, :, :],
                                          torch.matmul(L_cube, F0_quad)[:, :, :]
                                          ),
                                         -1))

        F1 = torch.cat((g_L1_1, g_L1_2, g_L1_3, g_L1_4), -1)
        F1 = self.activ(F1) + F0
        F1 = self.normalization_1(F1)


        # Layer 2

        F1_squared = torch.mul(F1[:, :, :], F1[:, :, :])
        F1_cube = torch.mul(F1[:, :, :], F1_squared[:, :, :])
        F1_quad = torch.mul(F1[:, :, :], F1_cube[:, :, :])
        g_L2_1 = self.W_L_2_G1(torch.cat((F1[:, :, :],
                                          torch.matmul(L, F1)[:, :, :],
                                          torch.matmul(L_squared, F1)[:, :, :],
                                          torch.matmul(L_cube, F1)[:, :, :]
                                          ),
                                         -1))
        g_L2_2 = self.W_L_2_G2(torch.cat((F1_squared[:, :, :],
                                          torch.matmul(L, F1_squared)[:, :, :],
                                          torch.matmul(L_squared, F1_squared)[:, :, :],
                                          torch.matmul(L_cube, F1_squared)[:, :, :]
                                          ),
                                         -1))

        g_L2_3 = self.W_L_2_G3(torch.cat((F1_cube[:, :, :],
                                          torch.matmul(L, F1_cube)[:, :, :],
                                          torch.matmul(L_squared, F1_cube)[:, :, :],
                                          torch.matmul(L_cube, F1_cube)[:, :, :]
                                          ),
                                         -1))

        g_L2_4 = self.W_L_2_G4(torch.cat((F1_quad[:, :, :],
                                          torch.matmul(L, F1_quad)[:, :, :],
                                          torch.matmul(L_squared, F1_quad)[:, :, :],
                                          torch.matmul(L_cube, F1_quad)[:, :, :]
                                          ),
                                         -1))


        F2 = self.activ(torch.cat((g_L2_1, g_L2_2, g_L2_3, g_L2_4), -1)) + F1
        F2 = self.normalization_2(F2)

        # Layer 3

        F2_squared = torch.mul(F2[:, :, :], F2[:, :, :])
        F2_cube = torch.mul(F2[:, :, :], F2_squared[:, :, :])
        F2_quad = torch.mul(F2[:, :, :], F2_cube[:, :, :])
        g_L3_1 = self.W_L_3_G1(torch.cat((F2[:, :, :],
                                          torch.matmul(L, F2)[:, :, :],
                                          torch.matmul(L_squared, F2)[:, :, :],
                                          torch.matmul(L_cube, F2)[:, :, :]
                                          ),
                                         -1))
        g_L3_2 = self.W_L_3_G2(torch.cat((F2_squared[:, :, :],
                                          torch.matmul(L, F2_squared)[:, :, :],
                                          torch.matmul(L_squared, F2_squared)[:, :, :],
                                          torch.matmul(L_cube, F2_squared)[:, :, :]
                                          ),
                                         -1))

        g_L3_3 = self.W_L_3_G3(torch.cat((F2_cube[:, :, :],
                                          torch.matmul(L, F2_cube)[:, :, :],
                                          torch.matmul(L_squared, F2_cube)[:, :, :],
                                          torch.matmul(L_cube, F2_cube)[:, :, :]
                                          ),
                                         -1))
        g_L3_4 = self.W_L_3_G4(torch.cat((F2_quad[:, :, :],
                                          torch.matmul(L, F2_quad)[:, :, :],
                                          torch.matmul(L_squared, F2_quad)[:, :, :],
                                          torch.matmul(L_cube, F2_quad)[:, :, :]
                                          ),
                                         -1))

        F3 = self.activ(torch.cat((g_L3_1, g_L3_2, g_L3_3, g_L3_4), -1)) + F2
        F3 = self.normalization_3(F3)

        F_final = self.activ(self.W_F(F3))

        init_depot_embed = self.init_embed_depot(data['depot'])[:, None]
        h = torch.cat((init_depot_embed, F_final), 1)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )
