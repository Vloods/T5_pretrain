import logging
import os

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
'''
from transformers.file_utils import (
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_NAME,
#    cached_path,
    get_from_cache,
    hf_bucket_url,
    is_remote_url,
)
'''
from transformers.utils.hub import cached_file as cached_path
from modeling import modeling_gnn
from utils import layers
from utils import utils

import json
from peft import LoraConfig, get_peft_model

from transformers.models.llama import modeling_llama
from llm2vec.models.bidirectional_llama import LlamaBiModel
from transformers import LlamaConfig
import llm2vec
import torch

from transformers import LlamaModel, LlamaForCausalLM, LlamaPreTrainedModel, LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaSdpaAttention,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

from torch import nn
from transformers.cache_utils import Cache, StaticCache

from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from peft import PeftModel




class ModifiedLlamaAttention(LlamaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedLlamaFlashAttention2(LlamaFlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedLlamaSdpaAttention(LlamaSdpaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


LLAMA_ATTENTION_CLASSES = {
    "eager": ModifiedLlamaAttention,
    "flash_attention_2": ModifiedLlamaFlashAttention2,
    "sdpa": ModifiedLlamaSdpaAttention,
}


class ModifiedLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx
        )

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class Args:
    def __init__(self):
        self.residual_ie = 2
        self.fp16 = True
        self.upcast = True

        # For LMGNN class
        self.mlm_task = True
        self.link_task = True
        if self.link_task:
            self.link_decoder = "DistMult"
            self.link_negative_adversarial_sampling = True
            self.link_negative_adversarial_sampling_temperature = 1.0
            self.link_regularizer_weight = 0.0
            self.link_proj_headtail = True
            self.link_normalize_headtail = 3
            self.scaled_distmult = True
            # self.no_node_score = True
            # self.end_task = True
        self.no_node_score = True
        self.end_task = True


class DRAGON(nn.Module):

    def __init__(self, config, args={}, model_name="McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp", k=5, n_ntype=4, n_etype= 819 * 2,
                 n_concept=4335104, concept_dim=200, concept_in_dim=1024, n_attention_head=2,
                 fc_dim=200, n_fc_layer=0, p_emb=0.2, p_gnn=0.2, p_fc=0.2,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.02, ie_dim=400, info_exchange=True, ie_layer_num=1, sep_ie_layers=False, layer_id=-1):
        super().__init__()

        self.n_ntype = n_ntype
        self.n_etype = n_etype

        self.lmgnn = LMGNN(config=config, args=args, model_name=model_name, k=k, n_ntype=n_ntype, n_etype=n_etype, n_concept=n_concept, concept_dim=concept_dim, concept_in_dim=concept_in_dim, n_attention_head=n_attention_head, fc_dim=fc_dim, n_fc_layer=n_fc_layer, p_emb=p_emb, p_gnn=p_gnn, p_fc=p_fc, pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb, init_range=init_range, ie_dim=ie_dim, info_exchange=info_exchange, ie_layer_num=ie_layer_num,  sep_ie_layers=sep_ie_layers, layer_id=layer_id)
        self.loading_info = json.load(open("llama_lora.loading_info", 'r'))

        
    def batch_graph(self, edge_index_init, edge_type_init, pos_triples_init, neg_nodes_init, n_nodes):
        """
        edge_index_init:  list of (n_examples, ). each entry is torch.tensor(2, E?)    ==> [2, total_E]
        edge_type_init:   list of (n_examples, ). each entry is torch.tensor(E?, )     ==> [total_E, ]
        pos_triples_init: list of (n_examples, ). each entry is [h,r,t] where h/r/t: torch.tensor(n_triple?, ) ==> [3, `total_n_triple`]
        neg_nodes_init:   list of (n_examples, ). each entry is torch.tensor(n_triple?, n_neg) ==> [`total_n_triple`, n_neg]
        """
        n_examples = len(edge_index_init)
        edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1) #[2, total_E]
        edge_type = torch.cat(edge_type_init, dim=0) #[total_E, ]

        pos_triples = [[], [], []]
        for _i_ in range(n_examples):
            h = pos_triples_init[_i_][0] + _i_ * n_nodes #tensor[n_triple?,]
            r = pos_triples_init[_i_][1]                 #tensor[n_triple?,]
            t = pos_triples_init[_i_][2] + _i_ * n_nodes #tensor[n_triple?,]
            pos_triples[0].append(h)
            pos_triples[1].append(r)
            pos_triples[2].append(t)
        pos_triples = torch.stack([torch.cat(item) for item in pos_triples]) #[3, `total_n_triple`] where `total_n_triple` is sum of n_triple within batch
        assert pos_triples.size(0) == 3

        neg_nodes = [neg_nodes_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        neg_nodes = torch.cat(neg_nodes) #[`total_n_triple`, n_neg]
        assert neg_nodes.dim() == 2
        assert pos_triples.size(1) == neg_nodes.size(0)
        return edge_index, edge_type, pos_triples, neg_nodes

    def forward(self, *inputs, cache_output=False, detail=False):
        """
        inputs_ids: (batch_size, num_choice, seq_len)    -> (batch_size * num_choice, seq_len)
        concept_ids: (batch_size, num_choice, n_node)  -> (batch_size * num_choice, n_node)
        node_type_ids: (batch_size, num_choice, n_node) -> (batch_size * num_choice, n_node)
        node_scores: [bs, nc, n_node, 1]
        adj_lengths: means the "actual" number of nodes (excluding padding)(batch_size, num_choice)          -> (batch_size * num_choice, )
        adj -> edge_index, edge_type
            edge_index: list of (batch_size, num_choice) -> list of (batch_size * num_choice, ); each entry is torch.tensor(2, E(variable))
                                                         -> (2, total E)
            edge_type:  list of (batch_size, num_choice) -> list of (batch_size * num_choice, ); each entry is torch.tensor(E(variable), )
                                                         -> (total E, )

        returns:
        logits: [bs, nc]
        """
        bs, nc = inputs[0].size(0), inputs[0].size(1)

        #Here, merge the batch dimension and the num_choice dimension
        assert len(inputs) == 6 + 5 + 2 + 2  #6 lm_data, 5 gnn_data, (edge_index, edge_type), (pos_triples, neg_nodes)
        edge_index_orig, edge_type_orig = inputs[-2:]
        _inputs = [x.reshape(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[:6]] + [x.reshape(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[6:11]] + [sum(x,[]) for x in inputs[11:15]]

        *lm_inputs, concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask, edge_index, edge_type, pos_triples, neg_nodes = _inputs
        # node_scores = torch.zeros_like(node_scores) #handled in LMGNN forward
        edge_index, edge_type, pos_triples, neg_nodes = self.batch_graph(edge_index, edge_type, pos_triples, neg_nodes, concept_ids.size(1))
        device = node_type_ids.device
        adj     = (edge_index.to(device), edge_type.to(device))
        lp_data = (pos_triples.to(device), neg_nodes.to(device))

        logits, lm_loss, link_losses = self.lmgnn(lm_inputs, concept_ids,
                                    node_type_ids, node_scores, adj_lengths, special_nodes_mask, adj, lp_data,
                                    emb_data=None, cache_output=cache_output)
        # logits: [bs * nc], lm_loss: scalar, link_losses: (scalar, scalar, scalar)
        if logits is not None:
            logits = logits.view(bs, nc)
        lm_loss = lm_loss * bs
        link_losses = [item * bs for item in link_losses]
        if not detail:
            return logits, lm_loss, link_losses
        else:
            return logits, lm_loss, link_losses, concept_ids.view(bs, nc, -1), node_type_ids.view(bs, nc, -1), edge_index_orig, edge_type_orig
            # edge_index_orig: list of (batch_size, num_choice). each entry is torch.tensor(2, E)
            # edge_type_orig: list of (batch_size, num_choice). each entry is torch.tensor(E, )

    def get_fake_inputs(self, device="cuda:0"):
        bs = 2
        nc = 5
        seq_len = 100
        lm_inputs = torch.zeros([bs, nc, seq_len], dtype=torch.long).to(device)
        lm_labels = torch.zeros([bs, nc, seq_len], dtype=torch.long).to(device)
        input_ids = torch.zeros([bs, nc, seq_len], dtype=torch.long).to(device)
        token_type_ids = torch.zeros([bs, nc, seq_len], dtype=torch.long).to(device)
        attention_mask = torch.ones([bs, nc, seq_len]).to(device)
        output_mask = torch.zeros([bs, nc]).to(device)

        n_node = 200
        concept_ids = torch.arange(end=n_node).repeat(bs, nc, 1).to(device)
        adj_lengths = torch.zeros([bs, nc], dtype=torch.long).fill_(10).to(device)

        n_edges = 3
        edge_index = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(device)
        edge_type = torch.zeros(n_edges, dtype=torch.long).fill_(2).to(device)

        node_type = torch.zeros([bs, nc, n_node], dtype=torch.long).to(device)
        node_type[:, :, 0] = 3
        node_score = torch.zeros([bs, nc, n_node, 1]).to(device)
        node_score[:, :, 1] = 180

        special_nodes_mask = torch.zeros([bs, nc, n_node], dtype=torch.long).to(device)

        edge_index = [[edge_index] * nc] * bs
        edge_type = [[edge_type] * nc] * bs

        pos_triples = [[ [torch.zeros( [100,], dtype=torch.long ) ] * 3] * nc] * bs
        neg_nodes = [[torch.zeros([seq_len,64], dtype=torch.long) ] * nc] * bs
        return lm_inputs, lm_labels, input_ids, attention_mask, token_type_ids, output_mask, \
            concept_ids, node_type, node_score, adj_lengths, special_nodes_mask, \
            edge_index, edge_type, pos_triples, neg_nodes

    def check_outputs(self, logits, lm_loss, link_losses):
        bs = 2
        nc = 5
        assert logits.size() == (bs, nc)
        print("logits", logits)
        print("lm_loss", lm_loss)
        print("link_losses", link_losses)
        print("Success")


def test_DRAGON(device):
    config, _ = LlamaConfig.from_pretrained(
        "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
        cache_dir=None, return_unused_kwargs=True,
        force_download=False,
        output_hidden_states=True,
    )
    class Args:
        def __init__(self):
            self.residual_ie = 2
            self.fp16 = True
            self.upcast = True

            # For LMGNN class
            self.mlm_task = True
            self.link_task = True
            if self.link_task:
                self.link_decoder = "DistMult"
                self.link_negative_adversarial_sampling = True
                self.link_negative_adversarial_sampling_temperature = 1.0
                self.link_regularizer_weight = 0.0
                self.link_proj_headtail = True
                self.link_normalize_headtail = 3
                self.scaled_distmult = True
                # self.no_node_score = True
                # self.end_task = True
            self.no_node_score = True
            self.end_task = True

    test_args = Args()
    cp_emb = [np.load("data/cpnet/tzw.ent.npy")]
    cp_emb = np.concatenate(cp_emb, 1)
    cp_emb = torch.tensor(cp_emb, dtype=torch.float32)
    model = DRAGON(config, test_args, pretrained_concept_emb=cp_emb,k=2).to(device)
    inputs = model.get_fake_inputs(device)
    outputs = model(*inputs)
    model.check_outputs(*outputs)


ModelClass = llm2vec.models.bidirectional_llama.LlamaBiModel
PreTrainedModelClass = modeling_llama.LlamaPreTrainedModel
print("PreTrainedModelClass:", PreTrainedModelClass)


class LMGNN(PreTrainedModelClass):

    def __init__(self, config, args={}, model_name="McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp", k=5, n_ntype=4, n_etype=38,
                 n_concept=4335104, concept_dim=200, concept_in_dim=1024, n_attention_head=2,
                 fc_dim=200, n_fc_layer=0, p_emb=0.2, p_gnn=0.2, p_fc=0.2,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.02, ie_dim=200, info_exchange=True, ie_layer_num=1, sep_ie_layers=False, layer_id=-1):
        super().__init__(config)
        self.args = args
        self.config = config

        self.init_range = init_range

        self.k = k
        self.concept_dim = concept_dim
        self.n_attention_head = n_attention_head
        self.activation = layers.GELU()
        if k >= 0:
            self.concept_emb = layers.CustomizedEmbedding(concept_num=n_concept, concept_out_dim=concept_dim, use_contextualized=False, concept_in_dim=concept_in_dim, pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb)
            self.pooler = layers.MultiheadAttPoolLayer(n_attention_head, config.hidden_size, concept_dim)

        concat_vec_dim = concept_dim * 2 + config.hidden_size if k>=0 else config.hidden_size
        self.fc = layers.MLP(concat_vec_dim, fc_dim, 1, n_fc_layer, p_fc, layer_norm=True)

        self.dropout_e = nn.Dropout(p_emb)
        self.dropout_fc = nn.Dropout(p_fc)

        if init_range > 0:
            self.apply(self._init_weights)


        self.llama = TextKGMessagePassing(config, args=args, k=k, n_ntype=n_ntype, n_etype=n_etype, dropout=p_gnn, concept_dim=concept_dim, ie_dim=ie_dim, p_fc=p_fc, info_exchange=info_exchange, ie_layer_num=ie_layer_num, sep_ie_layers=sep_ie_layers) #this is equivalent to RobertaModel
        if args.mlm_task:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=torch.float32)

        self.layer_id = layer_id
        self.cpnet_vocab_size = n_concept

        if args.link_task:
            if args.link_decoder == 'DistMult':
                self.linkpred = modeling_gnn.DistMultDecoder(args, num_rels=n_etype, h_dim=concept_dim)
            elif args.link_decoder == 'TransE':
                self.linkpred = modeling_gnn.TransEDecoder(args, num_rels=n_etype, h_dim=concept_dim)
            elif args.link_decoder == 'RotatE':
                self.linkpred = modeling_gnn.RotatEDecoder(args, num_rels=n_etype, h_dim=concept_dim)
            else:
                raise NotImplementedError
            if args.link_proj_headtail:
                self.linkpred_proj = nn.Linear(concept_dim, concept_dim)
            if args.link_normalize_headtail == 3:
                self.emb_LayerNorm = nn.LayerNorm(concept_dim)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs, concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask, adj, lp_data, emb_data=None, cache_output=False):
        """
        concept_ids: (batch_size, n_node)
        adj_lengths: (batch_size,)
        node_type_ids: (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_scores: (batch_size, n_node, 1)
        adj: edge_index, edge_type
        lp_data: pos_triples, neg_nodes

        returns:
        logits: [bs]
        """
        #LM inputs
        lm_input_ids, lm_labels, input_ids, attention_mask, token_type_ids, output_mask = inputs
        if self.args.mlm_task:
            input_ids = lm_input_ids

        # GNN inputs
        concept_ids[concept_ids == 0] = self.cpnet_vocab_size + 2
        if self.k >= 0:
            gnn_input = self.concept_emb(concept_ids - 1, emb_data).to(node_type_ids.device)
        else:
            gnn_input = torch.zeros((concept_ids.size(0), concept_ids.size(1), self.concept_dim)).float().to(node_type_ids.device)
        gnn_input[:, 0] = 0
        gnn_input = self.dropout_e(gnn_input) #(batch_size, n_node, dim_node)

        #Normalize node sore (use norm from Z)
        if self.args.no_node_score:
            node_scores = node_scores.new_zeros(node_scores.size())
        else:
            _mask = (torch.arange(node_scores.size(1), device=node_scores.device) < adj_lengths.unsqueeze(1)).float() #0 means masked out #[batch_size, n_node]
            node_scores = -node_scores
            node_scores = node_scores - node_scores[:, 0:1, :] #[batch_size, n_node, 1]
            node_scores = node_scores.squeeze(2) #[batch_size, n_node]
            node_scores = node_scores * _mask
            mean_norm  = (torch.abs(node_scores)).sum(dim=1) / adj_lengths  #[batch_size, ]
            node_scores = node_scores / (mean_norm.unsqueeze(1) + 1e-05) #[batch_size, n_node]
            node_scores = node_scores.unsqueeze(2) #[batch_size, n_node, 1]

 #       llamagnn = self.llama
        # Merged core
        lm_outputs, gnn_output = self.llama(input_ids, token_type_ids, attention_mask, output_mask, gnn_input, adj, node_type_ids, node_scores, special_nodes_mask, output_hidden_states=True)
        # lm_outputs: ([bs, seq_len, sent_dim], [bs, sent_dim], ([bs, seq_len, sent_dim] for _ in range(25)))
        # gnn_output: [bs, n_node, dim_node]

        # LM outputs
        all_hidden_states = lm_outputs[-1] # ([bs, seq_len, sent_dim] for _ in range(25))
        lm_hidden_states = all_hidden_states[self.layer_id] # [bs, seq_len, sent_dim]
        sent_vecs = self.llama.pooler(lm_hidden_states) # [bs, sent_dim]
        del lm_outputs
        # sent_token_mask = output_mask.clone()
        # sent_token_mask[:, 0] = 0

        _bs, _seq_len, _ = lm_hidden_states.size()
        
        if self.args.mlm_task:
            lm_logits = self.lm_head(lm_hidden_states)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            lm_labels = lm_labels.to(lm_logits.device)
            lm_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))

        else:
            lm_loss = 0.

        # GNN outputs
        Z_vecs = gnn_output[:,0]   #(batch_size, dim_node)

        node_mask = torch.arange(node_type_ids.size(1), device=node_type_ids.device) >= adj_lengths.unsqueeze(1) #[bs, nodes] 1 means masked out
        gnn_output = gnn_output * (~node_mask).float().unsqueeze(2)
        node_mask = node_mask | (node_type_ids == 3) # pool over all KG nodes (excluding the context node)
        node_mask[node_mask.all(1), 0] = 0  # a temporary solution to avoid zero node


        # sent_node_mask = special_nodes_mask.clone()
        # sent_node_mask[:, 0] = 0

        if self.args.link_task:
            pos_triples, neg_nodes = lp_data #pos_triples: [3, `total_n_triple`],  neg_nodes: [`total_n_triple`, n_neg]

            pos_samples = pos_triples #[3, `total_n_triple`]

            _n_neg = neg_nodes.size(1)
            head_negative_sample = neg_nodes[:, :_n_neg//2]             #[`total_n_triple`, n_neg//2]
            tail_negative_sample = neg_nodes[:, _n_neg//2:_n_neg//2*2]  #[`total_n_triple`, n_neg//2]

            _bs, _, gnn_dim = gnn_output.size()
            embs = gnn_output.view(-1, gnn_dim) #[`total_n_nodes`, gnn_dim]

            if self.args.link_proj_headtail:
                embs = self.linkpred_proj(embs)
            if self.args.link_normalize_headtail == 1:
                embs = embs / torch.norm(embs, p=2, dim=1, keepdim=True).detach()
            elif self.args.link_normalize_headtail == 2:
                embs = torch.tanh(embs)
            elif self.args.link_normalize_headtail == 3:
                embs = self.emb_LayerNorm(embs)

            positive_score  = self.linkpred(embs, pos_samples) #[`total_n_triple`, 1]
            head_neg_scores = self.linkpred(embs, (pos_samples, head_negative_sample), mode='head-batch')
            tail_neg_scores = self.linkpred(embs, (pos_samples, tail_negative_sample), mode='tail-batch')
            negative_score = torch.cat([head_neg_scores, tail_neg_scores], dim=-1) #[`total_n_triple`, total_n_neg]
            scores = (positive_score, negative_score)

            link_loss, pos_link_loss, neg_link_loss = self.linkpred.loss(scores)
        else:
            link_loss = pos_link_loss = neg_link_loss = 0.

        # Concatenated pool
        if self.args.end_task:
            sent_vecs_for_pooler = sent_vecs
            if self.k >= 0:
                graph_vecs, pool_attn = self.pooler(sent_vecs_for_pooler, gnn_output.to(torch.float32), node_mask) #graph_vecs: [bs, node_dim]
                concat_pool = torch.cat((graph_vecs, sent_vecs, Z_vecs), 1)
            else:
                concat_pool = sent_vecs
            logits = self.fc(self.dropout_fc(concat_pool)) #[bs, 1]
        else:
            logits = None

        return logits, lm_loss, (link_loss, pos_link_loss, neg_link_loss)

   

    def get_fake_inputs(self, device="cuda:0"):
        bs = 1
        seq_len = 100
        input_ids = torch.zeros([bs, seq_len], dtype=torch.long).to(device)
        lm_input_ids = torch.zeros([bs, seq_len], dtype=torch.long).to(device)
        lm_labls = torch.zeros([bs, seq_len], dtype=torch.long).to(device)

        token_type_ids = torch.zeros([bs, seq_len], dtype=torch.long).to(device)
        attention_mask = torch.ones([bs, seq_len]).to(device)
        output_mask = torch.ones([bs, seq_len]).to(device)

        n_node = 200
        concept_ids = torch.arange(end=n_node).repeat(bs, 1).to(device)
        adj_lengths = torch.zeros([bs], dtype=torch.long).fill_(10).to(device)

        n_edges = 3
        edge_index = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(device)
        edge_type = torch.zeros(n_edges, dtype=torch.long).fill_(2).to(device)
        adj = (edge_index, edge_type)

        node_type = torch.zeros([bs, n_node], dtype=torch.long).to(device)
        node_type[:, 0] = 3
        node_score = torch.zeros([bs, n_node, 1]).to(device)
        node_score[:, 1] = 180

        # if link_task == False:
        pos_triples = torch.zeros([3, 1000]).to(device).type(torch.long)
        neg_nodes = torch.zeros([1000, 64]).to(device).type(torch.long)
        lp_data = (pos_triples, neg_nodes)
        special_nodes_mask = torch.zeros([bs, n_node]).to(device)
        return (lm_input_ids, lm_labls, input_ids, attention_mask, token_type_ids, output_mask), \
            concept_ids, node_type, node_score, adj_lengths, special_nodes_mask, adj, lp_data

    def check_outputs(self, logits):
        bs = 1
        assert logits.size() == (bs, 1)
        print("Success")
        
import numpy as np

def test_LMGNN(device):
    config, _ = LlamaConfig.from_pretrained(
        "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
        cache_dir=None, return_unused_kwargs=True,
        force_download=False,
        output_hidden_states=True,
    )
    class Args:
        def __init__(self):
            self.residual_ie = 2
            self.fp16 = True
            self.upcast = True

            # For LMGNN class
            self.mlm_task = True
            self.link_task = True
            if self.link_task:
                self.link_decoder = "DistMult"
                self.link_negative_adversarial_sampling = True
                self.link_negative_adversarial_sampling_temperature = 1.0
                self.link_regularizer_weight = 0.0
                self.link_proj_headtail = True
                self.link_normalize_headtail = 3
                self.scaled_distmult = True
                # self.no_node_score = True
                # self.end_task = True
            self.no_node_score = True
            self.end_task = True

    test_args = Args()
    cp_emb = [np.load("data/cpnet/tzw.ent.npy")]
    cp_emb = np.concatenate(cp_emb, 1)
    cp_emb = torch.tensor(cp_emb, dtype=torch.bfloat16)
    model = LMGNN(config, test_args, pretrained_concept_emb=cp_emb).to(device)
    inputs = model.get_fake_inputs(device)
    outputs = model(*inputs)
    model.check_outputs(outputs[0])

from transformers import PretrainedConfig, BitsAndBytesConfig

class LLaMaAvgPooler(nn.Module):
    '''
    T5AvgPooler
    '''
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, dtype=torch.float32)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Average pooling
        avg_pooled_output = hidden_states.mean(dim=1)
        pooled_output = self.dense(avg_pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output


from peft import PeftModel

class TextKGMessagePassing(ModelClass):

    def __init__(self, config, args={}, k=5, n_ntype=4, n_etype=38, dropout=0.2, concept_dim=200, ie_dim=200, p_fc=0.2, info_exchange=True, ie_layer_num=1, sep_ie_layers=False):
        super().__init__(config=config)

        self.n_ntype = n_ntype
        self.n_etype = n_etype

        self.hidden_size = concept_dim
        self.emb_node_type = nn.Linear(self.n_ntype, concept_dim // 2)

        self.basis_f = 'sin' #['id', 'linact', 'sin', 'none']
        if self.basis_f in ['id']:
            self.emb_score = nn.Linear(1, concept_dim // 2)
        elif self.basis_f in ['linact']:
            self.B_lin = nn.Linear(1, concept_dim // 2)
            self.emb_score = nn.Linear(concept_dim // 2, concept_dim // 2)
        elif self.basis_f in ['sin']:
            self.emb_score = nn.Linear(concept_dim // 2, concept_dim // 2)

        self.k = k

        self.Vh = nn.Linear(concept_dim, concept_dim)
        self.Vx = nn.Linear(concept_dim, concept_dim)

        self.activation = layers.GELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout

        self.encoder = LLaMaGAT(config, args, k=k, n_ntype=n_ntype, n_etype=n_etype, hidden_size=concept_dim, dropout=dropout, concept_dim=concept_dim, ie_dim=ie_dim, p_fc=p_fc, info_exchange=info_exchange, 
                                ie_layer_num=ie_layer_num, sep_ie_layers=sep_ie_layers)

        self.encoder.layers = PeftModel.from_pretrained(self.encoder, 'McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-unsup-simcse').layers
        
        self.pooler = LLaMaAvgPooler(config)
        
        self.sent_dim = config.hidden_size

        del self.layers

    def forward(self, input_ids, token_type_ids, attention_mask, special_tokens_mask, H, A, node_type, node_score, special_nodes_mask, cache_output=False, position_ids=None, head_mask=None, output_hidden_states=True):
        """
        input_ids: [bs, seq_len]
        token_type_ids: [bs, seq_len]
        attention_mask: [bs, seq_len]
        H: tensor of shape (batch_size, n_node, d_node)
            node features from the previous layer
        A: (edge_index, edge_type)
            edge_index: [2, n_edges]
            edge_type: [n_edges]
        node_type: long tensor of shape (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_score: tensor of shape (batch_size, n_node, 1)
        """
        # LM inputs
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 1D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.encoder.embed_tokens(input_ids)
        
        extended_attention_mask = attention_mask #self.encoder.model._prepare_decoder_attention_mask(
        #    attention_mask, attention_mask.size(), embedding_output, 0
        #")
        
        # GNN inputs
        _batch_size, _n_nodes = node_type.size()

        #Embed type
        T = modeling_gnn.make_one_hot(node_type.view(-1).contiguous(), self.n_ntype).view(_batch_size, _n_nodes, self.n_ntype)
        node_type_emb = self.activation(self.emb_node_type(T)) #[batch_size, n_node, dim/2]

        #Embed score
        if self.basis_f == 'sin':
            js = torch.arange(self.hidden_size//2).unsqueeze(0).unsqueeze(0).float().to(node_type.device) #[1,1,dim/2]
            js = torch.pow(1.1, js) #[1,1,dim/2]
            B = torch.sin(js * node_score) #[batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]
        elif self.basis_f == 'id':
            B = node_score
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]
        elif self.basis_f == 'linact':
            B = self.activation(self.B_lin(node_score)) #[batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]

        X = H
        edge_index, edge_type = A #edge_index: [2, total_E]   edge_type: [total_E, ]  where total_E is for the batched graph
        _X = X.view(-1, X.size(2)).contiguous() #[`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node
        _node_type = node_type.view(-1).contiguous() #[`total_n_nodes`, ]
        _node_feature_extra = torch.cat([node_type_emb, node_score_emb], dim=2).view(_node_type.size(0), -1).contiguous() #[`total_n_nodes`, dim]

        # Merged core
        encoder_outputs, _X = self.encoder(embedding_output,
                                       extended_attention_mask, special_tokens_mask, head_mask, _X, edge_index, edge_type, _node_type, _node_feature_extra, special_nodes_mask)

        # LM outputs
        sequence_output = encoder_outputs[0]
        print(sequence_output.dtype)

        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here

        # GNN outputs
        X = _X.view(node_type.size(0), node_type.size(1), -1) #[batch_size, n_node, dim]

        output = self.activation(self.Vh(H) + self.Vx(X))
        output = self.dropout(output)

        return outputs, output


    def get_fake_inputs(self, device="cuda:0"):
        bs = 1
        seq_len = 100
        input_ids = torch.zeros([bs, seq_len], dtype=torch.long).to(device)
        token_type_ids = torch.zeros([bs, seq_len], dtype=torch.long).to(device)
        attention_mask = torch.ones([bs, seq_len]).to(device)
        
        n_node = 200
        H = torch.zeros([bs, n_node, self.hidden_size]).to(device)
        n_edges = 3
        edge_index = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(device)
        edge_type = torch.zeros(n_edges, dtype=torch.long).fill_(2).to(device)
        A = (edge_index, edge_type)

        node_type = torch.zeros([bs, n_node], dtype=torch.long).to(device)
        node_type[:, 0] = 3
        node_score = torch.zeros([bs, n_node, 1]).to(device)
        node_score[:, 1] = 180
        
        special_nodes_mask = torch.zeros([bs, n_node], dtype=torch.long).to(device)
        special_tokens_mask = torch.zeros([bs, seq_len], dtype=torch.long).to(device)
        
        return input_ids, token_type_ids, attention_mask, special_nodes_mask, H, A,  node_type, node_score, \
            special_tokens_mask,

    def check_outputs(self, outputs, gnn_output):
        bs = 1
        seq_len = 100
        print(outputs[0].size())
        print(self.sent_dim)
        assert outputs[0].size() == (bs, seq_len, self.sent_dim)
        n_node = 200
        assert gnn_output.size() == (bs, n_node, self.hidden_size)


def test_TextKGMessagePassing(device):
    config, _ = LlamaConfig.from_pretrained(
        "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
        cache_dir=None, return_unused_kwargs=True,
        force_download=False,
        output_hidden_states=True,
    )
    class Args:
        def __init__(self):
            self.residual_ie = 2
            self.fp16 = True
            self.upcast = True
            self.update_ie = True

    test_args = Args()
    model = TextKGMessagePassing(config, args=test_args, k=5).to(device)
    inputs = model.get_fake_inputs(device)
    lm_outputs, gnn_output  = model(*inputs)
    model.check_outputs(lm_outputs, gnn_output)
    print("Success")


from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


class LLaMaGAT(LlamaModel):

    def __init__(self, config, args, k=5, n_ntype=4, n_etype=38, hidden_size=200, dropout=0.2, concept_dim=200, ie_dim=200, p_fc=0.2, info_exchange=True, ie_layer_num=1, sep_ie_layers=False):

        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                ModifiedLlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
        # Gemma2 initialization
#        self._set_default_torch_dtype(torch.bfloat16)
        # Initialize weights and apply final processing
        self.post_init()
        
        self.args = args
        self.k = k
        self.concept_dim = concept_dim
        self.num_hidden_layers = config.num_hidden_layers
        self.info_exchange = info_exchange
        if k >= 1:
            self.edge_encoder = torch.nn.Sequential(torch.nn.Linear(n_etype + 1 + n_ntype * 2, hidden_size), torch.nn.BatchNorm1d(hidden_size), torch.nn.ReLU(), torch.nn.Linear(hidden_size, hidden_size))
            self.gnn_layers = nn.ModuleList([modeling_gnn.GATConvE(args, hidden_size, n_ntype, n_etype, self.edge_encoder) for _ in range(k)])
            self.activation = layers.GELU()
            self.dropout_rate = dropout

            self.sent_dim = config.hidden_size
            self.sep_ie_layers = sep_ie_layers
            if sep_ie_layers:
                self.ie_layers = nn.ModuleList([layers.MLP(self.sent_dim + concept_dim, ie_dim, self.sent_dim + concept_dim, ie_layer_num, p_fc) for _ in range(k)])
            else:
                self.ie_layer = layers.MLP(self.sent_dim + concept_dim, ie_dim, self.sent_dim + concept_dim, ie_layer_num, p_fc)
            if self.args.residual_ie == 2:
                self.ie_LayerNorm = nn.LayerNorm(self.sent_dim + concept_dim)
                
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, hidden_states, attention_mask, special_tokens_mask, head_mask, _X, edge_index, edge_type, _node_type, _node_feature_extra, special_nodes_mask, output_attentions=False, output_hidden_states=True, past_key_values=None, position_ids=None, use_cache=False, **flash_attn_kwargs):
        """
        hidden_states: [bs, seq_len, sent_dim]
        attention_mask: [bs, 1, 1, seq_len]
        head_mask: list of shape [num_hidden_layers]

        _X: [`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node
        edge_index: [2, n_edges]
        edge_type: [n_edges]
        _node_type: [bs * n_nodes]
        _node_feature_extra: [bs * n_nodes, node_dim]
        """
        input_ids = None
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        inputs_embeds = hidden_states

        if inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
        )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, hidden_states, cache_position, past_key_values, output_attentions
        )

        bs, seq_len, _ = hidden_states.shape
        
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        for i, decoder_layer in enumerate(self.layers):
            # LM
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            if i >= self.num_hidden_layers - self.k:
                # GNN
                gnn_layer_index = i - self.num_hidden_layers + self.k
                _X = self.gnn_layers[gnn_layer_index](_X, edge_index, edge_type, _node_type, _node_feature_extra)
                _X = self.activation(_X)
                _X = F.dropout(_X, self.dropout_rate, training = self.training)

                # Exchange info between LM and GNN hidden states (Modality interaction)
                if self.info_exchange == True or (self.info_exchange == "every-other-layer" and (i - self.num_hidden_layers + self.k) % 2 == 0):
                    X = _X.view(bs, -1, _X.size(1)) # [bs, max_num_nodes, node_dim]
                    context_node_lm_feats = hidden_states[:, 0, :] # [bs, sent_dim]
                    context_node_gnn_feats = X[:, 0, :] # [bs, node_dim]
                    context_node_feats = torch.cat([context_node_lm_feats, context_node_gnn_feats], dim=1)
                    if self.sep_ie_layers:
                        _context_node_feats = self.ie_layers[gnn_layer_index](context_node_feats)
                    else:
                        _context_node_feats = self.ie_layer(context_node_feats)
                    if self.args.residual_ie == 1:
                        context_node_feats = context_node_feats + _context_node_feats
                    elif self.args.residual_ie == 2:
                        context_node_feats = self.ie_LayerNorm(context_node_feats + _context_node_feats)
                    else:
                        context_node_feats = _context_node_feats
                    context_node_lm_feats, context_node_gnn_feats = torch.split(context_node_feats, [context_node_lm_feats.size(1), context_node_gnn_feats.size(1)], dim=1)
                    hidden_states[:, 0, :] = context_node_lm_feats
                    X[:, 0, :] = context_node_gnn_feats
                    _X = X.view_as(_X)

        hidden_states = self.norm(hidden_states)
        
        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs, _X # last-layer hidden state, (all hidden states), (all attentions)
        
    def _update_causal_mask(
        self,
        attention_mask,
        input_tensor,
        cache_position,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        # if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
        #     if AttentionMaskConverter._ignore_causal_mask_sdpa(
        #         attention_mask,
        #         inputs_embeds=input_tensor,
        #         past_key_values_length=past_seen_tokens,
        #         is_training=self.training,
        #     ):
        #         return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = torch.zeros(
            (sequence_length, target_length), dtype=dtype, device=device
        )  # in original implementation - torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        # Commenting out next 2 lines to disable causal masking
        # if sequence_length != 1:
        #     causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(
            target_length, device=device
        ) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(
            input_tensor.shape[0], 1, -1, -1
        )
        if attention_mask is not None:
            causal_mask = (
                causal_mask.clone()
            )  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[
                    :, None, None, :
                ].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[
                    ..., :mask_length
                ].masked_fill(padding_mask, min_dtype)
            elif attention_mask.dim() == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0],
                    : mask_shape[1],
                    offset : mask_shape[2] + offset,
                    : mask_shape[3],
                ] = mask_slice

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        return causal_mask


    
    def get_fake_inputs(self, device="cuda:0"):
        bs = 1

        # LM input
        seq_len = 100
        hidden_states = torch.zeros([bs, seq_len, self.sent_dim]).to(device)
        attention_mask = torch.zeros([bs, 1, seq_len]).to(device)
        
        head_mask = [None] * self.num_hidden_layers

        # gnn input
        n_node = 200
        _X = torch.zeros([bs * n_node, self.concept_dim]).to(device)
        n_edges = 3
        edge_index = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(device)
        edge_type = torch.zeros(n_edges, dtype=torch.long).fill_(2).to(device)
        _node_type = torch.zeros([bs, n_node], dtype=torch.long).to(device)
        _node_type[:, 0] = 3
        _node_type = _node_type.view(-1)
        _node_feature_extra = torch.zeros([bs * n_node, self.concept_dim]).to(device)

        # these are not used
        special_nodes_mask = torch.zeros([bs, n_node], dtype=torch.long).to(device)
        special_tokens_mask = torch.zeros([bs, seq_len], dtype=torch.long).to(device)

        ## test args
        output_attentions = True
        output_hidden_states = True
        return hidden_states, attention_mask, special_tokens_mask, head_mask, _X, edge_index, edge_type, _node_type, _node_feature_extra, special_nodes_mask, output_attentions, output_hidden_states


    def check_outputs(self, outputs, _X):
        bs = 1
        seq_len = 100
        assert outputs[0].size() == (bs, seq_len, self.sent_dim)
        n_node = 200
        assert _X.size() == (bs * n_node, self.concept_dim)
        print("Success")


def test_LLaMaGAT(device):
    config, unk = LlamaConfig.from_pretrained(
        "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
        cache_dir=None, return_unused_kwargs=True,
        force_download=False,
        output_hidden_states=True,
        torch_dtype=torch.bfloat16
    )
    class Args:
        def __init__(self):
            self.residual_ie = 2
            self.fp16 = True
            self.upcast = True
            self.update_ie = True
            self.local_rank = 0
    print(unk)
    test_args = Args()
    model = LLaMaGAT(config, test_args,  sep_ie_layers=True).to(device)
    inputs = model.get_fake_inputs(device)
    outputs = model(*inputs)
    print(outputs[0][0].dtype)
    model.check_outputs(*outputs)
        

def test_RoBERTaGAT(device):
    config, _ = modeling_roberta.RobertaModel.config_class.from_pretrained(
        "roberta-large",
        cache_dir=None, return_unused_kwargs=True,
        force_download=False,
        output_hidden_states=True
    )
    model = RoBERTaGAT(config, sep_ie_layers=True).to(device)
    inputs = model.get_fake_inputs(device)
    outputs = model(*inputs)
    
    model.check_outputs(*outputs)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(name)s:%(funcName)s():%(lineno)d] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    utils.print_cuda_info()
    free_gpus = utils.select_free_gpus()
    device = torch.device("cuda:{}".format(free_gpus[0]))

    # test_RoBERTaGAT(device)

    # test_TextKGMessagePassing(device)

    # test_LMGNN(device)

    test_DRAGON(device)
