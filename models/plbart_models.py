import torch.nn as nn
import torch
import os
import torch.nn.functional as F
import numpy as np
from codeattack.models.wrappers import ModelWrapper
from torch.nn import CrossEntropyLoss, MSELoss
import sentencepiece as spm
from fairseq.models.bart import BARTModel
import argparse
from fairseq.models.fairseq_encoder import EncoderOut


class Dictionary():
    def __init__(self, vocab_size, pad_id) -> None:
        self.vocab_size = vocab_size
        self.pad_id = pad_id

    def __len__(self):
        return self.vocab_size
    
    def pad(self):
        return self.pad_id

class Task():
    def __init__(self, vocab_size, pad_id) -> None:
        self.source_dictionary = Dictionary(vocab_size, pad_id)
        self.target_dictionary = self.source_dictionary

def build_mbart(args):
    task = Task(args.vocab_size, args.pad_id)
    model = BARTModel.build_model(args, task)
    return model

def build_wrapper(args):
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer_name)
    config = argparse.Namespace(activation_fn='gelu', adam_betas='(0.9, 0.98)', adam_eps=1e-08, adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0, add_prev_output_tokens=True, all_gather_list_size=16384, arch='mbart_base', attention_dropout=0.1, batch_size=4, batch_size_valid=4, best_checkpoint_metric='accuracy', bf16=False, bpe=None, broadcast_buffers=False, bucket_cap_mb=25, checkpoint_shard_count=1, checkpoint_suffix='', classification_head_name='sentence_classification_head', clip_norm=1.0, cpu=False, criterion='sentence_prediction', cross_self_attention=False, curriculum=0, data='/home/zzr/CodeStudy/Defect-detection/plbart/processed/data-bin', data_buffer_size=10, dataset_impl=None, ddp_backend='no_c10d', decoder_attention_heads=12, decoder_embed_dim=768, decoder_embed_path=None, decoder_ffn_embed_dim=3072, decoder_input_dim=768, decoder_layerdrop=0, decoder_layers=6, decoder_layers_to_keep=None, decoder_learned_pos=True, decoder_normalize_before=False, decoder_output_dim=768, device_id=0, disable_validation=False, distributed_backend='nccl', distributed_init_method=None, distributed_no_spawn=False, distributed_num_procs=0, distributed_port=-1, distributed_rank=0, distributed_world_size=1, distributed_wrapper='DDP', dropout=0.1, empty_cache_freq=0, encoder_attention_heads=12, encoder_embed_dim=768, encoder_embed_path=None, encoder_ffn_embed_dim=3072, encoder_layerdrop=0, encoder_layers=6, encoder_layers_to_keep=None, encoder_learned_pos=True, encoder_normalize_before=False, end_learning_rate=0.0, fast_stat_sync=False, find_unused_parameters=True, finetune_from_model=None, fix_batches_to_gpus=False, fixed_validation_seed=None, force_anneal=None, fp16=False, fp16_init_scale=128, fp16_no_flatten_grads=False, fp16_scale_tolerance=0.0, fp16_scale_window=None, gen_subset='test', init_token=0, keep_best_checkpoints=-1, keep_interval_updates=-1, keep_last_epochs=-1, langs='java,python,en_XX', layernorm_embedding=True, local_rank=0, localsgd_frequency=3, log_format='json', log_interval=10, lr=[5e-05], lr_scheduler='polynomial_decay', max_epoch=5, max_positions=512, max_source_positions=1024, max_target_positions=1024, max_tokens=2048, max_tokens_valid=2048, max_update=15000, maximize_best_checkpoint_metric=True, memory_efficient_bf16=False, memory_efficient_fp16=False, min_loss_scale=0.0001, min_lr=-1.0, model_parallel_size=1, no_cross_attention=False, no_epoch_checkpoints=True, no_last_checkpoints=False, no_progress_bar=False, no_save=False, no_save_optimizer_state=False, no_scale_embedding=False, no_seed_provided=False, no_shuffle=False, no_token_positional_embeddings=False, nprocs_per_node=1, num_classes=2, num_shards=1, num_workers=1, optimizer='adam', optimizer_overrides='{}', patience=-1, pipeline_balance=None, pipeline_checkpoint='never', pipeline_chunks=0, pipeline_decoder_balance=None, pipeline_decoder_devices=None, pipeline_devices=None, pipeline_encoder_balance=None, pipeline_encoder_devices=None, pipeline_model_parallel=False, pooler_activation_fn='tanh', pooler_dropout=0.0, power=1.0, profile=False, quant_noise_pq=0, quant_noise_pq_max_source_length=8, quant_noise_scalar=0, quantization_config_path=None, regression_target=False, relu_dropout=0.0, required_batch_size_multiple=1, required_seq_len_multiple=1, reset_dataloader=True, reset_lr_scheduler=False, reset_meters=True, reset_optimizer=True, restore_file='/data2/cg/CodeStudy/PLBART/pretrain/checkpoint_11_100000.pt', save_dir='/home/zzr/CodeStudy/Defect-detection/plbart/devign', save_interval=1, save_interval_updates=0, scoring='bleu', seed=1234, sentence_avg=False, separator_token=None, shard_id=0, share_all_embeddings=True, share_decoder_input_output_embed=True, shorten_data_split_list='', shorten_method='truncate', skip_invalid_size_inputs_valid_test=False, slowmo_algorithm='LocalSGD', slowmo_momentum=None, stop_time_hours=0, task='plbart_sentence_prediction', tensorboard_logdir=None, threshold_loss_scale=None, tokenizer=None, total_num_update=1000000, tpu=False, train_subset='train', update_freq=[4], use_bmuf=False, use_old_adam=False, user_dir='/home/zzr/CodeStudy/PLBART/source', valid_subset='valid', validate_after_updates=0, validate_interval=1, validate_interval_updates=0, warmup_updates=500, weight_decay=0.0, zero_sharding='none')
    config.num_labels = 1
    config.vocab_size = tokenizer.GetPieceSize() + 5
    config.pad_id = 1
    assert config.vocab_size == 50005
    model = build_mbart(config)   

    if args.task == "clone_bcb":
        model = CloneDetectionBCBModel(model, config, tokenizer, args)
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.save_dir, '{}/{}'.format(args.model, checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))
        model_wrapper = CloneDetectionBCBModelWrapper(model, tokenizer, args)

    elif args.task == "clone_poj":
        model = CloneDetectionPOJModel(model, config, tokenizer, args)
        checkpoint_prefix = 'checkpoint-best-map/model.bin'
        output_dir = os.path.join(args.save_dir, '{}/{}'.format(args.model, checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))
        model_wrapper = CloneDetectionPOJModelWrapper(model, tokenizer, args)

    elif args.task == "defect":
        model = DefectDetectionModel(model, config, tokenizer, args)
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        output_dir = os.path.join(args.save_dir, '{}/{}'.format(args.model, checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))

        model_wrapper = DefectDetectionModelWrapper(model, tokenizer, args)
    
    elif args.task == "search":
        model = SearchModel(model, config, tokenizer, args)
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.save_dir, '{}/{}'.format(args.model, checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))

        model_wrapper = SearchModelWrapper(model, tokenizer, args)
       
    elif args.task == "summarization":
        model=Seq2Seq(encoder=model.encoder,decoder=model.decoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=0,eos_id=2)

        checkpoint_prefix = 'checkpoint-best-bleu/pytorch_model.bin'
        output_dir = os.path.join(args.save_dir, '{}/{}'.format(args.model, checkpoint_prefix)) 
        model.load_state_dict(torch.load(output_dir))
        model_wrapper = SummarizationModelWrapper(model, tokenizer, args)

    else:
        print("Not Such Task: {}.".format(args.task))

    return model_wrapper

###############################################################
# Clone Detection BigCloneBench
############################################################### 
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.encoder_embed_dim*2, config.encoder_embed_dim)
        self.dropout = nn.Dropout(config.attention_dropout)
        self.out_proj = nn.Linear(config.encoder_embed_dim, 2)

    def forward(self, x, **kwargs):
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class CloneDetectionBCBModel(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(CloneDetectionBCBModel, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.args=args
        
    def forward(self, input_ids=None,prev_tokens_ids=None,lengths=None): 
        assert lengths.size(-1) % 2 == 0
        input_ids=input_ids.view(-1,self.args.max_source_length)
        prev_tokens_ids=prev_tokens_ids.view(-1,self.args.max_source_length)
        lengths=lengths.view(-1,lengths.size(-1)//2)-1
        lengths=lengths.squeeze(-1)

        outputs=self.encoder(src_tokens=input_ids,src_lengths=lengths,prev_output_tokens=prev_tokens_ids,features_only=True)[0] # L * 2B * D
        outputs=outputs[range(input_ids.size(0)),lengths,:] # 2B * D
        logits=self.classifier(outputs)
        prob=F.softmax(logits)
        return prob
    
    def get_input_embeddings(self):
        return self.encoder.roberta.embeddings.word_embeddings

class CloneDetectionBCBModelWrapper(ModelWrapper):
    """Loads a PyTorch model (`nn.Module`) and tokenizer.

    Args:
        model (torch.nn.Module): PyTorch model
        tokenizer: tokenizer whose output can be packed as a tensor and passed to the model.
            No type requirement, but most have `tokenizer` method that accepts list of strings.
    """

    def __init__(self, model, tokenizer, args):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"PyTorch model must be torch.nn.Module, got type {type(model)}"
            )

        self.model = model
        self.tokenizer = tokenizer
        self.max_source_length = args.max_source_length
        self.args = args

    def to(self, device):
        self.model.to(device)
    
    def get_ids(self, source, tokens_to_replace=None):
        source = ' '.join(source.split())
        source_ids = self.tokenizer.encode(source, out_type=int)
        source_ids = [0] + source_ids[: (self.args.max_source_length - 2)] + [2]
        source_lengths = len(source_ids)
        padding_length = self.args.max_source_length - source_lengths
        source_ids += [1]*padding_length
        prev_ids = np.roll(source_ids, 1)
        prev_ids[0] = 0
        prev_ids = prev_ids.tolist()

        if tokens_to_replace is not None:
            indices = []
            for token in tokens_to_replace:
                indices.append([i for i, x in enumerate(source_ids) if x == token])
            return source_ids, indices
        return source_ids, prev_ids, source_lengths

    def __call__(self, text_input_list, batch_size=32):
        code_ids, prev_ids, lengths = [], [], []
        model_device = next(self.model.parameters()).device
        codes0 = [self.get_ids(text["adv1"]) for text in text_input_list]
        codes1 = [self.get_ids(text["adv2"]) for text in text_input_list]
        for code0, code1 in zip(codes0, codes1):
            code_ids += [code0[0] + code1[0]]
            prev_ids += [code0[1] + code1[1]]
            lengths += [[code0[2], code1[2]]]

        code_ids = torch.tensor(code_ids).to(model_device)
        prev_ids = torch.tensor(prev_ids).to(model_device)
        lengths = torch.tensor(lengths).to(model_device)

        with torch.no_grad():
            outputs = self.model(input_ids=code_ids,prev_tokens_ids=prev_ids,lengths=lengths)

        return outputs
    
    def get_grad(self, attack_text, indices_to_replace):
        """Get gradient of loss with respect to input tokens.

        Args:
            text_input (str): input string
            loss_fn (torch.nn.Module): loss function. Default is `torch.nn.CrossEntropyLoss`
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """

        if not hasattr(self.model, "get_input_embeddings"):
            raise AttributeError(
                f"{type(self.model)} must have method `get_input_embeddings` that returns `torch.nn.Embedding` object that represents input embedding layer"
            )

        self.model.train()

        text_input = attack_text.tokenizer_input
        label = attack_text.ground_truth_output
        tokens_to_replace = []
        for indice in indices_to_replace:
            token_to_replace = self.tokenizer.encode(attack_text.words[indice], 
                                                     add_special_tokens=False,
                                                     max_length=1
                                                     )
            tokens_to_replace += [token_to_replace[0]]

        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        ids, ids_to_replace = self.get_ids(text_input[1], tokens_to_replace=tokens_to_replace)
        ids = torch.tensor([ids]).to(model_device)
        predictions = self.model(input_ids=ids)
        labels = torch.tensor([label]).to(model_device)
        labels=labels.float()
        loss=torch.log(predictions[:,0]+1e-10)*labels+torch.log((1-predictions)[:,0]+1e-10)*(1-labels)
        loss=-loss.mean()
            
        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": ids[0].tolist(), "gradient": grad, "ids_to_replace":ids_to_replace}

        return output

###############################################################
# Clone Detection POJ
############################################################### 
class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class CloneDetectionPOJModel(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(CloneDetectionPOJModel, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        self.pooler = Pooler(self.config.encoder_embed_dim)

    def forward(self, input_ids=None,prev_tokens_ids=None,lengths=None,
                      p_input_ids=None,p_prev_tokens_ids=None,p_lengths=None): 
        bs,_=input_ids.size()
        input_ids=torch.cat((input_ids,p_input_ids),0)
        prev_tokens_ids=torch.cat((prev_tokens_ids,p_prev_tokens_ids),0)
        lengths=torch.cat((lengths,p_lengths),0)-1
        
        vecs=self.encoder(src_tokens=input_ids,src_lengths=lengths,prev_output_tokens=prev_tokens_ids,features_only=True)[0] # L * 3B * D
        vecs=vecs[range(2*bs),lengths,:] # 3B * D
        vecs=self.pooler(vecs) # 3B * D
        vecs=vecs.split(bs,0) # B * D , B * D , B * D 

        outputs = []
        for adv, code in zip(vecs[0], vecs[1]):
            one = torch.cat((adv.unsqueeze(0), code.unsqueeze(0)), dim=0)
            outputs.append(one.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs
        
    def get_input_embeddings(self):
        return self.encoder.embeddings.word_embeddings

class CloneDetectionPOJModelWrapper(ModelWrapper):
    """Loads a PyTorch model (`nn.Module`) and tokenizer.

    Args:
        model (torch.nn.Module): PyTorch model
        tokenizer: tokenizer whose output can be packed as a tensor and passed to the model.
            No type requirement, but most have `tokenizer` method that accepts list of strings.
    """

    def __init__(self, model, tokenizer, args):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"PyTorch model must be torch.nn.Module, got type {type(model)}"
            )

        self.model = model
        self.tokenizer = tokenizer
        self.max_source_length = args.max_source_length
        self.args = args

    def to(self, device):
        self.model.to(device)
    
    def get_ids(self, source, tokens_to_replace=None):
        source = ' '.join(source.split())
        source_ids = self.tokenizer.encode(source, out_type=int)
        source_ids = [0] + source_ids[: (self.args.max_source_length - 2)] + [2]
        source_lengths = len(source_ids)
        padding_length = self.args.max_source_length - source_lengths
        source_ids += [1]*padding_length
        prev_ids = np.roll(source_ids, 1)
        prev_ids[0] = 0
        prev_ids = prev_ids.tolist()

        if tokens_to_replace is not None:
            indices = []
            for token in tokens_to_replace:
                indices.append([i for i, x in enumerate(source_ids) if x == token])
            return source_ids, indices
        return source_ids, prev_ids, source_lengths

    def __call__(self, text_input_list, batch_size=32):

        model_device = next(self.model.parameters()).device
        codes = [self.get_ids(text["code"]) for text in text_input_list]
        advs = [self.get_ids(text["adv"]) for text in text_input_list]

        code_ids = [code[0] for code in codes]
        code_prev_ids = [code[1] for code in codes]
        code_lengths = [code[2] for code in codes]
        adv_ids = [adv[0] for adv in advs]
        adv_prev_ids = [adv[1] for adv in advs]
        adv_lengths = [adv[2] for adv in advs]

        code_ids = torch.tensor(code_ids).to(model_device)
        code_prev_ids = torch.tensor(code_prev_ids).to(model_device)
        code_lengths = torch.tensor(code_lengths).to(model_device)
        adv_ids = torch.tensor(adv_ids).to(model_device)
        adv_prev_ids = torch.tensor(adv_prev_ids).to(model_device)
        adv_lengths = torch.tensor(adv_lengths).to(model_device)

        with torch.no_grad():
            outputs = self.model(input_ids=adv_ids,prev_tokens_ids=adv_prev_ids,lengths=adv_lengths,
                                 p_input_ids=code_ids,p_prev_tokens_ids=code_prev_ids,p_lengths=code_lengths)

        return outputs

    def get_grad(self, attack_text, indices_to_replace):
        """Get gradient of loss with respect to input tokens.

        Args:
            text_input (str): input string
            loss_fn (torch.nn.Module): loss function. Default is `torch.nn.CrossEntropyLoss`
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """

        if not hasattr(self.model, "get_input_embeddings"):
            raise AttributeError(
                f"{type(self.model)} must have method `get_input_embeddings` that returns `torch.nn.Embedding` object that represents input embedding layer"
            )

        self.model.train()

        text_input = attack_text.tokenizer_input
        tokens_to_replace = []
        for indice in indices_to_replace:
            token_to_replace = self.tokenizer.encode(attack_text.words[indice], 
                                                     add_special_tokens=False,
                                                     max_length=1
                                                     )
            tokens_to_replace += [token_to_replace[0]]

        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        code_ids, ids_to_replace = self.get_ids(text_input[1], tokens_to_replace=tokens_to_replace)
        nl_ids = self.get_ids(text_input[2])
        code_ids = torch.tensor([code_ids]).to(model_device)
        nl_ids = torch.tensor([nl_ids]).to(model_device)

        predictions = self.model(code_inputs=code_ids, nl_inputs=nl_ids)
        _, loss = self.model.model_loss(predictions[:,0], predictions[:,1])
            
        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": code_ids[0].tolist(), "gradient": grad, "ids_to_replace":ids_to_replace}

        return output

###############################################################
# Defect Detection
############################################################### 
class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, num_labels, hidden_dropout_prob=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[0, :, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class DefectDetectionModel(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(DefectDetectionModel, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        self.classifier = ClassificationHead(self.config.encoder_embed_dim, self.config.num_labels)


    def forward(self, input_ids=None,inputs_lengths=None,prev_tokens=None): 
        # labels : B
        outputs=self.encoder(src_tokens=input_ids,src_lengths=inputs_lengths,prev_output_tokens=prev_tokens,features_only=True)[0]
        # V * B * D
        outputs=outputs.transpose(0,1).contiguous()
        logits=self.classifier(outputs) # 4*1
        prob=F.sigmoid(logits)
        return prob
    
    def get_input_embeddings(self):
        return self.encoder.roberta.embeddings.word_embeddings

class DefectDetectionModelWrapper(ModelWrapper):
    """Loads a PyTorch model (`nn.Module`) and tokenizer.

    Args:
        model (torch.nn.Module): PyTorch model
        tokenizer: tokenizer whose output can be packed as a tensor and passed to the model.
            No type requirement, but most have `tokenizer` method that accepts list of strings.
    """

    def __init__(self, model, tokenizer, args):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"PyTorch model must be torch.nn.Module, got type {type(model)}"
            )

        self.model = model
        self.tokenizer = tokenizer
        self.max_source_length = args.max_source_length
        self.args = args

    def to(self, device):
        self.model.to(device)
    
    def get_ids(self, source, tokens_to_replace=None):
        source = ' '.join(source.split())
        source_ids = self.tokenizer.encode(source, out_type=int)
        source_ids = [0] + source_ids[: (self.args.max_source_length - 2)] + [2]
        source_lengths = len(source_ids)
        padding_length = self.args.max_source_length - source_lengths
        source_ids += [1]*padding_length
        prev_ids = np.roll(source_ids, 1)
        prev_ids[0] = 0
        prev_ids = prev_ids.tolist()

        if tokens_to_replace is not None:
            indices = []
            for token in tokens_to_replace:
                indices.append([i for i, x in enumerate(source_ids) if x == token])
            return source_ids, indices
        return source_ids, prev_ids, source_lengths

    def __call__(self, text_input_list, batch_size=32):

        model_device = next(self.model.parameters()).device
        codes = [self.get_ids(text["adv"]) for text in text_input_list]

        code_ids = [code[0] for code in codes]
        code_prev_ids = [code[1] for code in codes]
        code_lengths = [code[2] for code in codes]

        code_ids = torch.tensor(code_ids).to(model_device)
        code_prev_ids = torch.tensor(code_prev_ids).to(model_device)
        code_lengths = torch.tensor(code_lengths).to(model_device)

        with torch.no_grad():
            outputs = self.model(input_ids=code_ids,inputs_lengths=code_lengths,prev_tokens=code_prev_ids)

        return outputs
    
    def get_grad(self, attack_text, indices_to_replace):
        """Get gradient of loss with respect to input tokens.

        Args:
            text_input (str): input string
            loss_fn (torch.nn.Module): loss function. Default is `torch.nn.CrossEntropyLoss`
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """

        if not hasattr(self.model, "get_input_embeddings"):
            raise AttributeError(
                f"{type(self.model)} must have method `get_input_embeddings` that returns `torch.nn.Embedding` object that represents input embedding layer"
            )

        self.model.train()

        text_input = attack_text.tokenizer_input
        label = attack_text.ground_truth_output
        tokens_to_replace = []
        for indice in indices_to_replace:
            token_to_replace = self.tokenizer.encode(attack_text.words[indice], 
                                                     add_special_tokens=False,
                                                     max_length=1
                                                     )
            tokens_to_replace += [token_to_replace[0]]

        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        ids, ids_to_replace = self.get_ids(text_input[1], tokens_to_replace=tokens_to_replace)
        ids = torch.tensor([ids]).to(model_device)
        predictions = self.model(input_ids=ids)
        labels = torch.tensor([label]).to(model_device)
        labels=labels.float()
        loss=torch.log(predictions[:,0]+1e-10)*labels+torch.log((1-predictions)[:,0]+1e-10)*(1-labels)
        loss=-loss.mean()
            
        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": ids[0].tolist(), "gradient": grad, "ids_to_replace":ids_to_replace}

        return output

###############################################################
# Code Search
###############################################################
class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class SearchModel(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(SearchModel, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        self.pooler = Pooler(self.config.encoder_embed_dim)

    def forward(self, code_inputs,prev_code_inputs,nl_inputs,prev_nl_inputs): 
        bs=code_inputs.shape[0]
        input_ids=torch.cat((code_inputs,nl_inputs),0)
        prev_tokens_ids=torch.cat((prev_code_inputs,prev_nl_inputs),0)
        lengths = torch.ne(input_ids, 0).sum(-1) - 1
        outputs=self.encoder(src_tokens=input_ids,src_lengths=lengths,prev_output_tokens=prev_tokens_ids,features_only=True)[0] # L * B * D 
        outputs=outputs[range(2*bs),lengths,:] # 3B * D
        code_vec=outputs[:bs]
        nl_vec=outputs[bs:]
        outputs = []
        for code, nl in zip(code_vec, nl_vec):
            one = torch.cat((code.unsqueeze(0), nl.unsqueeze(0)), dim=0)
            outputs.append(one.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs
        
        # batch_loss=-(nl_vec*code_vec).sum(-1)
        # loss = torch.mean(batch_loss)
        # return batch_loss.unsqueeze(-1)

    def get_input_embeddings(self):
        return self.encoder.embeddings.word_embeddings

class SearchModelWrapper(ModelWrapper):
    """Loads a PyTorch model (`nn.Module`) and tokenizer.

    Args:
        model (torch.nn.Module): PyTorch model
        tokenizer: tokenizer whose output can be packed as a tensor and passed to the model.
            No type requirement, but most have `tokenizer` method that accepts list of strings.
    """

    def __init__(self, model, tokenizer, args):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"PyTorch model must be torch.nn.Module, got type {type(model)}"
            )

        self.model = model
        self.tokenizer = tokenizer
        self.max_source_length = args.max_source_length
        self.args = args

    def to(self, device):
        self.model.to(device)
    
    def get_ids(self, source, tokens_to_replace=None):
        source = ' '.join(source.split())
        source_ids = self.tokenizer.encode(source, out_type=int)
        source_ids = [0] + source_ids[: (self.args.max_source_length - 2)] + [2]
        source_lengths = len(source_ids)
        padding_length = self.args.max_source_length - source_lengths
        source_ids += [1]*padding_length
        prev_ids = np.roll(source_ids, 1)
        prev_ids[0] = 0
        prev_ids = prev_ids.tolist()

        if tokens_to_replace is not None:
            indices = []
            for token in tokens_to_replace:
                indices.append([i for i, x in enumerate(source_ids) if x == token])
            return source_ids, indices
        return source_ids, prev_ids, source_lengths

    def __call__(self, text_input_list, batch_size=32):

        model_device = next(self.model.parameters()).device
        codes = [self.get_ids(text["code"]) for text in text_input_list]
        nls = [self.get_ids(text["nl"]) for text in text_input_list]

        code_ids = [code[0] for code in codes]
        code_prev_ids = [code[1] for code in codes]
        code_lengths = [code[2] for code in codes]
        nl_ids = [nl[0] for nl in nls]
        nl_prev_ids = [nl[1] for nl in nls]
        nl_lengths = [nl[2] for nl in nls]

        code_ids = torch.tensor(code_ids).to(model_device)
        code_prev_ids = torch.tensor(code_prev_ids).to(model_device)
        code_lengths = torch.tensor(code_lengths).to(model_device)
        nl_ids = torch.tensor(nl_ids).to(model_device)
        nl_prev_ids = torch.tensor(nl_prev_ids).to(model_device)
        nl_lengths = torch.tensor(nl_lengths).to(model_device)

        with torch.no_grad():
            outputs = self.model(code_inputs=code_ids,prev_code_inputs=code_prev_ids,nl_inputs=nl_ids,prev_nl_inputs=nl_prev_ids)

        return outputs

    def get_grad(self, attack_text, indices_to_replace):
        """Get gradient of loss with respect to input tokens.

        Args:
            text_input (str): input string
            loss_fn (torch.nn.Module): loss function. Default is `torch.nn.CrossEntropyLoss`
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """

        if not hasattr(self.model, "get_input_embeddings"):
            raise AttributeError(
                f"{type(self.model)} must have method `get_input_embeddings` that returns `torch.nn.Embedding` object that represents input embedding layer"
            )

        self.model.train()

        text_input = attack_text.tokenizer_input
        tokens_to_replace = []
        for indice in indices_to_replace:
            token_to_replace = self.tokenizer.encode(attack_text.words[indice], 
                                                     add_special_tokens=False,
                                                     max_length=1
                                                     )
            tokens_to_replace += [token_to_replace[0]]

        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        code_ids, ids_to_replace = self.get_ids(text_input[1], tokens_to_replace=tokens_to_replace)
        nl_ids = self.get_ids(text_input[2])
        code_ids = torch.tensor([code_ids]).to(model_device)
        nl_ids = torch.tensor([nl_ids]).to(model_device)

        predictions = self.model(code_inputs=code_ids, nl_inputs=nl_ids)
        _, loss = self.model.model_loss(predictions[:,0], predictions[:,1])
            
        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": code_ids[0].tolist(), "gradient": grad, "ids_to_replace":ids_to_replace}

        return output

###############################################################
# Code Summarization
############################################################### 
def ids_to_strs(Y, sp):
    ids = []
    eos_id = sp.PieceToId("</s>")
    pad_id = sp.PieceToId("<pad>")
    for idx in Y:
        ids.append(int(idx))
        if int(idx) == eos_id or int(idx) == pad_id:
            break
    return sp.DecodeIds(ids)

class SummarizationModelWrapper(ModelWrapper):
    """Loads a PyTorch model (`nn.Module`) and tokenizer.

    Args:
        model (torch.nn.Module): PyTorch model
        tokenizer: tokenizer whose output can be packed as a tensor and passed to the model.
            No type requirement, but most have `tokenizer` method that accepts list of strings.
    """

    def __init__(self, model, tokenizer, args):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"PyTorch model must be torch.nn.Module, got type {type(model)}"
            )

        self.model = model
        self.tokenizer = tokenizer
        self.max_source_length = args.max_source_length
        self.max_target_length = args.max_target_length
        self.args = args

    def to(self, device):
        self.model.to(device)
    
    def get_ids(self, source, max_length=-1, tokens_to_replace=None):
        max_length = max_length if max_length>0 else self.max_source_length
        source = ' '.join(source.split())
        source_ids = self.tokenizer.encode(source, out_type=int)
        source_ids = [0] + source_ids[: (max_length - 2)] + [2]
        source_lengths = len(source_ids)
        padding_length = max_length - source_lengths
        source_ids += [1]*padding_length
        prev_ids = np.roll(source_ids, 1)
        prev_ids[0] = 0
        prev_ids = prev_ids.tolist()

        if tokens_to_replace is not None:
            indices = []
            for token in tokens_to_replace:
                indices.append([i for i, x in enumerate(source_ids) if x == token])
            return source_ids, indices
        return source_ids, prev_ids, source_lengths

    def decode(self, outputs):
        preds = []
        for pred in outputs:
            if 0 in pred:
                pred=pred[:pred.index(0)]
            pred = ids_to_strs(pred, self.tokenizer)
            preds += [pred]
        return preds

    def __call__(self, text_input_list, batch_size=32):

        model_device = next(self.model.parameters()).device
        src = [self.get_ids(text["code"], max_length=self.max_source_length) for text in text_input_list]
        tgt = [self.get_ids(text["nl"], max_length=self.max_target_length) for text in text_input_list]

        src_ids = [s[0] for s in src]
        source_mask = [[True]*s[2] + [False]*(self.max_source_length-s[2]) for s in src]
        tgt_ids = [t[0] for t in tgt]
        # target_mask = [[True]*t[2] + [False]*(self.max_target_length-t[2]) for t in tgt]

        src_ids = torch.tensor(src_ids).to(model_device)
        tgt_ids = torch.tensor(tgt_ids).to(model_device)
        source_mask = torch.tensor(source_mask).to(model_device)

        with torch.no_grad():
            outputs = self.model.generate(source_ids=src_ids,source_mask=source_mask,target_ids=tgt_ids)
        
        outputs = outputs.cpu().numpy().tolist()
        outputs = self.decode(outputs)
            
        return outputs
        # return [self.tokenizer.decode(ids) for ids in outputs]

    def get_grad(self, attack_text, indices_to_replace):
        """Get gradient of loss with respect to input tokens.

        Args:
            text_input (str): input string
            loss_fn (torch.nn.Module): loss function. Default is `torch.nn.CrossEntropyLoss`
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """

        if not hasattr(self.model, "get_input_embeddings"):
            raise AttributeError(
                f"{type(self.model)} must have method `get_input_embeddings` that returns `torch.nn.Embedding` object that represents input embedding layer"
            )

        self.model.train()

        text_input = attack_text.tokenizer_input
        tokens_to_replace = []
        for indice in indices_to_replace:
            token_to_replace = self.tokenizer.encode(attack_text.words[indice], 
                                                     add_special_tokens=False,
                                                     max_length=1
                                                     )
            tokens_to_replace += [token_to_replace[0]]

        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        src_ids, ids_to_replace = self.get_ids(text_input[1], max_length=self.max_source_length, tokens_to_replace=tokens_to_replace)
        tgt_ids = self.get_ids(text_input[2], max_length=self.max_target_length)

        src_ids = torch.tensor([src_ids]).to(model_device)
        tgt_ids = torch.tensor([tgt_ids]).to(model_device)

        lm_logits = self.model(source_ids=src_ids, target_ids=tgt_ids)
        _, loss = self.model.model_loss(lm_logits, tgt_ids)
            
        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": src_ids[0].tolist(), "gradient": grad, "ids_to_replace":ids_to_replace}

        return output

class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """
    def __init__(self, encoder,decoder,config,beam_size=None,max_length=None,sos_id=None,eos_id=None,pad_id=1):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder=decoder
        self.config=config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.lsm = nn.LogSoftmax(dim=-1)
        
        self.beam_size=beam_size
        self.max_length=max_length
        self.sos_id=sos_id
        self.eos_id=eos_id
        
    def get_input_embeddings(self):
        return self.encoder.embeddings.word_embeddings      

    def forward(self, source_ids=None,source_mask=None,target_ids=None):   
        encoder_output = self.encoder(source_ids, src_lengths=source_mask.sum(dim=1)) # L * B * D
        out = self.decoder(target_ids,encoder_out=encoder_output,features_only=False) # B * L * V
        lm_logits = out[0] # B * L * V
        return lm_logits

    def generate(self, source_ids=None,source_mask=None,target_ids=None):
        encoder_output = self.encoder(source_ids, src_lengths=source_mask.sum(dim=1)) # L * B * D
        out = self.decoder(target_ids,encoder_out=encoder_output,features_only=False) # B * L * V
        lm_logits = out[0] # B * L * V
        
        if self.beam_size == 0:
            #Predict 
            preds=[]       
            out = self.lsm(lm_logits).data # B * L * V
            max_probs, preds = torch.max(out, dim=2)
        else:
            #Predict 
            preds=[]       
            zero=torch.cuda.LongTensor(1).fill_(0)     # [0]
            encoder_out = encoder_output.encoder_out
            encoder_padding_mask = encoder_output.encoder_padding_mask
            for i in range(source_ids.shape[0]): # B
                context = encoder_out[:,i:i+1] # L * 1 * D
                context_mask = encoder_padding_mask[i:i+1,:] # 1 * L
                beam = Beam(self.beam_size,self.sos_id,self.eos_id)
                input_ids=beam.getCurrentState() # beam * 1
                context=context.repeat(1, self.beam_size,1) # L * beam * D
                context_mask=context_mask.repeat(self.beam_size,1) # beam * L
                context = EncoderOut(
                                encoder_out=context,  # T x B x C
                                encoder_padding_mask=context_mask,  # B x T
                                encoder_embedding=None,
                                encoder_states=None,
                                src_tokens=None,
                                src_lengths=None
                            )
                for _ in range(self.max_length): 
                    if beam.done():
                        break
                    # attn_mask=-1e4 *(1-self.bias[:input_ids.shape[1],:input_ids.shape[1]])
                    # tgt_embeddings = self.encoder.embeddings(input_ids).permute([1,0,2]).contiguous() # 1 * beam * D
                    out = self.decoder(input_ids,encoder_out=context,features_only=False) # 1 * beam * D
                    # out = torch.tanh(self.dense(out))
                    # hidden_states=out.permute([1,0,2]).contiguous()[:,-1,:] # beam * D
                    lm_logits = out[0][:,-1,:].squeeze(1)
                    out = self.lsm(lm_logits).data # beam * V
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin())) # beam * 1
                    input_ids=torch.cat((input_ids,beam.getCurrentState()),-1) # beam * 2
                hyp= beam.getHyp(beam.getFinal())
                pred=beam.buildTargetTokens(hyp)[:self.beam_size]
                pred=[torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred] # beam * l
                preds.append(pred[0]) # 1 * l
                
            preds=torch.cat(preds,0)                         # B * l
        return preds


class Beam(object):
    def __init__(self, size,sos,eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk) # beam * V

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1) # beam*V
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True) # beam， beam

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))


        for i in range(self.nextYs[-1].size(0)): # beam
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >=self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i)) 
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps
    
    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok==self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
        
