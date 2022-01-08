import os
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

from codeattack.models.wrappers import ModelWrapper
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification

_newline_regex = re.compile(r"\n")
_whitespace_regex = re.compile(r"[ \t\n]+")

def build_wrapper(args):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(args.tokenizer_name)
    pad_id = tokenizer.PieceToId("[PAD]")

    if args.task == "clone_bcb":
        model = BCBEncoder(n_tokens=tokenizer.GetPieceSize(), pad_id=pad_id)
        model = CloneDetectionBCBModel(model, None, tokenizer, args)
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.save_dir, '{}/{}'.format(args.model, checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))
        model_wrapper = CloneDetectionBCBModelWrapper(model, tokenizer, args)

    elif args.task == "clone_poj":
        model = CodeModel(n_tokens=tokenizer.GetPieceSize(), pad_id=pad_id)
        model = CloneDetectionPOJModel(model, None, tokenizer, args)
        checkpoint_prefix = 'checkpoint-best-map/model.bin'
        output_dir = os.path.join(args.save_dir, '{}/{}'.format(args.model, checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))
        model_wrapper = CloneDetectionPOJModelWrapper(model, tokenizer, args)

    elif args.task == "defect":
        model = SentenceClassfierTransformer(
                    n_tokens=tokenizer.GetPieceSize(),
                    pad_id=pad_id)
        model = DefectDetectionModel(model, None, tokenizer, args)
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        output_dir = os.path.join(args.save_dir, '{}/{}'.format(args.model, checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))

        model_wrapper = DefectDetectionModelWrapper(model, tokenizer, args)
    
    elif args.task == "search":
        model = CodeModel(n_tokens=tokenizer.GetPieceSize(), pad_id=pad_id)
        model = SearchModel(model, None, tokenizer, args)
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.save_dir, '{}/{}'.format(args.model, checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))

        model_wrapper = SearchModelWrapper(model, tokenizer, args)
       
    elif args.task == "summarization":
        n_tokens = tokenizer.GetPieceSize()
        encoder = CodeEncoder(
                    n_tokens, d_model=512, d_rep=128, n_head=8, n_encoder_layers=6, d_ff=2048, dropout=0.1, activation="relu", norm=True, pad_id=pad_id, project=False
                )   
        config = encoder.config
        decoder_layer = nn.TransformerDecoderLayer(d_model=config["d_model"], nhead=8)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                    beam_size=args.beam_size,max_length=args.max_target_length,
                    sos_id=tokenizer.PieceToId("<s>"),eos_id=tokenizer.PieceToId("</s>"),
                    pad_id=pad_id)

        checkpoint_prefix = 'checkpoint-best-bleu/pytorch_model.bin'
        output_dir = os.path.join(args.save_dir, '{}/{}'.format(args.model, checkpoint_prefix)) 
        model.load_state_dict(torch.load(output_dir))
        model_wrapper = SummarizationModelWrapper(model, tokenizer, args)

    else:
        print("Not Such Task: {}.".format(args.task))

    return model_wrapper

class PositionalEncoding(nn.Module):
    """From https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""

    def __init__(self, d_model, dropout=0.1, max_len=9000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

    def _load_from_state_dict(self, *args):
        print("PositionalEncoding: doing nothing on call to _load_from_state_dict")

class CodeEncoder(nn.Module):
    def __init__(
        self,
        n_tokens,
        d_model=512,
        d_rep=256,
        n_head=8,
        n_encoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        activation="relu",
        norm=True,
        pad_id=None,
        project=False,
    ):
        super().__init__()
        self.config = {k: v for k, v in locals().items() if k != "self"}
        self.embedding = nn.Embedding(n_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=9000)
        norm_fn = nn.LayerNorm(d_model) if norm else None
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, activation)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers, norm=norm_fn)
        if project:
            self.project_layer = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_rep))
        # NOTE: We use the default PyTorch intialization, so no need to reset parameters.

    def forward(self, input_ids=None, inputs_embeds=None, lengths=None, src_key_padding_mask=None, no_project_override=False):
        if inputs_embeds == None:
            src_emb = self.embedding(input_ids).transpose(0, 1) * math.sqrt(self.config["d_model"])
        else:
            src_emb = inputs_embeds
        src_emb = self.pos_encoder(src_emb)
        if src_key_padding_mask is not None:
            pass
        elif input_ids is not None and self.config["pad_id"] is not None:
            src_key_padding_mask = input_ids == self.config["pad_id"]
        else:
            src_key_padding_mask = None
        out = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)  # TxBxD

        if not no_project_override and self.config["project"]:
            return self.project(out)

        return out, None

    def project(self, out, h_n=None):
        assert self.config["project"]
        assert h_n is None  # second argument for compatibility with CodeEncoderLSTM
        # NOTE: This computes a mean pool of the token representations across ALL tokens,
        # including padding from uneven lengths in the batch.
        return self.project_layer(out.mean(dim=0))

def normalize_program(fn: str):
    if not isinstance(fn, (str, bytes)):
        print(f"normalize_program got non-str: {type(fn)}, {fn}")
    fn = _newline_regex.sub(r" [EOL]", fn)
    fn = _whitespace_regex.sub(" ", fn)
    return fn

def EncodeAsIds(sp, alpha, prog):
    # Encode as ids with sentencepiece
    if alpha:
        # using subword regularization: https://arxiv.org/pdf/1804.10959.pdf
        # NOTE: what is the second argument here (-1)?
        return sp.SampleEncodeAsIds(prog, -1, alpha)

    # using the best decoding
    return sp.EncodeAsIds(prog)
###############################################################
# Clone Detection BigCloneBench
############################################################### 
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(hidden_size*2, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, 2)

    def forward(self, x, **kwargs):
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class BCBEncoder(nn.Module):
    def __init__(
        self,
        n_tokens,
        d_model=512,
        d_rep=128,
        n_head=8,
        n_encoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        activation="relu",
        norm=True,
        pad_id=None,
        encoder_type="transformer",
    ):
        super(BCBEncoder, self).__init__()
        assert norm
        assert pad_id is not None
        self.config = {k: v for k, v in locals().items() if k != "self"}
        self.d_model = d_model
        # Encoder and output for type prediction
        assert encoder_type in ["transformer", "lstm"]
        if encoder_type == "transformer":
            self.encoder = CodeEncoder(
                n_tokens, d_model, d_rep, n_head, n_encoder_layers, d_ff, dropout, activation, norm, pad_id, project=False
            )
        self.pooler = RobertaClassificationHead(d_model, dropout)

    def forward(self, src_tok_ids, lengths=None):
        r"""
        Arguments:
            src_tok_ids: [B, L] long tensor
            output_attention: [B, L, L] float tensor
        """
        # Encode
        memory, _ = self.encoder(src_tok_ids, lengths)  # LxBxD
        x = memory[0, :, :] # BxD
        x = self.pooler(x)

        return (memory, x)
        
class CloneDetectionBCBModel(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(CloneDetectionBCBModel, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        
    def forward(self, input_ids=None): 
        input_ids=input_ids.view(-1,self.args.max_source_length)
        logits = self.encoder(src_tok_ids=input_ids)[1]
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
        source = normalize_program(source)
        source_ids =  EncodeAsIds(self.tokenizer, 0, source)
        source_ids = [self.tokenizer.PieceToId("<s>")] + source_ids[: (self.args.max_source_length - 2)] + [self.tokenizer.PieceToId("</s>")]
        padding_length = self.args.max_source_length - len(source_ids)
        source_ids += [self.tokenizer.PieceToId("<pad>")]*padding_length

        if tokens_to_replace is not None:
            indices = []
            for token in tokens_to_replace:
                indices.append([i for i, x in enumerate(source_ids) if x == token])
            return source_ids, indices
        return source_ids

    def __call__(self, text_input_list, batch_size=32):
        code_ids = []
        model_device = next(self.model.parameters()).device
        code0_ids = [self.get_ids(text[0]) for text in text_input_list]
        code1_ids = [self.get_ids(text[1]) for text in text_input_list]
        for code0, code1 in zip(code0_ids, code1_ids):
            code_ids += [code0 + code1]
        code_ids = torch.tensor(code_ids).to(model_device)

        with torch.no_grad():
            outputs = self.model(input_ids=code_ids)

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

class CodeModel(nn.Module):
    def __init__(
        self,
        n_tokens,
        d_model=512,
        d_rep=128,
        n_head=8,
        n_encoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        activation="relu",
        norm=True,
        pad_id=None,
        encoder_type="transformer",
    ):
        super(CodeModel, self).__init__()
        assert norm
        assert pad_id is not None
        self.config = {k: v for k, v in locals().items() if k != "self"}
        self.d_model = d_model
        # Encoder and output for type prediction
        assert encoder_type in ["transformer", "lstm"]
        if encoder_type == "transformer":
            self.encoder = CodeEncoder(
                n_tokens, d_model, d_rep, n_head, n_encoder_layers, d_ff, dropout, activation, norm, pad_id, project=False
            )
        self.pooler = Pooler(d_model)

    def forward(self, src_tok_ids, lengths=None):
        r"""
        Arguments:
            src_tok_ids: [B, L] long tensor
            output_attention: [B, L, L] float tensor
        """

        # Encode
        memory, _ = self.encoder(src_tok_ids, lengths)  # LxBxD
        x = memory[0, :, :] # BxD
        x = self.pooler(x)

        return (memory, x)
class CloneDetectionPOJModel(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(CloneDetectionPOJModel, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
    
    def forward(self, input_ids=None,p_input_ids=None): 
        bs,_=input_ids.size()
        input_ids=torch.cat((input_ids,p_input_ids),0)
        
        vecs=self.encoder(input_ids)[1] # 3B * D
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
        source = normalize_program(source)
        source_ids =  EncodeAsIds(self.tokenizer, 0, source)
        source_ids = [self.tokenizer.PieceToId("<s>")] + source_ids[: (self.args.max_source_length - 2)] + [self.tokenizer.PieceToId("</s>")]
        padding_length = self.args.max_source_length - len(source_ids)
        source_ids += [self.tokenizer.PieceToId("<pad>")]*padding_length

        if tokens_to_replace is not None:
            indices = []
            for token in tokens_to_replace:
                indices.append([i for i, x in enumerate(source_ids) if x == token])
            return source_ids, indices
        return source_ids

    def __call__(self, text_input_list, batch_size=32):

        model_device = next(self.model.parameters()).device
        code_ids = [self.get_ids(text[0]) for text in text_input_list]
        adv_ids = [self.get_ids(text[1]) for text in text_input_list]

        code_ids = torch.tensor(code_ids).to(model_device)
        adv_ids = torch.tensor(adv_ids).to(model_device)

        with torch.no_grad():
            outputs = self.model(input_ids=adv_ids,p_input_ids=code_ids)

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
class SentenceClassfierTransformer(nn.Module):
    def __init__(
        self,
        n_tokens,
        d_model=512,
        d_rep=128,
        n_head=8,
        n_encoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        activation="relu",
        norm=True,
        pad_id=None,
        encoder_type="transformer",
    ):
        super(SentenceClassfierTransformer, self).__init__()
        assert norm
        assert pad_id is not None
        self.config = {k: v for k, v in locals().items() if k != "self"}
        self.d_model = d_model
        # Encoder and output for type prediction
        assert encoder_type in ["transformer", "lstm"]
        if encoder_type == "transformer":
            self.encoder = CodeEncoder(
                n_tokens, d_model, d_rep, n_head, n_encoder_layers, d_ff, dropout, activation, norm, pad_id, project=False
            )

    def forward(self, src_tok_ids, lengths=None):
        r"""
        Arguments:
            src_tok_ids: [B, L] long tensor
            output_attention: [B, L, L] float tensor
        """

        # Encode
        memory, _ = self.encoder(src_tok_ids, lengths)  # LxBxD

        return memory


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, num_labels, hidden_dropout_prob=0.5):
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
        self.classifier=ClassificationHead(encoder.d_model, 1)
        
    def forward(self, input_ids=None): 
        outputs=self.encoder(input_ids)
        logits=self.classifier(outputs)
        prob=torch.sigmoid(logits)
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
        source = normalize_program(source)
        source_ids =  EncodeAsIds(self.tokenizer, 0, source)
        source_ids = [self.tokenizer.PieceToId("<s>")] + source_ids[: (self.args.max_source_length - 2)] + [self.tokenizer.PieceToId("</s>")]
        padding_length = self.args.max_source_length - len(source_ids)
        source_ids += [self.tokenizer.PieceToId("<pad>")]*padding_length

        if tokens_to_replace is not None:
            indices = []
            for token in tokens_to_replace:
                indices.append([i for i, x in enumerate(source_ids) if x == token])
            return source_ids, indices
        return source_ids

    def __call__(self, text_input_list, batch_size=32):

        model_device = next(self.model.parameters()).device
        code_ids = [self.get_ids(text[1]) for text in text_input_list]

        code_ids = torch.tensor(code_ids).to(model_device)

        with torch.no_grad():
            outputs = self.model(input_ids=code_ids)

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
class SearchModel(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(SearchModel, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
    
    def forward(self, code_inputs,nl_inputs): 
        bs=code_inputs.shape[0]
        inputs=torch.cat((code_inputs,nl_inputs),0)
        outputs=self.encoder(inputs)[1]
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
        source = normalize_program(source)
        source_ids =  EncodeAsIds(self.tokenizer, 0, source)
        source_ids = [self.tokenizer.PieceToId("<s>")] + source_ids[: (self.args.max_source_length - 2)] + [self.tokenizer.PieceToId("</s>")]
        padding_length = self.args.max_source_length - len(source_ids)
        source_ids += [self.tokenizer.PieceToId("<pad>")]*padding_length

        if tokens_to_replace is not None:
            indices = []
            for token in tokens_to_replace:
                indices.append([i for i, x in enumerate(source_ids) if x == token])
            return source_ids, indices
        return source_ids

    def __call__(self, text_input_list, batch_size=32):

        model_device = next(self.model.parameters()).device
        code_ids = [self.get_ids(text[1]) for text in text_input_list]
        nl_ids = [self.get_ids(text[2]) for text in text_input_list]

        code_ids = torch.tensor(code_ids).to(model_device)
        nl_ids = torch.tensor(nl_ids).to(model_device)

        with torch.no_grad():
            outputs = self.model(code_inputs=code_ids, nl_inputs=nl_ids)

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
        source = normalize_program(source)
        source_ids =  EncodeAsIds(self.tokenizer, 0, source)
        source_ids = [self.tokenizer.PieceToId("<s>")] + source_ids[: (max_length - 2)] + [self.tokenizer.PieceToId("</s>")]
        padding_length = max_length - len(source_ids)
        source_ids += [self.tokenizer.PieceToId("<pad>")]*padding_length

        if tokens_to_replace is not None:
            indices = []
            for token in tokens_to_replace:
                indices.append([i for i, x in enumerate(source_ids) if x == token])
            return source_ids, indices
        return source_ids
    
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
        src_ids = [self.get_ids(text[1], max_length=self.max_source_length) for text in text_input_list]
        tgt_ids = [self.get_ids(text[2], max_length=self.max_target_length) for text in text_input_list]

        src_ids = torch.tensor(src_ids).to(model_device)
        tgt_ids = torch.tensor(tgt_ids).to(model_device)

        with torch.no_grad():
            outputs = self.model.generate(source_ids=src_ids,target_ids=tgt_ids)
        
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
    def __init__(self, encoder,decoder,config,beam_size=None,max_length=None,sos_id=None,eos_id=None,pad_id=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder=decoder
        self.config=config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config["d_model"], config["d_model"])
        self.lm_head = nn.Linear(config["d_model"], config["n_tokens"], bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()
        
        self.beam_size=beam_size
        self.max_length=max_length
        self.sos_id=sos_id
        self.eos_id=eos_id
        self.pad_id=pad_id
        
    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        first_module.weight = second_module.weight
                  
    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embedding)              

    def forward(self, source_ids=None,target_ids=None):   
        source_mask = source_ids.ne(self.pad_id)
        outputs = self.encoder(source_ids) # L * B * D
        encoder_output = outputs[0] # L * B * D
        attn_mask=-1e4 *(1-self.bias[:target_ids.shape[1],:target_ids.shape[1]])
        tgt_embeddings = self.encoder.embedding(target_ids).permute([1,0,2]).contiguous() # L * B * D
        out = self.decoder(tgt_embeddings,encoder_output,tgt_mask=attn_mask,memory_key_padding_mask=(~source_mask).bool()) # L * B * D
        hidden_states = torch.tanh(self.dense(out)).permute([1,0,2]).contiguous() # B * L * D
        lm_logits = self.lm_head(hidden_states) # B * L * V
        return lm_logits

    def generate(self, source_ids=None,target_ids=None):   
        source_mask = source_ids.ne(self.pad_id)
        outputs = self.encoder(source_ids) # L * B * D
        encoder_output = outputs[0] # L * B * D
        attn_mask=-1e4 *(1-self.bias[:target_ids.shape[1],:target_ids.shape[1]])
        tgt_embeddings = self.encoder.embedding(target_ids).permute([1,0,2]).contiguous() # L * B * D
        out = self.decoder(tgt_embeddings,encoder_output,tgt_mask=attn_mask,memory_key_padding_mask=(~source_mask).bool()) # L * B * D
        hidden_states = torch.tanh(self.dense(out)).permute([1,0,2]).contiguous() # B * L * D
        lm_logits = self.lm_head(hidden_states) # B * L * V
        
        if self.beam_size == 0:
            #Predict 
            preds=[]       
            out = self.lsm(lm_logits).data # B * L * V
            max_probs, preds = torch.max(out, dim=2)
        else:
            preds=[]       
            zero=torch.cuda.LongTensor(1).fill_(0)     # [0]
            for i in range(source_ids.shape[0]): # B
                context=encoder_output[:,i:i+1] # L * 1 * D
                context_mask=source_mask[i:i+1,:] # 1 * L
                beam = Beam(self.beam_size,self.sos_id,self.eos_id)
                input_ids=beam.getCurrentState() # beam * 1
                context=context.repeat(1, self.beam_size,1) # L * beam * D
                context_mask=context_mask.repeat(self.beam_size,1) # beam * L
                for _ in range(self.max_length): 
                    if beam.done():
                        break
                    attn_mask=-1e4 *(1-self.bias[:input_ids.shape[1],:input_ids.shape[1]])
                    tgt_embeddings = self.encoder.embedding(input_ids).permute([1,0,2]).contiguous() # 1 * beam * D
                    out = self.decoder(tgt_embeddings,context,tgt_mask=attn_mask,memory_key_padding_mask=(~context_mask).bool()) # 1 * beam * D
                    out = torch.tanh(self.dense(out))
                    hidden_states=out.permute([1,0,2]).contiguous()[:,-1,:] # beam * D
                    out = self.lsm(self.lm_head(hidden_states)).data # beam * V
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
        
