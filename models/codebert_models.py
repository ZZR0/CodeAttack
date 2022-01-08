import torch.nn as nn
import torch
import os
import torch.nn.functional as F

from codeattack.models.wrappers import ModelWrapper
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification

def build_wrapper(args):

    if args.task == "clone_bcb":
        config_class, model_class, tokenizer_class = RobertaConfig, RobertaModel, RobertaTokenizer
        config = config_class.from_pretrained(args.model_name_or_path)
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, use_fast=True)
        config.num_labels=2
        model = model_class.from_pretrained(args.model_name_or_path, config=config)
        model = CloneDetectionBCBModel(model, config, tokenizer, args)
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.save_dir, '{}/{}'.format(args.model, checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))
        model_wrapper = CloneDetectionBCBModelWrapper(model, tokenizer, args)

    elif args.task == "clone_poj":
        config_class, model_class, tokenizer_class = RobertaConfig, RobertaModel, RobertaTokenizer
        config = config_class.from_pretrained(args.model_name_or_path)
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, use_fast=True)
        config.num_labels=1
        model = model_class.from_pretrained(args.model_name_or_path, config=config)
        model = CloneDetectionPOJModel(model, config, tokenizer, args)
        checkpoint_prefix = 'checkpoint-best-map/model.bin'
        output_dir = os.path.join(args.save_dir, '{}/{}'.format(args.model, checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))
        model_wrapper = CloneDetectionPOJModelWrapper(model, tokenizer, args)

    elif args.task == "defect":
        config_class, model_class, tokenizer_class = RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
        config = config_class.from_pretrained(args.model_name_or_path)
        config.num_labels=1

        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, use_fast=True)
        model = model_class.from_pretrained(args.model_name_or_path, config=config)

        model = DefectDetectionModel(model, config, tokenizer, args)
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        output_dir = os.path.join(args.save_dir, '{}/{}'.format(args.model, checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))

        model_wrapper = DefectDetectionModelWrapper(model, tokenizer, args)
    
    elif args.task == "search":
        config_class, model_class, tokenizer_class = RobertaConfig, RobertaModel, RobertaTokenizer
        config = config_class.from_pretrained(args.model_name_or_path)
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, use_fast=True)
        model = model_class.from_pretrained(args.model_name_or_path, config=config)

        model = SearchModel(model, config, tokenizer, args)
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.save_dir, '{}/{}'.format(args.model, checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))

        model_wrapper = SearchModelWrapper(model, tokenizer, args)
       
    elif args.task == "summarization":
        config_class, model_class, tokenizer_class = RobertaConfig, RobertaModel, RobertaTokenizer
        config = config_class.from_pretrained(args.model_name_or_path)
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
        encoder = model_class.from_pretrained(args.model_name_or_path, config=config)    

        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                    beam_size=args.beam_size,max_length=args.max_target_length,
                    sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)

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
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
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
    
    def model_loss(self, prob, labels):
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(prob, labels)
        return loss
        
    def forward(self, input_ids): 
        input_ids=input_ids.view(-1,self.args.max_source_length)
        outputs = self.encoder(input_ids=input_ids,attention_mask=input_ids.ne(1))[0]
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
        source_tokens=self.tokenizer.tokenize(source)[:self.max_source_length-2]
        source_tokens =[self.tokenizer.cls_token]+source_tokens+[self.tokenizer.sep_token]
        source_ids =  self.tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = self.max_source_length - len(source_ids)
        source_ids+=[self.tokenizer.pad_token_id]*padding_length

        if tokens_to_replace is not None:
            indices = []
            for token in tokens_to_replace:
                indices.append([i for i, x in enumerate(source_ids) if x == token])
            return source_ids, indices
        return source_ids

    def __call__(self, text_input_list, batch_size=32):
        code_ids = []
        model_device = next(self.model.parameters()).device
        code0_ids = [self.get_ids(text["adv1"]) for text in text_input_list]
        code1_ids = [self.get_ids(text["adv2"]) for text in text_input_list]
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
class CloneDetectionPOJModel(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(CloneDetectionPOJModel, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
    
    # B * D
    # def model_loss(self, code_vec, nl_vec):
    #     batch_loss=-(nl_vec*code_vec).sum(-1)
    #     loss = torch.mean(batch_loss)
    #     return batch_loss, loss

    def forward(self, input_ids=None,p_input_ids=None): 
        bs,_=input_ids.size()
        input_ids=torch.cat((input_ids,p_input_ids),0)
        
        vecs=self.encoder(input_ids,attention_mask=input_ids.ne(1))[1] # 2B * D
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
        source_tokens=self.tokenizer.tokenize(source)[:self.max_source_length-2]
        source_tokens =[self.tokenizer.cls_token]+source_tokens+[self.tokenizer.sep_token]
        source_ids =  self.tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = self.max_source_length - len(source_ids)
        source_ids+=[self.tokenizer.pad_token_id]*padding_length

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
            outputs = self.model(input_ids=adv_ids, p_input_ids=code_ids)

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
class DefectDetectionModel(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(DefectDetectionModel, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        
    def model_loss(self, prob, labels):
        batch_wish_loss=-torch.log(prob[:,0]+1e-10)*labels-torch.log((1-prob)[:,0]+1e-10)*(1-labels)
        mean_loss = batch_wish_loss.mean()
        return batch_wish_loss, mean_loss

    def forward(self, input_ids): 
        # labels : B
        attention_mask = input_ids.ne(1)
        outputs = self.encoder(input_ids=input_ids,attention_mask=attention_mask)[0]
        logits=outputs # 4*1
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
        source_tokens=self.tokenizer.tokenize(source)[:self.max_source_length-2]
        source_tokens =[self.tokenizer.cls_token]+source_tokens+[self.tokenizer.sep_token]
        source_ids =  self.tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = self.max_source_length - len(source_ids)
        source_ids+=[self.tokenizer.pad_token_id]*padding_length

        if tokens_to_replace is not None:
            indices = []
            for token in tokens_to_replace:
                indices.append([i for i, x in enumerate(source_ids) if x == token])
            return source_ids, indices
        return source_ids

    def __call__(self, text_input_list, batch_size=32):

        model_device = next(self.model.parameters()).device
        code_ids = [self.get_ids(text["adv"]) for text in text_input_list]

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
    
    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight
                  
    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)      
    
    # B * D
    def model_loss(self, code_vec, nl_vec):
        batch_loss=-(nl_vec*code_vec).sum(-1)
        loss = torch.mean(batch_loss)
        return batch_loss, loss

    # B * D
    # def model_loss(self, code_vec, nl_vec):
    #     bs=code_vec.shape[0]
    #     scores=(nl_vec[:,None,:]*code_vec[None,:,:]).sum(-1)
    #     loss_fct = CrossEntropyLoss(reduction="none")
    #     batch_loss = loss_fct(scores, torch.arange(bs, device=scores.device))
    #     loss = torch.mean(batch_loss)
    #     return batch_loss, loss

    def forward(self, code_inputs=None, nl_inputs=None): 
        inputs=torch.cat((code_inputs,nl_inputs),0)
        attention_mask=inputs.ne(1)
        bs=code_inputs.shape[0]
        inputs_embeds = None

        outputs=self.encoder(input_ids=inputs,inputs_embeds=inputs_embeds,attention_mask=attention_mask)[1]
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
        source_tokens=self.tokenizer.tokenize(source)[:self.max_source_length-2]
        source_tokens =[self.tokenizer.cls_token]+source_tokens+[self.tokenizer.sep_token]
        source_ids =  self.tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = self.max_source_length - len(source_ids)
        source_ids+=[self.tokenizer.pad_token_id]*padding_length

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
        source_tokens=self.tokenizer.tokenize(source)[:max_length-2]
        source_tokens =[self.tokenizer.cls_token]+source_tokens+[self.tokenizer.sep_token]
        source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = max_length - len(source_ids)
        source_ids+=[self.tokenizer.pad_token_id]*padding_length

        if tokens_to_replace is not None:
            indices = []
            for token in tokens_to_replace:
                indices.append([i for i, x in enumerate(source_ids) if x == token])
            return source_ids, indices

        return source_ids
    
    def decode(self, outputs):
        preds = []
        for pred in outputs:
            pred = self.tokenizer.decode(pred,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            preds += [pred]
        return preds


    def __call__(self, text_input_list, batch_size=32):

        model_device = next(self.model.parameters()).device
        src_ids = [self.get_ids(text["adv"], max_length=self.max_source_length) for text in text_input_list]
        tgt_ids = [self.get_ids(text["nl"], max_length=self.max_target_length) for text in text_input_list]

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
    def __init__(self, encoder,decoder,config,beam_size=None,max_length=None,sos_id=None,eos_id=None,pad_id=1):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder=decoder
        self.config=config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()
        
        self.beam_size=beam_size
        self.max_length=max_length
        self.sos_id=sos_id
        self.eos_id=eos_id
        self.pad_id=pad_id
        
    def get_input_embeddings(self):
        return self.encoder.embeddings.word_embeddings

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight
                  
    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)        

    def model_loss(self, lm_logits, target_ids,target_mask=None):
        # Shift so that tokens < n predict n
        target_mask = target_ids.ne(self.pad_id)
        active_loss = target_mask[..., 1:].half() # [B*(L-1)]
        shift_logits = lm_logits[..., :-1, :].contiguous() # B * L-1 * V
        shift_labels = target_ids[..., 1:].contiguous() # B * L-1
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
        batch_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)) # [B*(L-1)] * V , [B*(L-1)]
        batch_loss = batch_loss.view(target_ids.shape[0], -1).contiguous()
        # batch_loss = torch.sum(batch_loss * active_loss, dim=1)
        count = torch.sum(active_loss, dim=1)
        batch_loss = torch.sum(batch_loss * active_loss, dim=1)
        batch_loss = batch_loss / count

        loss = torch.mean(batch_loss)
        return batch_loss, loss

    def forward(self, source_ids=None, target_ids=None):
        source_mask = source_ids.ne(self.pad_id)

        outputs = self.encoder(input_ids=source_ids, attention_mask=source_mask) # B * L * D
        encoder_output = outputs[0].permute([1,0,2]).contiguous() # L * B * D 
        attn_mask=-1e4 *(1-self.bias[:target_ids.shape[1],:target_ids.shape[1]])
        tgt_embeddings = self.encoder.embeddings(target_ids)
        tgt_embeddings = tgt_embeddings.permute([1,0,2]).contiguous() # L * B * D
        out = self.decoder(tgt_embeddings,encoder_output,tgt_mask=attn_mask,memory_key_padding_mask=(~source_mask).bool()) # L * B * D
        hidden_states = torch.tanh(self.dense(out)).permute([1,0,2]).contiguous() # B * L * D
        lm_logits = self.lm_head(hidden_states) # B * L * V
        return lm_logits

    def generate(self, source_ids=None, target_ids=None):
        source_mask = source_ids.ne(self.pad_id)
        outputs = self.encoder(source_ids, attention_mask=source_mask) # B * L * D
        encoder_output = outputs[0].permute([1,0,2]).contiguous() # L * B * D 
        attn_mask=-1e4 *(1-self.bias[:target_ids.shape[1],:target_ids.shape[1]])
        tgt_embeddings = self.encoder.embeddings(target_ids)
        tgt_embeddings = tgt_embeddings.permute([1,0,2]).contiguous() # L * B * D
        out = self.decoder(tgt_embeddings,encoder_output,tgt_mask=attn_mask,memory_key_padding_mask=(~source_mask).bool()) # L * B * D
        hidden_states = torch.tanh(self.dense(out)).permute([1,0,2]).contiguous() # B * L * D
        lm_logits = self.lm_head(hidden_states) # B * L * V
        
        if self.beam_size == 0:
            #Predict 
            preds=[]       
            out = self.lsm(lm_logits).data # B * L * V
            max_probs, preds = torch.max(out, dim=2)
        else:
            #Predict 
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
                    tgt_embeddings = self.encoder.embeddings(input_ids).permute([1,0,2]).contiguous() # 1 * beam * D
                    out = self.decoder(tgt_embeddings,context,tgt_mask=attn_mask,memory_key_padding_mask=~context_mask) # 1 * beam * D
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
        
