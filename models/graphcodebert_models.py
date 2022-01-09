import torch.nn as nn
import torch
import os
import torch.nn.functional as F
import numpy as np

from codeattack.models.wrappers import ModelWrapper
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification

from .parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript,DFG_c
from .parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser

dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript,
    'c':DFG_c
}
#load parsers
parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('./models/parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser

#remove comments, tokenize code and extract dataflow                                        
def extract_dataflow(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg

def convert_examples_to_features(code, tokenizer, args):
    lang = args.lang
    code_tokens, dfg = extract_dataflow(code, parsers[lang], lang)
    code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
    ori2cur_pos={}
    ori2cur_pos[-1]=(0,0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
    code_tokens=[y for x in code_tokens for y in x]  
    
    #truncating
    code_tokens=code_tokens[:args.max_source_length-2]
    code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
    dfg=dfg[:args.max_source_length-len(code_tokens)]
    code_tokens+=[x[0] for x in dfg]
    position_idx+=[0 for x in dfg]
    code_ids+=[tokenizer.unk_token_id for x in dfg]
    padding_length=args.max_source_length-len(code_ids)
    position_idx+=[tokenizer.pad_token_id]*padding_length
    code_ids+=[tokenizer.pad_token_id]*padding_length     

    #reindex
    reverse_index={}
    for idx,x in enumerate(dfg):
        reverse_index[x[1]]=idx
    for idx,x in enumerate(dfg):
        dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
    dfg_to_dfg=[x[-1] for x in dfg]
    dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
    length=len([tokenizer.cls_token])
    dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]

    #calculate graph-guided masked function
    attn_mask=np.zeros((args.max_source_length,args.max_source_length),dtype=np.bool)
    #calculate begin index of node and max length of input
    node_index=sum([i>1 for i in position_idx])
    max_length=sum([i!=1 for i in position_idx])
    #sequence can attend to sequence
    attn_mask[:node_index,:node_index]=True
    #special tokens attend to all tokens
    for idx,i in enumerate(code_ids):
        if i in [0,2]:
            attn_mask[idx,:max_length]=True
    #nodes attend to code tokens that are identified from
    for idx,(a,b) in enumerate(dfg_to_code):
        if a<node_index and b<node_index:
            attn_mask[idx+node_index,a:b]=True
            attn_mask[a:b,idx+node_index]=True
    #nodes attend to adjacent nodes         
    for idx,nodes in enumerate(dfg_to_dfg):
        for a in nodes:
            if a+node_index<len(position_idx):
                attn_mask[idx+node_index,a+node_index]=True  

    return code_ids, position_idx, attn_mask

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
        tokenizer = tokenizer_class.f
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
    
    def forward(self, input_ids=None,position_idx=None,attn_mask=None): 
        assert input_ids.size(-1) % 2 == 0
        assert position_idx.size(-1) % 2 == 0
        assert attn_mask.size(-1) % 2 == 0

        input_ids=input_ids.view(-1,input_ids.size(-1)//2) # 2B * L
        position_idx=position_idx.view(-1,position_idx.size(-1)//2) # 2B * L
        attn_mask=attn_mask.view(-1,attn_mask.size(1)//2,attn_mask.size(2),attn_mask.size(3))
        attn_mask = attn_mask.squeeze(1) # 2B * L * L

        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)        
        inputs_embeddings=self.encoder.embeddings.word_embeddings(input_ids)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    

        # inputs_embeddings: 96 * 640 * 768 
        # attn_mask: 96 * 640 * 640 
        # position_idx: 96 * 640
        outputs=self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[0] # 2B * L * D

        logits=self.classifier(outputs)
        prob=F.softmax(logits, dim=-1)
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
        source_ids, position_idx, attn_mask = convert_examples_to_features(source, self.tokenizer, self.args)

        if tokens_to_replace is not None:
            indices = []
            for token in tokens_to_replace:
                indices.append([i for i, x in enumerate(source_ids) if x == token])
            return source_ids, indices
        return source_ids, position_idx, attn_mask

    def __call__(self, text_input_list, batch_size=32):
        input_ids, position_idx, attn_mask = [], [], []
        model_device = next(self.model.parameters()).device
        code0_inputs = [self.get_ids(text["adv1"]) for text in text_input_list]
        code1_inputs = [self.get_ids(text["adv2"]) for text in text_input_list]
        for code0, code1 in zip(code0_inputs, code1_inputs):
            input_ids += [code0[0] + code1[0]]
            position_idx += [code0[1] + code1[1]]
            attn_mask += [[code0[2].tolist(), code1[2].tolist()]]
        
        input_ids = torch.tensor(input_ids).to(model_device)
        position_idx = torch.tensor(position_idx).to(model_device)
        attn_mask = torch.tensor(attn_mask).to(model_device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids,position_idx=position_idx,attn_mask=attn_mask)

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

    def forward(self, input_ids=None,position_idx=None,attn_mask=None,
                p_input_ids=None,p_position_idx=None,p_attn_mask=None): 
        bs,_=input_ids.size()
        input_ids=torch.cat((input_ids,p_input_ids),0)
        position_idx=torch.cat((position_idx,p_position_idx),0)
        attn_mask=torch.cat((attn_mask,p_attn_mask),0)
        
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)        
        inputs_embeddings=self.encoder.embeddings.word_embeddings(input_ids)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask # None to add one dimention
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None] # normalization
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    

        # inputs_embeddings: 96 * 640 * 768 
        # attn_mask: 96 * 640 * 640 
        # position_idx: 96 * 640
        vecs=self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[1] # 3B * D
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
        source_ids, position_idx, attn_mask = convert_examples_to_features(source, self.tokenizer, self.args)

        if tokens_to_replace is not None:
            indices = []
            for token in tokens_to_replace:
                indices.append([i for i, x in enumerate(source_ids) if x == token])
            return source_ids, indices
        return source_ids, position_idx, attn_mask

    def __call__(self, text_input_list, batch_size=32):

        model_device = next(self.model.parameters()).device
        code = [self.get_ids(text["code"]) for text in text_input_list]
        adv = [self.get_ids(text["adv"]) for text in text_input_list]

        code_ids = [c[0] for c in code]
        code_ids = torch.tensor(code_ids).to(model_device)
        adv_ids = [a[0] for a in adv]
        adv_ids = torch.tensor(adv_ids).to(model_device)
        code_position_idx = [c[1] for c in code]
        code_position_idx = torch.tensor(code_position_idx).to(model_device)
        adv_position_idx = [a[1] for a in adv]
        adv_position_idx = torch.tensor(adv_position_idx).to(model_device)
        code_attn_mask = [c[2] for c in code]
        code_attn_mask = torch.tensor(code_attn_mask).to(model_device)
        adv_attn_mask = [a[2] for a in adv]
        adv_attn_mask = torch.tensor(adv_attn_mask).to(model_device)

        with torch.no_grad():
            outputs = self.model(input_ids=adv_ids,position_idx=adv_position_idx,attn_mask=adv_attn_mask,
                                 p_input_ids=code_ids,p_position_idx=code_position_idx,p_attn_mask=code_attn_mask)

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
        
    def forward(self, inputs_ids,position_idx,attn_mask): 
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)        
        inputs_embeddings=self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
        outputs = self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[0]
        logits=outputs
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
        source_ids, position_idx, attn_mask = convert_examples_to_features(source, self.tokenizer, self.args)

        if tokens_to_replace is not None:
            indices = []
            for token in tokens_to_replace:
                indices.append([i for i, x in enumerate(source_ids) if x == token])
            return source_ids, indices
        return source_ids, position_idx, attn_mask

    def __call__(self, text_input_list, batch_size=32):

        model_device = next(self.model.parameters()).device
        code = [self.get_ids(text["adv"]) for text in text_input_list]

        code_ids = [c[0] for c in code]
        code_ids = torch.tensor(code_ids).to(model_device)
        code_position_idx = [c[1] for c in code]
        code_position_idx = torch.tensor(code_position_idx).to(model_device)
        code_attn_mask = [c[2] for c in code]
        code_attn_mask = torch.tensor(code_attn_mask).to(model_device)

        with torch.no_grad():
            outputs = self.model(inputs_ids=code_ids, position_idx=code_position_idx, attn_mask=code_attn_mask)

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

    def forward(self, code_inputs=None, attn_mask=None,position_idx=None, nl_inputs=None): 
        bs=code_inputs.shape[0]

        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)        
        inputs_embeddings=self.encoder.embeddings.word_embeddings(code_inputs)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
        code_vec=self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[1]
        nl_vec=self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[1]

        outputs = []
        for code, nl in zip(code_vec, nl_vec):
            one = torch.cat((code.unsqueeze(0), nl.unsqueeze(0)), dim=0)
            outputs.append(one.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs

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
    
    def get_code_ids(self, source, tokens_to_replace=None):
        source_ids, position_idx, attn_mask = convert_examples_to_features(source, self.tokenizer, self.args)

        if tokens_to_replace is not None:
            indices = []
            for token in tokens_to_replace:
                indices.append([i for i, x in enumerate(source_ids) if x == token])
            return source_ids, indices
        return source_ids, position_idx, attn_mask

    def get_nl_ids(self, source, tokens_to_replace=None):
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
        code = [self.get_code_ids(text["adv"]) for text in text_input_list]
        nl_ids = [self.get_nl_ids(text["nl"]) for text in text_input_list]

        code_ids = [c[0] for c in code]
        code_ids = torch.tensor(code_ids).to(model_device)
        code_position_idx = [c[1] for c in code]
        code_position_idx = torch.tensor(code_position_idx).to(model_device)
        code_attn_mask = [c[2] for c in code]
        code_attn_mask = torch.tensor(code_attn_mask).to(model_device)

        nl_ids = torch.tensor(nl_ids).to(model_device)

        with torch.no_grad():
            outputs = self.model(code_inputs=code_ids, attn_mask=code_attn_mask,position_idx=code_position_idx, nl_inputs=nl_ids)

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
    
    def get_code_ids(self, source, tokens_to_replace=None):
        source_ids, position_idx, attn_mask = convert_examples_to_features(source, self.tokenizer, self.args)

        if tokens_to_replace is not None:
            indices = []
            for token in tokens_to_replace:
                indices.append([i for i, x in enumerate(source_ids) if x == token])
            return source_ids, indices
        return source_ids, position_idx, attn_mask

    def get_nl_ids(self, source, tokens_to_replace=None):
        source_tokens=self.tokenizer.tokenize(source)[:self.max_target_length-2]
        source_tokens =[self.tokenizer.cls_token]+source_tokens+[self.tokenizer.sep_token]
        source_ids =  self.tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = self.max_target_length - len(source_ids)
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
        code = [self.get_code_ids(text["adv"]) for text in text_input_list]
        tgt_ids = [self.get_nl_ids(text["nl"]) for text in text_input_list]

        code_ids = [c[0] for c in code]
        code_ids = torch.tensor(code_ids).to(model_device)
        code_position_idx = [c[1] for c in code]
        code_position_idx = torch.tensor(code_position_idx).to(model_device)
        code_attn_mask = [c[2] for c in code]
        code_attn_mask = torch.tensor(code_attn_mask).to(model_device)

        tgt_ids = torch.tensor(tgt_ids).to(model_device)

        with torch.no_grad():
            outputs = self.model.generate(source_ids=code_ids,position_idx=code_position_idx,attn_mask=code_attn_mask,target_ids=tgt_ids)
        
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
        
        self.beam_size=beam_size
        self.max_length=max_length
        self.sos_id=sos_id
        self.eos_id=eos_id
        self.pad_id=pad_id
        
    def get_input_embeddings(self):
        return self.encoder.embeddings.word_embeddings      

    def forward(self, source_ids,source_mask,position_idx,attn_mask,target_ids=None):   
        #embedding
        source_mask=source_ids.ne(self.pad_id)
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)        
        inputs_embeddings=self.encoder.embeddings.word_embeddings(source_ids)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]  

        outputs = self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx) # B * L * D
        encoder_output = outputs[0].permute([1,0,2]).contiguous() # L * B * D
        attn_mask=-1e4 *(1-self.bias[:target_ids.shape[1],:target_ids.shape[1]])
        tgt_embeddings = self.encoder.embeddings(target_ids).permute([1,0,2]).contiguous() # L * B * D
        out = self.decoder(tgt_embeddings,encoder_output,tgt_mask=attn_mask,memory_key_padding_mask=(~source_mask).bool()) # L * B * D
        hidden_states = torch.tanh(self.dense(out)).permute([1,0,2]).contiguous() # B * L * D
        lm_logits = self.lm_head(hidden_states) # B * L * V
        return lm_logits


    def generate(self, source_ids,position_idx,attn_mask,target_ids=None):
        source_mask = source_ids.ne(self.pad_id)
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)        
        inputs_embeddings=self.encoder.embeddings.word_embeddings(source_ids)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]  

        outputs = self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx) # B * L * D
        encoder_output = outputs[0].permute([1,0,2]).contiguous() # L * B * D
        attn_mask=-1e4 *(1-self.bias[:target_ids.shape[1],:target_ids.shape[1]])
        tgt_embeddings = self.encoder.embeddings(target_ids).permute([1,0,2]).contiguous() # L * B * D
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
        
