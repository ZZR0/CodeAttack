import torch.nn as nn
import torch
import os

class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
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
