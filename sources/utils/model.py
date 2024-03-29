from turtle import forward
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import T5Config, T5ForConditionalGeneration, RobertaTokenizer,RobertaConfig, RobertaModel, T5Tokenizer, AutoTokenizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

import logging
import os

logger = logging.getLogger(__name__)

MODEL_CLASSES = {'unixcoder':(RobertaConfig, RobertaModel, RobertaTokenizer)}

def build_or_load_gen_model(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    tokenizer.cls_token = tokenizer.eos_token
    tokenizer.sep_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens':["KEEP", "ADD", "DEL"]})

    # import！！！you must set is_decoder as True for generation
    config.is_decoder = True
    encoder = model_class.from_pretrained(args.model_name_or_path,config=config) 
    model = Seq2Seq_pretrain(encoder=encoder,decoder=encoder,config=config,
                beam_size=args.beam_size,max_length=args.max_target_length,
                    sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0],eos_id=tokenizer.sep_token_id)

    logger.info("Finish loading model [%s] from %s", args.model_type, args.model_name_or_path)
    if args.load_model_path is not None:
        logger.info("Reload model from {}".format(args.load_model_path))
        load_model_dir = os.path.join(args.pre_train_output_root, args.load_model_path, 'pytorch_model.bin') 
        model.load_state_dict(torch.load(load_model_dir))

    return model, tokenizer

# https://github.com/microsoft/CodeBERT/blob/master/CodeBERT/code2nl/model.py
class Seq2Seq_pretrain(nn.Module):       
    def __init__(self, encoder,decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2Seq_pretrain, self).__init__()
        self.encoder = encoder
        self.decoder=decoder
        self.config=config
        self.register_buffer(
            "bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1,1024, 1024)
        )
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight
        self.lsm = nn.LogSoftmax(dim=-1)
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id       
        
    def forward(self, source_ids, task, target_ids=None):  
        if task == 'tp':
            mask = source_ids.ne(1)[:,None,:]*source_ids.ne(1)[:,:,None]
            encoder_output = self.encoder(source_ids,attention_mask=mask,use_cache=True)  
            ids = torch.cat((source_ids,target_ids),-1)
            mask = self.bias[:,source_ids.size(-1):ids.size(-1),:ids.size(-1)].bool()
            mask = mask & ids[:,None,:].ne(1)
            out = self.decoder(target_ids,attention_mask=mask,past_key_values=encoder_output.past_key_values).last_hidden_state
            return self.forward_dp(source_ids=source_ids, target_ids=target_ids, out=out)

        if task == 'cg':
            if target_ids is None:
                return self.generate(source_ids)
            mask = source_ids.ne(1)[:,None,:]*source_ids.ne(1)[:,:,None]
            encoder_output = self.encoder(source_ids,attention_mask=mask,use_cache=True)  
            ids = torch.cat((source_ids,target_ids),-1)
            mask = self.bias[:,source_ids.size(-1):ids.size(-1),:ids.size(-1)].bool()
            mask = mask & ids[:,None,:].ne(1)
            out = self.decoder(target_ids,attention_mask=mask,past_key_values=encoder_output.past_key_values).last_hidden_state
            return self.forward_cg(source_ids=source_ids, target_ids=target_ids, out=out)


    def forward_dp(self,source_ids, target_ids, out):

        logits = self.lm_head(out)
        active_loss = source_ids[..., 1:].eq(4).view(-1)
        shift_logits = logits[..., :-1, :].contiguous()    
        shift_labels = target_ids[..., 1:].contiguous()   

        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                        shift_labels.view(-1)[active_loss])

        return loss, loss* active_loss.sum() , active_loss.sum()

    def forward_cg(self, source_ids, target_ids, out):
        logits = self.lm_head(out)
        active_loss = target_ids[..., 1:].ne(1).view(-1)
        shift_logits = logits[..., :-1, :].contiguous()    
        shift_labels = target_ids[..., 1:].contiguous()   

        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                        shift_labels.view(-1)[active_loss])

        return loss, loss* active_loss.sum() , active_loss.sum()

    def generate(self, source_ids):
        mask = source_ids.ne(1)[:,None,:]*source_ids.ne(1)[:,:,None]
        encoder_output = self.encoder(source_ids,attention_mask=mask,use_cache=True)        
        preds = []       
        zero = torch.cuda.LongTensor(1).fill_(0)   
        source_len = list(source_ids.ne(1).sum(-1).cpu().numpy())
        for i in range(source_ids.shape[0]):
            context = [[x[i:i+1,:,:source_len[i]].repeat(self.beam_size,1,1,1) for x in y] 
                     for y in encoder_output.past_key_values]
            beam = Beam(self.beam_size,self.sos_id,self.eos_id)
            input_ids = beam.getCurrentState()
            context_ids = source_ids[i:i+1,:source_len[i]].repeat(self.beam_size,1)
            for _ in range(self.max_length): 
                if beam.done():
                    break

                ids = torch.cat((context_ids,input_ids),-1)
                mask = self.bias[:,context_ids.size(-1):ids.size(-1),:ids.size(-1)].bool()
                mask = mask & ids[:,None,:].ne(1)
                out = self.decoder(input_ids,attention_mask=mask,past_key_values=context).last_hidden_state
                hidden_states = out[:,-1,:]
                out = self.lsm(self.lm_head(hidden_states)).data
                beam.advance(out)
                input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                input_ids = torch.cat((input_ids,beam.getCurrentState()),-1)
            hyp = beam.getHyp(beam.getFinal())
            pred = beam.buildTargetTokens(hyp)[:self.beam_size]
            pred = [torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
            preds.append(torch.cat(pred,0).unsqueeze(0))

        preds = torch.cat(preds,0)    

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
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))


        for i in range(self.nextYs[-1].size(0)):
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
        
