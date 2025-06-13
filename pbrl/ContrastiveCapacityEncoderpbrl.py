
import os
import random


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb


import sys
import copy

#sys.path.append("..")



from pathlib import Path
import random
from torch.utils.data import DataLoader


from torch import nn, Tensor
import math


import pickle

def set_seed(seed) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Ensures deterministic cuBLAS behavior

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)  # Ensure deterministic algorithms
    torch.set_float32_matmul_precision('high')


    print(f"Random seed set as {seed}")




class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        #max_len is max sequence length (set in config)
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(float(max_len)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe, persistent=False) #will not be trained and persistent=False means not part of model's state dict

    def forward(self, x): 
        x = x + self.pe[:x.size(1), :] 
        return self.dropout(x)



class TransformerModel(nn.Module):

    def __init__(self, inputSize: int, cfg: dict, device=torch.device('cpu')): 

        super().__init__() 
        
        self.cfg=cfg
        if cfg['transformerCfg']['avgPool']:
            self.max_len=self.cfg['stepMax']
        else:
            self.max_len=self.cfg['stepMax']+1 #+1 is for the additional class token 

        self.transformer_input_layer = nn.Linear(in_features=inputSize, out_features=self.cfg['transformerCfg']['d_model'] )
        self.positional_encoding_layer = PositionalEncoding(d_model=self.cfg['transformerCfg']['d_model'],max_len=self.max_len,dropout=self.cfg['transformerCfg']['dropout_pos_enc'])
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.cfg['transformerCfg']['d_model'], nhead=self.cfg['transformerCfg']['n_heads'],dim_feedforward=self.cfg['transformerCfg']['dim_feedforward_encoder'],dropout=self.cfg['transformerCfg']['dropout_encoder'],batch_first=self.cfg['transformerCfg']['batch_first'])
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,num_layers=self.cfg['transformerCfg']['n_encoder_layers'])
        self.out_linear=nn.Linear(in_features=self.cfg['transformerCfg']['d_model'], out_features=self.cfg['embedDim'])

        self.secondEncoder_layer=nn.TransformerEncoderLayer(d_model=self.cfg['transformerCfg']['d_model'], nhead=self.cfg['transformerCfg']['n_heads'],dim_feedforward=self.cfg['transformerCfg']['dim_feedforward_encoder'],dropout=self.cfg['transformerCfg']['dropout_encoder'],batch_first=True)
        self.secondEncoder = nn.TransformerEncoder(encoder_layer=self.secondEncoder_layer,num_layers=self.cfg['transformerCfg']['n_encoder_layers'])

        self.class_tokenSecond = torch.randn(1, 1, self.cfg['transformerCfg']['d_model'])
        self.device=device #needed to put the enc causal mask on the same device this is initiated on
        
        
    

    def forward(self, input: Tensor, src_key_padding_mask: Tensor, setSize, numSetsPerAgent, eval=False) -> Tensor:
        ##input is size batch_size by numTimeSteps (+1 if avgPool is False) by inputDim. 
        #numSetsPerAgent and SetSize are input rather than computed so we ensure that it is kept the same for both agents. 
        out = self.transformer_input_layer(input)
        out = self.positional_encoding_layer(out)
        
        if self.cfg['transformerCfg']['timestep_mask']=='causal':
            src_mask=nn.Transformer.generate_square_subsequent_mask(input.shape[1]).bool().to(self.device) #if causal_mask is True, we only do causal masking for the first encoder. Due to the pooling step that follows, the second encoder does not operate on sequential data (rather, just sets)
        elif self.cfg['transformerCfg']['timestep_mask']=='window':
            window_size=self.cfg['transformerCfg']['windowEnc']
            if window_size is None:
                raise ValueError("Config error for window size")
            # Create a sliding window causal mask (can only attend to past timesteps up to window_size)
            max_ep_len=input.shape[1]
            src_mask = torch.full((max_ep_len, max_ep_len), fill_value=True, dtype=torch.bool).to(self.device)

            for i in range(max_ep_len):
                start = max(0, i - window_size + 1)  # +1 to ensure exactly 100 past timesteps
                src_mask[i, start:i+1] = False  # Allow self and past 99 tokens
        elif self.cfg['transformerCfg']['timestep_mask']=='NoMask':
            src_mask=None
        else:
            raise ValueError("Config error for timestep_mask")
        
        out=self.encoder(src=out,mask=src_mask,src_key_padding_mask=src_key_padding_mask.bool())
        if self.cfg['transformerCfg']['avgPool']:
            tokenOut=torch.mean(out,dim=1)
        else:
            tokenOut=out[:,0,:] #read off the embedding for the cls token which was put at position 0 by design. This is now shape batch_size by d_model
        
       
        if setSize is not None: #ie taking sets over the agent's trajectories, otherwise not and looking at individual trajectories
            if setSize==0:
                raise ValueError("Should not have zero set size")
            outputEmbed=None
            if not eval: #
        
                
                #numSetsPerAgent=int(len(tokenOut)/(setSize)) #divy up the number of total sets you want by the number of trajectories you have 

                for k in range(numSetsPerAgent):

                    subset = tokenOut[setSize*k:min(setSize*(k+1),len(tokenOut)),:]
    
                    # If the subset is empty, randomly sample `setSize` elements
                    if subset.shape[0] == 0:
                        indicesSubset = torch.randint(0, len(tokenOut), (setSize,))  # Random indices
                        subset = tokenOut[indicesSubset, :]  # Select random elements

                    inputForSecondLayer = subset.unsqueeze(0) #unsqueeze because batch_first is true. dimension of inputForSecondLayer is 1,setSize,embedDim
                   
                    if inputForSecondLayer.shape[1]!=setSize:
                        print('set size {} but input is {}'.format(setSize,inputForSecondLayer.shape[1]))
                        print('batch input size {}'.format(input.shape))
                        raise ValueError("Input for second layer should be set size") #for edge cases if they arise

                    

                    if not self.cfg['transformerCfg']['avgPool']:
                        inputForSecondLayer=torch.cat([self.class_tokenSecond,inputForSecondLayer],dim=1)
                    embed=self.secondEncoder(src=inputForSecondLayer)
                    if self.cfg['transformerCfg']['avgPool']:
                        embed=torch.mean(embed,dim=1)
                    else:
                        embed=embed[:,0,:] #read off the embedding for the cls token which was put at position 0 by design. This is now shape batch_size by d_model

                
                    
                    if outputEmbed is None:
                        outputEmbed=embed
                    else:
                        outputEmbed=torch.cat([outputEmbed,embed],dim=0)

            else: #if in eval mode. The only difference between eval and above is that in eval we pass in the whole batch whereas above we divy it up into sub-batches to have embeddings over subsets for the agent
                inputForSecondLayer=tokenOut.unsqueeze(0) #in eval mode, pass everything we have for this agent in. This will be the embeddings for multiple trajectories, but all pertaining to the same agent

                if not self.cfg['transformerCfg']['avgPool']:
                    inputForSecondLayer=torch.cat([self.class_tokenSecond,inputForSecondLayer],dim=1)
                
                embed=self.secondEncoder(src=inputForSecondLayer)
                if self.cfg['transformerCfg']['avgPool']:
                    embed=torch.mean(embed,dim=1)
                else:
                    embed=embed[:,0,:] #read off the embedding for the cls token which was put at position 0 by design. This is now shape batch_size by d_model

            
                
                if outputEmbed is None:
                    outputEmbed=embed
                else:
                    outputEmbed=torch.cat([outputEmbed,embed],dim=0)
            outputEmbed=self.out_linear(outputEmbed)
            
        else:
            embed=self.secondEncoder(src=tokenOut.unsqueeze(1))
            embed=embed.squeeze()
            outputEmbed=self.out_linear(embed) #size is batch_size by embedDim
            
        return outputEmbed


   






class CapacityEncoderV2(torch.nn.Module):
    def __init__(self, cfg, train_set, test_set, use_wandb=True):
        super(CapacityEncoderV2, self).__init__()
        set_seed(seed=cfg['seed'])
        filepath=cfg['filepath']
        timestamp=filepath[-6:] 
        filepath = Path(filepath)
        filepath.mkdir(exist_ok=True, parents=True)
        self.cfg=cfg
        self.use_wandb=use_wandb
        if cfg['obsOnly']:
            self.inputDim=self.cfg['obs_dim']
        else:
            self.inputDim=self.cfg['obs_dim']+self.cfg['action_dim']
        
        self.device=cfg['device']
        self.capacity_encoder=TransformerModel(inputSize=self.inputDim, cfg=self.cfg, device=self.device).to(self.device)#torch.nn.RNN(input_size=inputDim, hidden_size=cfg['embedDim'],num_layers=1, batch_first=True)
        self.capacity_encoder.train()


      
        self.agent_list=['unpreferred','preferred']
    
       
        print("#######Using preloaded train/test sets#######")
        print("Input train set size {} and test set size is {}".format(len(train_set),len(test_set)))
        if cfg['useFullQuerySet'] and len(test_set)>0:
            train_set=torch.cat([train_set,test_set],dim=0)
            print("#######Using full query set with train set size {}##########".format(len(train_set)))
        

        self.train_set,self.test_set=train_set,test_set



        self.train_set=self.train_set.to(self.device)
        self.trainDataloader=DataLoader(self.train_set, batch_size=int(self.cfg['batch_size']), shuffle=True, num_workers=0, worker_init_fn=lambda x: np.random.seed(cfg['seed']))
        if len(test_set)>0:
            self.test_set=self.test_set.to(self.device)
            self.testDataloader=DataLoader(self.test_set, batch_size=int(self.cfg['batch_size']), shuffle=True, num_workers=0, worker_init_fn=lambda x: np.random.seed(cfg['seed']))
        else:
            self.testDataloader=None

       
        if self.use_wandb:
            parts = list(filepath.parts)
            if cfg['task_name'] in parts:
                indexfp = parts.index(cfg['task_name'])
                wandbDir=os.path.join(os.sep.join(parts[:indexfp + 1]),'wandb')
            else:
                wandbDir='/mnt/vast-react/projects/rl_pref_constraint/wandb'
            projectName=cfg['task_name']
            if 'scriptLabel' in cfg['group']:
                projectName += '_scriptLabel'
            if 'mistake' in cfg['group']:
                projectName += '_error'
            self.logger=wandb.init(project="CapacityEncoder_{}".format(projectName),group=self.cfg['group'], name=cfg['exp_name'], config=self.cfg, job_type=cfg['job_type'],dir=wandbDir) 
        self.optimizer = torch.optim.Adam(self.capacity_encoder.parameters(), lr=cfg['lr'])
        if self.cfg['cosine_lr']:
            #self.scheduler=torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=cfg['lr'],steps_per_epoch=len(self.trainDataloader),epochs=cfg['epochs'],anneal_strategy = 'cos')
            self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg['epochs'], eta_min=cfg['lr']/10, 
                                        last_epoch=-1, verbose='deprecated')

        self.seqLenList=cfg['seqLenList']
      
    
    def forward(self,batch, setSize, numSetsPerAgent, eval=False, seqLen=None):
        '''batch is the batch for one agent and includes the mask (but not agent id): size is batch_size, numTimesteps, inputDim (if obs only, then obsDim, otherwise obsDim+acitonDim) + 1 (mask)
        
        for a given batch, first transformer layer does normal self attention: output is batch_size, embedDim. For each agent, randomly picks the setSize number of latents
        . Second transformer layer does attention on that set.
         
        The output is then number of sets in batch by embedDim 
        '''
        if self.cfg['numSetsPerAgent']<2 and not eval:
            raise ValueError("numSetsPerAgent should be at least 2 unless doing eval")
        assert batch.shape[2] == (self.inputDim+1),(batch.shape, (self.inputDim+1))

        ##########Sample subsequence##################
        if seqLen is None:
            seqLen=batch.shape[1]

        if self.cfg['random_seqLen_start']:
            # Make sure seq_len <= time dimension
            max_timesteps = batch.size(1)
            assert seqLen <= max_timesteps, "seq_len must be less than or equal to time_steps"

            # Random start index
            start = torch.randint(0, max_timesteps - seqLen + 1, (1,)).item()
        else:
            start=0
        ############################################
 
        batchData=batch[:,start:start + seqLen,:self.inputDim]
        mask=batch[:,start:start + seqLen,self.inputDim].to(torch.bool)
        latents=self.capacity_encoder(batchData,src_key_padding_mask=mask, setSize=setSize, numSetsPerAgent=numSetsPerAgent,eval=eval) 
        #latent=hn.squeeze()
        return latents
    

    
    def get_setSize(self,data,numSetsPerAgent):
        #data is shape batch_size, numTimesteps, inputDim, where the last index of inputDim tells us the agent index
        agent_labels = data[:,0,-1]
        numTrajs_agent0 = (agent_labels == 0.0).sum().item()
        numTrajs_agent1 = (agent_labels == 1.0).sum().item()
        if (numTrajs_agent0+numTrajs_agent1) != data.shape[0]:
            raise ValueError('get_setSize, numTrajs dont add up')
        setSizeAgent0=int(numTrajs_agent0/numSetsPerAgent)
        setSizeAgent1=int(numTrajs_agent1/numSetsPerAgent)
        setSize=min(setSizeAgent0,setSizeAgent1)
        if setSize<1:
            setSize=1
            #raise ValueError("Returned 0 set size")
        return setSize
    
    def simclr_loss(self,data,numSetsPerAgent,seqLen=None): 
        '''
        batch includes  the mask info, agentID
        batch is dimension of batch_size by numTimesteps (or numTimesteps+1 if we have cls token) by inputDim (ie obs or obs+action)  +1 (for mask), +1 (agentID)
        '''
        
        numAgents=len(self.agent_list)
        
        embedDim=self.cfg['embedDim']
        if numSetsPerAgent is None: #if numSetsPerAgent is None, we are not doing the set Transformer idea
            batchDim=data.shape[0]
            setSize=None
        else:
            batchDim=numSetsPerAgent
            setSize=self.get_setSize(data,numSetsPerAgent)
        all_latents=torch.zeros((numAgents,batchDim,embedDim))

        
        cos=torch.nn.CosineSimilarity(dim=1)
        numSetsByAgent=[]
        for agIdx in range(numAgents):
            allIndsAgent=(data[:,0,-1]==agIdx).nonzero().flatten()
            dataAgent=data[allIndsAgent]
            dataAgent=dataAgent[:,:,:-1].clone().detach() #excludes agentID but still includes the mask info
            latents=self.forward(dataAgent, setSize=setSize, numSetsPerAgent=numSetsPerAgent, seqLen=seqLen) # numSetsAgent by embedDim
            numSetsByAgent.append(latents.shape[0]) #the number of latents we end up with is simply given by the number of trajectories for this agent in this batch. Otherwise it is numSetsPerAgent
            all_latents[agIdx,:latents.shape[0],:]=latents 

        
        cos=torch.nn.CosineSimilarity(dim=1)
        
        contrastiveLoss=0.0
        for i in range(numAgents):
            ###compute similarities for the set encodings completed for this agent, with the exception of the set with itself (for the same agent), which is of course 1
            numSetsAgenti=numSetsByAgent[i]
            indList1=torch.tensor(range(numSetsAgenti))
            indList2=torch.tensor(range(numSetsAgenti))

            #create combinations of sets for the agent, excluding the same set with itself 
            comboIndsSameAgent=torch.cartesian_prod(indList1, indList2) 
            elementsToDrop=(comboIndsSameAgent[:,0]==comboIndsSameAgent[:,1]).nonzero().squeeze()
            elementsToKeep=list(set(torch.arange(len(comboIndsSameAgent)).numpy()).difference(set(elementsToDrop.numpy())))
            comboIndsSameAgent=comboIndsSameAgent[elementsToKeep]

            latentsSame1=all_latents[i,comboIndsSameAgent[:,0],:] #latents indexed as 000111222 (for example)
            latentsSame2=all_latents[i,comboIndsSameAgent[:,1],:]
            if len((latentsSame1.count_nonzero(dim=1)==0).nonzero())>0: ##sanity to check to ensure no zero rows
                raise ValueError("Check accessing of latents in simclr loss")
            if len((latentsSame2.count_nonzero(dim=1)==0).nonzero())>0: ##sanity to check to ensure no zero rows
                raise ValueError("Check accessing of latents in simclr loss")

            
            simSameAgent=torch.exp(cos(latentsSame1,latentsSame2)/self.current_temp) #size of (numSetsPerAgent)*(numSetsPerAgent-1), ie number of unique combinations of different sets for the same agent
            if self.cfg['NoNegative']:
                contrastiveLoss=contrastiveLoss-torch.mean(torch.log(simSameAgent))
            else:
                for j in range(numAgents):
                    if i==j:
                        continue
                    numSetsOther=numSetsByAgent[j]
                    indListOther=torch.tensor(range(numSetsOther))

                    comboIndsCross=torch.cartesian_prod(indList1, indListOther)
                    #compute similarities for the denominator 

                    latentsAgent=all_latents[i,comboIndsCross[:,0],:] #these are latents indexed as 000111222 up to the number of latents for agent_i -1 
                    latentsOther=all_latents[j,comboIndsCross[:,1],:] #these are latents indexed as 0123401234 up to the number of latents for agent_j -1
                    if len((latentsAgent.count_nonzero(dim=1)==0).nonzero())>0: #this error handling probably not needed for this case since we just have one task. But keep it anyway
                        raise ValueError("Check accessing of latents in simclr loss")
                    if len((latentsOther.count_nonzero(dim=1)==0).nonzero())>0: #this error handling probably not needed for this case since we just have one task. But keep it anyway
                        raise ValueError("Check accessing of latents in simclr loss")
                    simCrossAgent=torch.exp(cos(latentsAgent,latentsOther)/self.current_temp)

                    s = torch.split(simCrossAgent, numSetsOther)
                    denomSimSum = torch.stack(s).sum(dim=1) #this is dimension of number of sets  by agent i. This gives the denominator terms for similarities between this agent and other agent on all the sets
                    numRepeats=(numSetsPerAgent-1) #repeat for the number of times we have positive samples for agent_i and set_m 
                    denomSimSum=denomSimSum.repeat_interleave(numRepeats)
                    r=simSameAgent/(simSameAgent+denomSimSum)
                    contrastiveLoss=contrastiveLoss-torch.mean(torch.log(r))
        
        return (contrastiveLoss/numAgents)
    

    
    def train_one_epoch(self):
        avgLossCon=0.0
        self.current_temp=self.cfg["temp"] #temp[epoch]# #
        for i, data in enumerate(self.trainDataloader):
            self.optimizer.zero_grad()
            contrastiveLoss=0.0
            for seqLen in self.seqLenList: 
                contrastiveLoss_k = self.simclr_loss(data,numSetsPerAgent=self.cfg['numSetsPerAgent'],seqLen=seqLen)
                contrastiveLoss=contrastiveLoss+contrastiveLoss_k
            loss=self.cfg['contWeight']*contrastiveLoss
            loss.backward()
            self.optimizer.step()
            avgLossCon += contrastiveLoss.item()
            
        if self.cfg['cosine_lr']:
            self.scheduler.step()
        return avgLossCon/(i+1)
    
    def get_test_loss(self):
        self.capacity_encoder.eval()
        self.current_temp=self.cfg["temp"] #temp[epoch]# #
        testDataset=self.testDataloader.dataset
        
        testDataset=testDataset[torch.randperm(testDataset.size()[0])]
        #with torch.no_grad():
        if self.cfg['evalOn2Sets']:
            numSetsPerAgent=2
        else:
            numSetsPerAgent=self.cfg['numSetsPerAgent']
        contrastiveLoss=0.0
        for seqLen in self.seqLenList:
            contrastiveLoss_k = self.simclr_loss(testDataset,numSetsPerAgent=numSetsPerAgent,seqLen=seqLen) #we feed in full test dataset but still divying it into sets and computing contrastive loss
            contrastiveLoss=contrastiveLoss+contrastiveLoss_k
            
        self.capacity_encoder.zero_grad()
        self.capacity_encoder.train()
        return contrastiveLoss.item()
    
    def fit(self):
        
        self.capacity_encoder.train()
        bestLoss=1000.0
       
        #temp=[max(self.tempMin,self.tempMax/ (1. + self.k*e)) for e in epochs]
        for epoch in range(self.cfg['epochs']):
            if epoch%100==0:
                print(epoch)
            avg_lossCon=self.train_one_epoch()
            if self.testDataloader is not None:
                testConLoss=self.get_test_loss()
            else:
                testConLoss=0.0,0.0
            if self.use_wandb:
                self.logger.log({'train/Contrastive_Loss':avg_lossCon,'eval/Contrastive_Loss':testConLoss}, step=epoch)
          
            if avg_lossCon<bestLoss:
                bestModel= copy.deepcopy(self.capacity_encoder.state_dict())
                bestLoss=avg_lossCon
                bestEpoch=epoch
        print("Best epoch at {} and best loss {}".format(bestEpoch,bestLoss))
        if self.cfg['early_stop']:
            self.capacity_encoder.load_state_dict(bestModel)
            #uses the bestEpoch and bestLoss from above in wandb logging, and loads the current contrastive encoder to be the best model corresponding to the lowest loss
        else:
            #uses the last epoch and last loss for wandb logging
            bestLoss=avg_lossCon
            bestEpoch=epoch
            #leave the contrastive encoder where it is, rather than loading the bestModel 

        ##save best model in the filepath and in wandb
        #save train and test sets
        torch.save(self.capacity_encoder.state_dict(), os.path.join(self.cfg['filepath'],'capacityEncoder.pt')) #save torch model
        with open(os.path.join(self.cfg['filepath'],'train_set.pkl'), 'wb') as f: 
            pickle.dump(self.train_set, f)
        with open(os.path.join(self.cfg['filepath'],'test_set.pkl'), 'wb') as f: 
            pickle.dump(self.test_set, f)
        with open(os.path.join(self.cfg['filepath'],'cfg.pkl'), 'wb') as f: 
            pickle.dump(self.cfg, f)

        #####################################

        #similarities using full datasets and full sequence lengths
        crossSimTrain=self.get_full_setCrossSim(self.trainDataloader)
        if self.testDataloader is not None:
            crossSimTest=self.get_full_setCrossSim(self.testDataloader)
        else:
            crossSimTest=0.0
        ##########################################
        metricsDict={'train/bestTrainLoss':bestLoss,'train/bestTrainEpoch':bestEpoch,'train/crossSimTrain': crossSimTrain, 'eval/crossSimTest': crossSimTest}
        
        #If len(self.seqLenList)>1 (ie we trained using smaller sequence lengths), then we also want to log the crossSimTrain and crossSimTest at the smallest length in our list
        if len(self.seqLenList)>1:
            crossSimTrain0=self.get_full_setCrossSim(self.trainDataloader,seqLen=self.seqLenList[0])
            if self.testDataloader is not None:
                crossSimTest0=self.get_full_setCrossSim(self.testDataloader,seqLen=self.seqLenList[0])
            else:
                crossSimTest0=0.0
            metricsDict['train/crossSimTrain{}'.format(self.seqLenList[0])]=crossSimTrain0
            metricsDict['eval/crossSimTest{}'.format(self.seqLenList[0])]=crossSimTest0

        if self.use_wandb:
            self.logger.save(os.path.join(self.cfg['filepath'],'capacityEncoder.pt'))
            self.logger.log(metricsDict)

        if self.use_wandb:
            self.logger.finish()

    
    def getEvalLatents(self,i,dataloader,seqLen=None):
        dataset=dataloader.dataset
        allIndsAgenti=(dataset[:,0,-1]==i).nonzero().flatten()
        uniqueTrajs_i=torch.unique(dataset[allIndsAgenti],dim=0)
        latenti=self.forward(uniqueTrajs_i[:,:,:-1],setSize=uniqueTrajs_i[:,:,:-1].shape[0],numSetsPerAgent=1, eval=True, seqLen=seqLen) #passing in the whole set to get one latent for the agent
        return latenti

    def get_full_setCrossSim(self,dataloader,seqLen=None):
        self.capacity_encoder.eval()

        latenti=self.getEvalLatents(0,dataloader,seqLen=seqLen)
        latentj=self.getEvalLatents(1,dataloader,seqLen=seqLen)
        cos=torch.nn.CosineSimilarity(dim=1)
        return cos(latenti,latentj).item()




        
        








    

        
 
    