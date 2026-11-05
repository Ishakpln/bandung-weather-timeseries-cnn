"""Reproducible exploratory convolutional autoencoder experiments (CPU friendly).
Run from any directory: python src/stage6_cnn_experiments.py --epochs 3
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from common import ROOT, CONTINUOUS, FIGURES, TABLES, REPORTS, json_write

OUT=ROOT/'output'/'cnn'; OUT.mkdir(parents=True,exist_ok=True)

class AE1D(nn.Module):
 def __init__(self, channels):
  super().__init__()
  self.encoder=nn.Sequential(nn.Conv1d(channels,16,5,padding=2),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(16,8,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(8))
  self.decoder=nn.Sequential(nn.Upsample(size=48,mode='linear',align_corners=False),nn.Conv1d(8,16,3,padding=1),nn.ReLU(),nn.Conv1d(16,channels,3,padding=1))
 def forward(self,x):
  z=self.encoder(x);return self.decoder(z),z
class AE2D(nn.Module):
 def __init__(self):
  super().__init__()
  self.encoder=nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(8,8,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)))
  self.decoder=nn.Sequential(nn.Upsample(size=(32,32),mode='bilinear',align_corners=False),nn.Conv2d(8,8,3,padding=1),nn.ReLU(),nn.Conv2d(8,1,3,padding=1),nn.Tanh())
 def forward(self,x):
  z=self.encoder(x);return self.decoder(z),z

def recurrence(x):
 """Per-window temperature recurrence similarity, no fitting on held-out data.
 Output R[i,j]=exp(-abs(x_i-x_j)/0.25) after per-window minmax normalization.
 This is a continuous recurrence similarity matrix, not binary threshold RP.
 """
 ids=np.linspace(0,len(x)-1,32).round().astype(int)
 a=x[ids].astype(np.float32);a=(a-a.min())/(np.ptp(a)+1e-6)
 return np.exp(-np.abs(a[:,None]-a[None,:])/0.25).astype('float32')

def windows(frame, indices, length=48, limit=2500, seed=42):
 starts=np.arange(indices[0],indices[-1]-length+2,length)
 rng=np.random.default_rng(seed);starts=np.sort(rng.choice(starts,min(limit,len(starts)),replace=False))
 return starts,np.stack([frame[s:s+length] for s in starts]).astype('float32')

def train(model, train_x, val_x, epochs, batch, lr):
 optimizer=torch.optim.Adam(model.parameters(),lr=lr)
 loader=DataLoader(TensorDataset(torch.from_numpy(train_x)),batch_size=batch,shuffle=True)
 history=[];best=float('inf');checkpoint=None
 for epoch in range(epochs):
  model.train();total=0
  for (x,) in loader:
   optimizer.zero_grad();y,_=model(x);loss=((y-x)**2).mean();loss.backward();optimizer.step();total+=loss.item()*len(x)
  model.eval()
  with torch.no_grad():
   xv=torch.from_numpy(val_x);val=np.mean([((model(z)[0]-z)**2).mean().item() for z in xv.split(batch)])
  history.append({'epoch':epoch+1,'train_mse':total/len(train_x),'validation_mse':val})
  if val<best:best=val;checkpoint={k:v.detach().clone() for k,v in model.state_dict().items()}
 model.load_state_dict(checkpoint);return pd.DataFrame(history)

def assess(model,x,batch=128):
 model.eval();recs=[];zs=[];maps=[]
 with torch.no_grad():
  for z in torch.from_numpy(x).split(batch):
   r,embedding=model(z);recs.append(r.numpy());zs.append(embedding.flatten(1).numpy())
  maps=model.encoder[1](model.encoder[0](torch.from_numpy(x[:1]))).numpy()[0]
 return np.concatenate(recs),np.concatenate(zs),maps

def saveplot(name):
 plt.tight_layout();plt.savefig(OUT/name,dpi=150,bbox_inches='tight');plt.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch',type=int,default=64);p.add_argument('--lr',type=float,default=.001);p.add_argument('--max-train',type=int,default=2500);args=p.parse_args()
 torch.set_num_threads(min(4,torch.get_num_threads()));torch.manual_seed(42);np.random.seed(42);random.seed(42)
 frame=pd.read_csv(ROOT/'data/processed/bandung_weather_hourly_clean.csv',parse_dates=['time'])
 assert frame.time.is_monotonic_increasing and frame.time.is_unique
 assert frame[CONTINUOUS].notna().all().all()
 dates=frame.time.dt.year.to_numpy();split={'train':np.flatnonzero(dates<=2023),'validation':np.flatnonzero(dates==2024),'test':np.flatnonzero(dates==2025)}
 scaler=StandardScaler().fit(frame.loc[split['train'],CONTINUOUS]);scaled=scaler.transform(frame[CONTINUOUS]).astype('float32')
 json_write(OUT/'scaler.json',{'columns':CONTINUOUS,'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'train_period':'2022-2023'})
 raw=frame[CONTINUOUS].to_numpy(dtype='float32')
 sets={};starts={}
 for name,idx in split.items():
  starts[name],sets[name]=windows(scaled,idx,limit=args.max_train if name=='train' else 500,seed=42)
  assert all(s+47 <= idx[-1] for s in starts[name])
 x1={k:v.transpose(0,2,1).copy() for k,v in sets.items()}
 x2={k:np.stack([recurrence(raw[s:s+48,0]) for s in st])[:,None] for k,st in starts.items()}
 results={};embeds={}
 for tag,model,x in [('cnn1d',AE1D(len(CONTINUOUS)),x1),('cnn2d',AE2D(),x2)]:
  t=time.perf_counter();history=train(model,x['train'],x['validation'],args.epochs,args.batch,args.lr)
  history.to_csv(OUT/f'{tag}_training.csv',index=False)
  torch.save({'state_dict':model.state_dict(),'features':CONTINUOUS,'window':48,'seed':42},OUT/f'{tag}.pt')
  fig,ax=plt.subplots(figsize=(6,3));ax.plot(history.epoch,history.train_mse,label='Train');ax.plot(history.epoch,history.validation_mse,label='Validation');ax.set(xlabel='Epoch',ylabel='MSE',title=f'{tag.upper()} training');ax.legend();saveplot(f'{tag}_loss.png')
  results[tag]={'elapsed_seconds':round(time.perf_counter()-t,2),'epochs':args.epochs,'samples':{k:len(v) for k,v in x.items()},'mse':{}}
  for part in ['train','validation','test']:
   rec,z,m=assess(model,x[part]);results[tag]['mse'][part]=float(np.mean((rec-x[part])**2))
   if part=='test':
    embeds[tag]=z;np.save(OUT/f'{tag}_test_embedding.npy',z)
    fig,axs=plt.subplots(1,2,figsize=(9,3.5));axs[0].imshow(x[part][0,0] if tag=='cnn2d' else x[part][0].T,aspect='auto',cmap='viridis');axs[0].set_title('Input');axs[1].imshow(rec[0,0] if tag=='cnn2d' else rec[0].T,aspect='auto',cmap='viridis');axs[1].set_title('Reconstruction');saveplot(f'{tag}_reconstruction.png')
    fig,ax=plt.subplots(figsize=(8,3));ax.imshow(m[:8].reshape(min(8,len(m)),-1),aspect='auto',cmap='viridis');ax.set(title=f'{tag.upper()} first-layer activations',xlabel='Time / image position',ylabel='Filter');saveplot(f'{tag}_activations.png')
    kernel=model.encoder[0].weight.detach().numpy();fig,ax=plt.subplots(figsize=(7,3));ax.imshow(kernel.reshape(kernel.shape[0],-1),aspect='auto',cmap='coolwarm');ax.set(title=f'{tag.upper()} learned first-layer kernels',xlabel='Flattened kernel coefficients',ylabel='Output channel');saveplot(f'{tag}_kernels.png')
  z=embeds[tag];proj=PCA(n_components=2).fit_transform(z);np.save(OUT/f'{tag}_test_pca.npy',proj)
  fig,ax=plt.subplots(figsize=(6,4));c=np.array([frame.iloc[s].time.month for s in starts['test']]);scatter=ax.scatter(proj[:,0],proj[:,1],c=c,cmap='viridis',s=12,alpha=.7);fig.colorbar(scatter,ax=ax,label='Month');ax.set(title=f'{tag.upper()} test embeddings (PCA)',xlabel='PC1',ylabel='PC2');saveplot(f'{tag}_pca.png')
 # input RP figure
 fig,axs=plt.subplots(1,2,figsize=(9,3.5));axs[0].plot(np.arange(48),raw[starts['test'][0]:starts['test'][0]+48,0]);axs[0].set(xlabel='Hour in window',ylabel='Temperature (°C)',title='Original temperature');axs[1].imshow(x2['test'][0,0],cmap='viridis',vmin=0,vmax=1);axs[1].set(title='Continuous recurrence similarity',xlabel='Sample index',ylabel='Sample index');saveplot('recurrence_example.png')
 results['configuration']={'seed':42,'window_hours':48,'recurrence':'exp(-abs(normalized_temperature_difference)/0.25); 32x32','features':CONTINUOUS,'chronological_split':{'train':'2022-2023','validation':'2024','test':'2025'},'optimizer':'Adam','lr':args.lr,'batch':args.batch,'torch':torch.__version__,'note':'1D and 2D reconstruction losses have different target spaces; do not rank models by raw MSE.'}
 json_write(OUT/'results.json',results)
 print(json.dumps(results,indent=2))
if __name__=='__main__':main()
