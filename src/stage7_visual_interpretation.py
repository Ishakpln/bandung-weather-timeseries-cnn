"""Visual interpretation of saved CNN checkpoints; no retraining or new claims.
Run: python src/stage7_visual_interpretation.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from stage6_cnn_experiments import AE1D, AE2D, recurrence
from common import ROOT, CONTINUOUS

OUT=ROOT/'output'/'cnn'; torch.set_num_threads(2)
frame=pd.read_csv(ROOT/'data/processed/bandung_weather_hourly_clean.csv',parse_dates=['time'])
raw=frame[CONTINUOUS].to_numpy(dtype='float32')
train=frame.time.dt.year.to_numpy()<=2023
scaler=StandardScaler().fit(raw[train]); scaled=scaler.transform(raw).astype('float32')
test=np.flatnonzero(frame.time.dt.year.to_numpy()==2025)
start=int(test[0]); end=start+48
x1=torch.from_numpy(scaled[start:end].T[None].copy())
x2=torch.from_numpy(recurrence(raw[start:end,0])[None,None].copy())
figs=[]
def save(name,caption):
 plt.tight_layout();plt.savefig(OUT/name,dpi=175,bbox_inches='tight');plt.close();figs.append({'file':name,'caption':caption})

m1=AE1D(len(CONTINUOUS));m1.load_state_dict(torch.load(OUT/'cnn1d.pt',map_location='cpu',weights_only=True)['state_dict']);m1.eval()
m2=AE2D();m2.load_state_dict(torch.load(OUT/'cnn2d.pt',map_location='cpu',weights_only=True)['state_dict']);m2.eval()
with torch.no_grad():
 r1,z1=m1(x1); a11=m1.encoder[1](m1.encoder[0](x1))[0].numpy(); a12=m1.encoder[:5](x1)[0].numpy()
 r2,z2=m2(x2); a21=m2.encoder[1](m2.encoder[0](x2))[0].numpy(); a22=m2.encoder[:5](x2)[0].numpy()

hours=np.arange(48)
fig,axs=plt.subplots(2,1,figsize=(10,6),sharex=True)
axs[0].plot(hours,raw[start:end,0],label='Temperature (°C)',color='tab:blue');axs[0].set(ylabel='°C',title='Original temperature and CNN 1D responses, first 48h of 2025');axs[0].legend()
image=axs[1].imshow(a11[:8],aspect='auto',origin='lower',extent=[0,47,0,8],cmap='viridis');axs[1].set(xlabel='Hour in input window',ylabel='Filter index (first 8)');fig.colorbar(image,ax=axs[1],label='ReLU activation (arbitrary units)');save('cnn1d_temporal_response.png','Temperature and first-layer activation on the same time axis; visual co-occurrence is not a learned semantic label.')

fig,axs=plt.subplots(2,1,figsize=(10,5));axs[0].imshow(a11,aspect='auto',cmap='viridis');axs[0].set(title='CNN 1D: first Conv1D + ReLU (16 filters, 48 positions)',ylabel='Filter',xlabel='Hour');axs[1].imshow(a12,aspect='auto',cmap='viridis');axs[1].set(title='CNN 1D: second Conv1D + ReLU (8 filters, 24 positions)',ylabel='Filter',xlabel='Pooled time position');save('cnn1d_intermediate_layers.png','Intermediate feature maps at two depths; temporal resolution changes after pooling.')

fig,axs=plt.subplots(2,3,figsize=(11,7));
for i,ax in enumerate(axs.flat):
 ax.imshow(a21[i],cmap='viridis',origin='lower');ax.set(title=f'Filter {i+1}',xlabel='Resampled time index',ylabel='Resampled time index')
fig.suptitle('CNN 2D: first-layer feature maps on recurrence similarity image');save('cnn2d_feature_map_grid.png','Six first-layer CNN 2D feature maps; image axes are pairs of time indices, not physical spatial coordinates.')

fig,axs=plt.subplots(1,3,figsize=(12,4));
for ax,img,title in zip(axs,[x2[0,0].numpy(),a21[0],a22[0]],['Input recurrence similarity','First-layer activation, filter 1','Second-layer activation, filter 1']):
 ax.imshow(img,cmap='viridis',origin='lower');ax.set(title=title,xlabel='Time index',ylabel='Time index')
save('cnn2d_intermediate_layers.png','Visual comparison of input recurrence similarity and two CNN 2D intermediate activations; feature map values are not meteorological units.')

fig,axs=plt.subplots(2,1,figsize=(10,6),sharex=True)
for i,col in enumerate([0,1]):
 axs[i].plot(hours,scaled[start:end,col],label='Standardized input');axs[i].plot(hours,r1[0,col].detach().numpy(),label='CNN 1D reconstruction');axs[i].set(ylabel='Standardized units',title=CONTINUOUS[col]);axs[i].legend()
axs[-1].set_xlabel('Hour in window');save('cnn1d_reconstruction_lines.png','Two example feature reconstructions on standardized scale; discrepancies are visible, and reconstruction is not forecasting.')

fig,axs=plt.subplots(1,3,figsize=(11,3.5))
for ax,im,title in zip(axs,[x2[0,0].numpy(),r2[0,0].detach().numpy(),np.abs(x2[0,0].numpy()-r2[0,0].detach().numpy())],['Input similarity','Reconstruction','Absolute error']):
 ax.imshow(im,cmap='viridis',origin='lower');ax.set(title=title,xlabel='Time index',ylabel='Time index')
save('cnn2d_reconstruction_error.png','Input, reconstruction, and pixelwise absolute error in the continuous recurrence similarity image.')

(OUT/'visual_interpretation_manifest.json').write_text(json.dumps({'source_checkpoints':['cnn1d.pt','cnn2d.pt'],'test_window_start':str(frame.iloc[start].time),'test_window_hours':48,'figures':figs,'limitations':['Single example window; not representative of all weather conditions','Activation does not imply causal explanation','No retraining or new quantitative model evaluation','2D input is univariate temperature recurrence similarity']},indent=2),encoding='utf-8')
print('Generated',len(figs),'visual interpretation figures')
