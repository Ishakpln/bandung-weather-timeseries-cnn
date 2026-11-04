from pathlib import Path
import json, time, random, platform
import numpy as np, pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/bandung_weather_hourly_clean.csv'
OUT=ROOT/'output/cnn'
FIG=OUT/'figures'; TAB=OUT/'tables'; MODELS=OUT/'models'
for p in [FIG,TAB,MODELS]: p.mkdir(parents=True,exist_ok=True)
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
DEVICE='cpu'
LOOKBACKS=[24,48,72]
HORIZONS=[1,6,24]
EPOCHS=10; BATCH=256; LR=1e-3; PATIENCE=3
CONT=['temperature_2m','relative_humidity_2m','dew_point_2m','precipitation','pressure_msl','cloud_cover','wind_speed_10m']
CYCL=['hour_sin','hour_cos','month_sin','month_cos','day_of_year_sin','day_of_year_cos']
# weather_code is nominal; one-hot categories are fitted using training period only.
df=pd.read_csv(DATA)
df['time']=pd.to_datetime(df['time'],utc=True).dt.tz_convert('Asia/Jakarta')
df=df.sort_values('time').reset_index(drop=True)
assert df.time.is_monotonic_increasing and not df.time.duplicated().any()
# fixed chronological split by calendar year; each sample's input and label must reside in same split
splits={'train':(df.time.iloc[0],pd.Timestamp('2023-12-31 23:00:00',tz='Asia/Jakarta')),
         'validation':(pd.Timestamp('2024-01-01 00:00:00',tz='Asia/Jakarta'),pd.Timestamp('2024-12-31 23:00:00',tz='Asia/Jakarta')),
         'test':(pd.Timestamp('2025-01-01 00:00:00',tz='Asia/Jakarta'),df.time.iloc[-1])}
records=[]; histories={}; prediction_store={}
for horizon in HORIZONS:
 for lookback in LOOKBACKS:
  print(f'RUN L={lookback} H={horizon}',flush=True)
  # Fit numeric and category encoders only on train rows
  trainmask=(df.time>=splits['train'][0])&(df.time<=splits['train'][1])
  valmask=(df.time>=splits['validation'][0])&(df.time<=splits['validation'][1])
  testmask=(df.time>=splits['test'][0])&(df.time<=splits['test'][1])
  tr=df.loc[trainmask].copy(); va=df.loc[valmask].copy(); te=df.loc[testmask].copy()
  cats=sorted(tr.weather_code.astype(int).unique().tolist())
  def matrix(part):
   base=part[CONT+CYCL].astype(float).to_numpy()
   cat=part.weather_code.astype(int).to_numpy()
   oh=np.zeros((len(part),len(cats)),dtype=np.float32)
   mp={c:i for i,c in enumerate(cats)}
   for j,c in enumerate(cat):
    if c in mp: oh[j,mp[c]]=1
   return np.concatenate([base,oh],axis=1).astype(np.float32)
  Xtr0=matrix(tr); Xva0=matrix(va); Xte0=matrix(te)
  sx=StandardScaler().fit(Xtr0)
  sy=StandardScaler().fit(tr[['temperature_2m']].to_numpy())
  Xtr=sx.transform(Xtr0).astype(np.float32); Xva=sx.transform(Xva0).astype(np.float32); Xte=sx.transform(Xte0).astype(np.float32)
  ytr=sy.transform(tr[['temperature_2m']]).astype(np.float32).ravel()
  yva=sy.transform(va[['temperature_2m']]).astype(np.float32).ravel()
  yte=sy.transform(te[['temperature_2m']]).astype(np.float32).ravel()
  # Build sliding windows lazily to control RAM use; windows stay inside each temporal partition.
  ntr=len(Xtr)-lookback-horizon+1; nva=len(Xva)-lookback-horizon+1; nte=len(Xte)-lookback-horizon+1
  class SeqDataset(torch.utils.data.Dataset):
   def __init__(self,X,y,L,H,n,stride=3): self.X=X; self.y=y; self.L=L; self.H=H; self.n=n; self.stride=stride
   def __len__(self): return (self.n+self.stride-1)//self.stride
   def __getitem__(self,i):
    i=i*self.stride
    return torch.from_numpy(self.X[i:i+self.L]),torch.tensor(self.y[i+self.L+self.H-1])
  train_ds=SeqDataset(Xtr,ytr,lookback,horizon,ntr,stride=3)
  def make_batch(X,y,indices,L,H):
   xb=np.stack([X[i:i+L] for i in indices])
   yb=y[indices+L+H-1]
   return torch.from_numpy(xb),torch.from_numpy(yb)
  def predict_all(model,X,y,n,L,H):
   outs=[]
   with torch.no_grad():
    for i in range(0,n,BATCH):
     ids=np.arange(i,min(i+BATCH,n)); xb,_=make_batch(X,y,ids,L,H)
     outs.append(model(xb).numpy())
   return np.concatenate(outs)
  Y_val=yva[np.arange(nva)+lookback+horizon-1]; Y_test=yte[np.arange(nte)+lookback+horizon-1]
  T_test=te.time.iloc[np.arange(nte)+lookback+horizon-1].reset_index(drop=True)
  class CNN(nn.Module):
   def __init__(self,features):
    super().__init__()
    self.net=nn.Sequential(
     nn.Conv1d(features,16,kernel_size=5,padding='same'),nn.ReLU(),
     nn.Conv1d(16,32,kernel_size=3,padding='same'),nn.ReLU(),
     nn.AdaptiveAvgPool1d(1),nn.Flatten(),
     nn.Linear(32,16),nn.ReLU(),nn.Dropout(0.15),nn.Linear(16,1))
   def forward(self,x): return self.net(x.transpose(1,2)).squeeze(-1)
  model=CNN(Xtr.shape[-1]).to(DEVICE)
  opt=torch.optim.Adam(model.parameters(),lr=LR)
  lossfn=nn.MSELoss()
  loader=torch.utils.data.DataLoader(train_ds,batch_size=BATCH,shuffle=True)
  best=float('inf'); beststate=None; stale=0; hist={'loss':[],'val_loss':[]}
  start=time.time()
  for epoch in range(EPOCHS):
   model.train(); losses=[]
   for xb,yb in loader:
    opt.zero_grad(); pred=model(xb); loss=lossfn(pred,yb); loss.backward(); opt.step(); losses.append(loss.item())
   model.eval()
   with torch.no_grad():
    vp=[]
    for i in range(0,nva,BATCH):
     ids=np.arange(i,min(i+BATCH,nva)); xb,yb=make_batch(Xva,yva,ids,lookback,horizon)
     vp.append(lossfn(model(xb),yb).item())
   tl=float(np.mean(losses)); vl=float(np.mean(vp))
   hist['loss'].append(tl);hist['val_loss'].append(vl)
   if vl<best-1e-5:
    best=vl; beststate={k:v.cpu().clone() for k,v in model.state_dict().items()};stale=0
   else: stale+=1
   if stale>=PATIENCE: break
  elapsed=time.time()-start
  model.load_state_dict(beststate); model.eval()
  pred_scaled=predict_all(model,Xte,yte,nte,lookback,horizon)
  pred_orig=sy.inverse_transform(pred_scaled.reshape(-1,1)).ravel()
  actual=sy.inverse_transform(Y_test.reshape(-1,1)).ravel()
  mae=float(np.mean(np.abs(actual-pred_orig))); rmse=float(np.sqrt(np.mean((actual-pred_orig)**2)))
  # validation metric at selected best checkpoint
  pval=predict_all(model,Xva,yva,nva,lookback,horizon)
  av=sy.inverse_transform(Y_val.reshape(-1,1)).ravel(); pv=sy.inverse_transform(pval.reshape(-1,1)).ravel()
  vmae=float(np.mean(np.abs(av-pv))); vrmse=float(np.sqrt(np.mean((av-pv)**2)))
  tag=f'L{lookback}_H{horizon}'
  torch.save({'state_dict':beststate,'lookback':lookback,'horizon':horizon,'features':CONT+CYCL,'weather_code_categories':cats,'scaler_x_mean':sx.mean_.tolist(),'scaler_x_scale':sx.scale_.tolist(),'scaler_y_mean':sy.mean_.tolist(),'scaler_y_scale':sy.scale_.tolist()},MODELS/f'{tag}.pt')
  histories[tag]=hist
  records.append({'lookback_hours':lookback,'horizon_hours':horizon,'n_train_windows':ntr,'n_validation_windows':nva,'n_test_windows':nte,'epochs_run':len(hist['loss']),'best_val_loss_scaled':best,'validation_MAE_C':vmae,'validation_RMSE_C':vrmse,'test_MAE_C':mae,'test_RMSE_C':rmse,'training_seconds':elapsed,'parameters':sum(p.numel() for p in model.parameters())})
  prediction_store[tag]=pd.DataFrame({'time':T_test.astype(str),'actual_temperature_2m_C':actual,'predicted_temperature_2m_C':pred_orig,'error_C':pred_orig-actual})
  print(f' finished epochs={len(hist["loss"])} val MAE={vmae:.3f} test MAE={mae:.3f} RMSE={rmse:.3f} seconds={elapsed:.1f}',flush=True)
# Results
res=pd.DataFrame(records).sort_values(['horizon_hours','lookback_hours'])
res.to_csv(TAB/'cnn_experiment_results.csv',index=False)
with open(TAB/'training_history.json','w') as f: json.dump(histories,f,indent=2)
for tag,pr in prediction_store.items(): pr.to_csv(TAB/f'predictions_{tag}.csv',index=False)
# Select best lookback per horizon using validation MAE (not test)
best_by_h=res.loc[res.groupby('horizon_hours').validation_MAE_C.idxmin()].sort_values('horizon_hours')
best_by_h.to_csv(TAB/'selected_models_by_horizon.csv',index=False)
# Figures
plt.rcParams.update({'figure.dpi':150,'savefig.dpi':300,'font.size':10})
# loss plots all 9 compact small multiples? 3 figures one per horizon, curves per L
for h in HORIZONS:
 fig,ax=plt.subplots(figsize=(8,4.5))
 for L in LOOKBACKS:
  tag=f'L{L}_H{h}'
  ax.plot(histories[tag]['loss'],label=f'Train L={L}')
  ax.plot(histories[tag]['val_loss'],linestyle='--',label=f'Val L={L}')
 ax.set(title=f'Loss pelatihan CNN (horizon {h} jam)',xlabel='Epoch',ylabel='MSE (target terstandardisasi)')
 ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.25);fig.tight_layout();fig.savefig(FIG/f'cnn_loss_h{h}.png');plt.close(fig)
# test actual vs predictions for validation-selected best per horizon
for _,row in best_by_h.iterrows():
 tag=f'L{int(row.lookback_hours)}_H{int(row.horizon_hours)}'
 pr=prediction_store[tag]
 # plot 7-day segment around middle of test year
 n=min(24*7,len(pr)); startidx=max(0,(len(pr)-n)//2); sample=pr.iloc[startidx:startidx+n]
 fig,ax=plt.subplots(figsize=(11,4.2))
 ax.plot(pd.to_datetime(sample.time),sample.actual_temperature_2m_C,label='Aktual',linewidth=1.5)
 ax.plot(pd.to_datetime(sample.time),sample.predicted_temperature_2m_C,label='CNN',linewidth=1.3)
 ax.set(title=f'Aktual vs prediksi — {tag} (test 2025)',xlabel='Waktu (WIB)',ylabel='Temperatur (°C)')
 ax.legend();ax.grid(alpha=.25);fig.autofmt_xdate();fig.tight_layout();fig.savefig(FIG/f'actual_vs_pred_{tag}.png');plt.close(fig)
# compare metrics by horizon across lookbacks
for metric in ['test_MAE_C','test_RMSE_C']:
 fig,ax=plt.subplots(figsize=(7.5,4.3))
 for L in LOOKBACKS:
  ss=res[res.lookback_hours==L].sort_values('horizon_hours')
  ax.plot(ss.horizon_hours,ss[metric],marker='o',label=f'Lookback {L} jam')
 ax.set(title=f'Perbandingan {metric.replace("_C","")} CNN menurut horizon',xlabel='Forecast horizon (jam)',ylabel='Error (°C)')
 ax.set_xticks(HORIZONS);ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(FIG/f'{metric}_by_horizon.png');plt.close(fig)
# MAE/RMSE across lookback at each horizon
for metric in ['test_MAE_C','test_RMSE_C']:
 fig,ax=plt.subplots(figsize=(7.5,4.3))
 for h in HORIZONS:
  ss=res[res.horizon_hours==h].sort_values('lookback_hours')
  ax.plot(ss.lookback_hours,ss[metric],marker='o',label=f'Horizon {h} jam')
 ax.set(title=f'Sensitivitas {metric.replace("_C","")} terhadap lookback',xlabel='Lookback (jam)',ylabel='Error (°C)')
 ax.set_xticks(LOOKBACKS);ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(FIG/f'{metric}_by_lookback.png');plt.close(fig)
# dataset trend illustration
df.set_index('time').temperature_2m.resample('D').mean().plot(figsize=(12,4),title='Rata-rata temperatur harian Bandung (2022–2025)')
plt.ylabel('Temperatur (°C)');plt.xlabel('Tanggal (WIB)');plt.grid(alpha=.2);plt.tight_layout();plt.savefig(FIG/'temperature_daily_mean_2022_2025.png');plt.close()
# save model config and audit
config={'seed':SEED,'framework':f'PyTorch {torch.__version__}','device':DEVICE,'architecture':'Conv1D(16,k5)-ReLU-Conv1D(32,k3)-ReLU-GlobalAvgPool-Dense16-ReLU-Dropout0.15-Dense1','optimizer':'Adam','learning_rate':LR,'loss':'MSE standardized target','batch_size':BATCH,'training_window_stride':3,'max_epochs':EPOCHS,'early_stopping_patience':PATIENCE,'lookbacks_hours':LOOKBACKS,'horizons_hours':HORIZONS,'split':'train=2022-2023; validation=2024; test=2025 (within-segment windows only)','features_continuous':CONT,'features_cyclical':CYCL,'weather_code':'one-hot categories present in train; unseen categories map all-zero','scalers':'StandardScaler fit only on train; separate target scaler; inverse transformed metrics in °C','seed':SEED,'warning':'EDA from prior stages included 2025; this is a chronological test but not a fully unseen test period from the entire research workflow.'}
with open(TAB/'cnn_config.json','w') as f:json.dump(config,f,indent=2)
print('\nALL EXPERIMENTS COMPLETE')
print(res.to_string(index=False))
