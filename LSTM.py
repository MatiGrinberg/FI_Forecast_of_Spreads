import os
path = r"\\solon.prd\files\P\Global\Users\C63954\UserData\Desktop\Work_fromARG_Covid\Fi"
f,word_file='Spreads.xlsx','Input_graphs.docx'
os.chdir(path)
from LSTM_fun_par import *

"""
# read file
df= pd.read_excel(f,sheet_name='Data',header=2).set_index('Dates')
df = df.sort_index().round(decimals=2).drop_duplicates()
null_col = round(df.isnull().sum()/len(df),2)
null_col = null_col[null_col>0.2].index
df.drop(null_col,axis=1,inplace=True)
null_rows(df)
df.dropna(inplace=True)
print(df.shape)
print(df)
save_pickle(df,'df.pickle')
"""
# PICK INDEX
#titles = ['EU HY' , 'US HY', 'Cembi', 'Embi', 'EU IG']
titles = ['HY','EM','EU IG']
title = titles[1]

df = read_pickle('df.pickle')#.dropna()
df['EMUSTRUU Index']=df['EMUSTRUU Index']-df['LUATTRUU Index']

col = df.columns.tolist()
perc = col[0:6]+col[8:10]
vola = col[6:8]+['USGGBE10 Index']
targ = ['LECPOAS Index','EMUSTRUU Index','LG50TRUU Index']
retc = col[:3]

# Percent changes
df_perc=pd.DataFrame()
for col in perc:
    for ran in ranges:
        nam = col.replace('Index','%_')+str(ran)
        df_perc[nam] = df[col].pct_change(periods=ran)
# Volatility
df_std=pd.DataFrame()
for col in vola:
    for ran in ranges:
        nam = col.replace('Index','Vol_')+str(ran)
        df_std[nam] = df[col].rolling(ran).std()/df[col].rolling(ran).mean()
# Concat & define_Targets
dfy = round(df[targ].pct_change(periods=days_fut),2).shift(-days_fut)
dfy_c= [c+'_Target' for c in dfy.columns]
dfy.columns = dfy_c
df_ret = df[retc].pct_change(periods=1).shift(-1).dropna()
df.drop(perc,axis=1,inplace=True)
df = pd.concat([df,dfy,df_std,df_perc],axis=1)
df_fcst = df[df[dfy_c].isnull().any(axis=1)].drop(dfy_c,axis=1)
df.dropna(inplace=True)
col_tar,drop,d_fil,ts = index_dic[title][0]+'_Target',index_dic[title][1],index_dic[title][2],index_dic[title][4]
df=df[df.index >= d_fil]
df = target(df,th,col_tar)
df.drop([c for c in dfy_c if c not in col_tar],axis=1,inplace=True)
c_inf,r_inf = check_inf(df)
df.drop(c_inf,axis=1,inplace=True)
c_inf,r_inf = check_inf(df_fcst)
df_fcst.drop(c_inf,axis=1,inplace=True)
df.drop(r_inf,inplace=True)

# Run Model
y_tr,y_v,y_ts,x_tr,x_ts,x_v,fcst,spli = prepare_df(df,col_tar,df_fcst,days_fut,ts,val_spl=0.9)
#print('\nTarget:',col_tar,'\nTs:',ts,'\nDrop:',drop,'\nMA:',ma_signal,'\nTs_Cut:',df.iloc[spli,:].name,'\nDate_Start:',d_fil,'\n')
print('\nTarget:{} Ts:{} Drop:{} MA:{} Ts_Cut:{} Date_Start:{}\n'.format(col_tar.replace(' Index_Target',''),ts,drop,ma_signal,str(df.iloc[spli,:].name.date()),d_fil))
model = run(x_tr,y_tr,x_v,y_v,drop,ep=epochs)

# Predictions
pred,y_ts,y_fcst,pred_tr = np.argmax(model.predict(x_ts),axis=1)-1,np.argmax(y_ts,axis=1)-1,np.argmax(model.predict(fcst),axis=1)-1,np.argmax(model.predict(x_tr),axis=1)-1
df_init = pd.DataFrame(df.iloc[ts-1:int(ts-1+len(x_tr)),:][index_dic[title][0]])
df_init["Pred"]=pred_tr
df_pred = pd.DataFrame(df.iloc[-len(x_ts):,:][index_dic[title][0]])
df_pred['Pred']=pred 
df_fcs = pd.DataFrame(df_fcst[index_dic[title][0]])
df_fcs['Pred']=y_fcst
l_tr,l_pr,l_fc = len(df_init),len(df_pred),len(df_fcs)
# Df_Graphs
df_graph = pd.concat([df_init,df_pred,df_fcs],axis=0)
df_graph['Pred'] = df_graph['Pred'].rolling(ma_signal).mean()
# Df_Returns
df_rets=pd.merge(df_ret[index_dic[title][3]],df_graph['Pred'],left_index=True,right_index=True)
df_rets=df_rets.iloc[-int(l_pr+l_fc):,:]
df_rets['Return']=df_rets[index_dic[title][3]]*-df_rets['Pred']
df_rets['Cum']=(1+df_rets.Return).cumprod()
# Graphs
conf_ma(y_ts,pred,title+'_Acc:'+str(round(accuracy_score(y_ts,pred),2)))
line_gr(df_graph,title,'_Mov_Avg_Ts+Fcst',l_tr,l_pr,l_fc,0.99)
#line_gr(df_graph,title,'_Mov_Avg_All',l_tr,l_pr,l_fc,0)
#rets_gr(df_rets,title)

# Print to Word
#copy(word_file,[title+'_Mov_Avg_Ts+Fcst.png'])

