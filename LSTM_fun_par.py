from Libraries import *
import os

# PARAMETERS
val_spl=0.9
tr_spl = val_spl-0.1
units,opt,th,ranges,loss,epochs,early_loss_patience,monitor_loss,sc,days_fut,ma_signal,today = 100,'Adam',0.05,[90],"categorical_crossentropy",1000,20,'loss',MinMaxScaler(), 120,15,dt.now().strftime("%d-%m-%Y")
#location_column,location_row ,location_column_fcst,title_hist= ['B','M','Y','B','M','Y','B','M','Y','B','M','Y','B','M','Y','B','M','Y','B','M','Y'],['4','4','4','42','42','42','80','80','80','120','120','120','160','160','160','200','200','200','240','240','240'],['B','M','Y','AJ','AU'],'changes in spreads when predicted as: '
y_lab_size,font_sc,font,title_size,linewidth,sz,graphs,sz_w,f_w=20,1.2,25,24,2,10,[],6,22
es=EarlyStopping(monitor=monitor_loss, verbose=1,patience=early_loss_patience)
index_dic={'HY':['LG50TRUU Index',0.0,'2001-03-01','LF98TRUU Index',60],
           'EU IG':['LECPOAS Index',0.0,'2001-03-01','LECPTREU Index',30], 
           'EM':['EMUSTRUU Index',0.0,'2001-03-01','JPGCHECP Index',60]}

"""
index_dic={'HY':['LG50TRUU Index',0.0,'2011-03-01','LF98TRUU Index',60],
           'EU IG':['LECPOAS Index',0.0,'2013-03-01','LECPTREU Index',30],
           'EM':['EMUSTRUU Index',0.0,'2013-03-01','JPGCHECP Index',60]}
"""

# FUNCTIONS
def null_rows(d):
    return d[d.isnull().any(axis=1)]

def colu(d):
    return d.columns.tolist()

def copy(f,g_l):
    r= Document(f)
    para = r.add_paragraph().add_run(today)
    para.font.size = Pt(f_w)
    for g in g_l:
        r.add_picture(g,width=Inches(sz_w), height=Inches(sz_w))
    r.save(f)


def conf_ma(y,p,title):
    conf_mat = np.round(confusion_matrix(y,p)/len(y),2)
    sns.set(font_scale=font_sc, rc={'figure.figsize':(sz,sz)})
    ax = sns.heatmap(conf_mat,annot=True)
    ax.set_title('Conf_Matrix_in_%_'+title)
    ax.title.set_size(30)
    ax.set_xlabel('Predictions',fontsize=font,color='b')
    ax.set_ylabel('Actual',fontsize=font,color='b')
    ax.xaxis.set_ticklabels(['-1','0','1'], fontsize = font*.75)
    ax.yaxis.set_ticklabels(['-1','0','1'], fontsize = font*.75)
    plt.savefig('conf_mat.png',bbox_inches='tight')
    graphs.append('conf_mat.png')
    plt.show()

def line_gr(d,tit,x_tit,l_tr,l_pr,l_f,frac):
    d_tr,d_pr,d_fc = d.iloc[int(frac*l_tr):l_tr,:],d.iloc[l_tr:int(l_tr+l_pr),:],d.iloc[-l_f:,:]
    d_sp=d.iloc[int(frac*l_tr):,:]
    fig, ax1 = plt.subplots()
    ax1.plot(d_sp.index, d_sp[index_dic[tit][0]], color = 'black') 
    ax1.set_ylabel('Spreads', color = 'black') 
    ax1.tick_params(axis ='y', labelcolor = 'black') 
    ax2 = ax1.twinx()
    ax2.plot(d_tr.index,d_tr.Pred,color='cyan',label='Train',linewidth=2) 
    ax2.plot(d_pr.index,d_pr.Pred,color='blue',label='Test',linewidth=2) 
    ax2.plot(d_fc.index,d_fc.Pred,color='red',label='Fcst',linewidth=3)
    ax2.axhline(y = 0, color = 'y', linestyle = ':',linewidth= 7)
    ax2.set_ylabel('Signal', color = 'green')
    ax2.tick_params(axis ='y', labelcolor = 'green')
    tit+=x_tit
    plt.legend()
    plt.title(tit,fontsize=30)
    plt.savefig(tit+'.png',bbox_inches='tight')
    graphs.append(tit+'.png')
    plt.show()

def rets_gr(d,tit):
    fig, ax1 = plt.subplots()
    ax1.plot(d.index, d['Cum'], color = 'black',linewidth= 2)
    plt.axhline(y = 1, color = 'r', linestyle = '--', label='always neutral (passive)',linewidth= 2)
    ax1.set_ylabel('Cum_Return', color = 'black') 
    ax1.tick_params(axis ='y', labelcolor = 'black')
    plt.legend()#loc="upper left"
    titu=tit+'_Ret'
    plt.title(titu,fontsize=30)
    plt.savefig(titu,bbox_inches='tight')
    graphs.append(titu+'.png')
    plt.show()

    
def run(x_tr,y_tr,x_v,y_v,drop,ep=epochs):
    model = Sequential()
    model.add(LSTM(units = int(units),dropout=drop, return_sequences = True, input_shape = (x_tr.shape[1], x_tr.shape[2])))
    model.add(LSTM(units = int(units/2),dropout=drop, return_sequences = True))
    model.add(LSTM(units = int(units/4),dropout=drop, return_sequences = True))
    model.add(LSTM(units = int(units/8),dropout=drop))
    model.add(Dense(units = 3,activation='softmax'))
    model.compile(optimizer = opt, loss = loss)
    #model.fit(x_tr,y_tr,epochs=ep,callbacks=[es],validation_data=(x_v, y_v))
    model.fit(x_tr,y_tr,epochs=ep,callbacks=[es]) # if Skipping Validat_Set
    return model

def prepare_df(df,c,df_fc,d,ts,val_spl=0.9):
    y_dumm = df.pop(c)
    y_dumm = y_dumm[ts-1:]
    y = pd.get_dummies(y_dumm).values
    df = pd.concat([df,df_fc],axis=0)
    x = sc.fit_transform(df)
    i,x_arr = 0,[]
    while i < len(x)-ts+1:
        x_arr.append(x[i:i+ts,:])
        i += 1
    x_arr=np.array(x_arr)
    fcst = x_arr[-d:]
    x_arr = x_arr[:-d]
    n = len(x_arr)
    tr_cut,v_cut = int(n*tr_spl),int(n*val_spl)
    x_tr,x_v,x_ts,y_tr,y_v,y_ts = x_arr[0:tr_cut],x_arr[tr_cut:v_cut],x_arr[v_cut:],y[0:tr_cut],y[tr_cut:v_cut],y[v_cut:]
    x_tr,y_tr = np.concatenate((x_tr,x_v), axis=0),np.concatenate((y_tr,y_v), axis=0) # if skipping Validat_Set
    return y_tr,y_v,y_ts,x_tr,x_ts,x_v,fcst,v_cut


def save_pickle(df,p):
    pickle_out = open(p,"wb")
    pickle.dump(df, pickle_out)
    pickle_out.close()

def read_pickle(p):
    pickle_in = open(p,"rb")
    df = pickle.load(pickle_in)
    return df
    

def target(df,th,col):
    df[col][df[col]<-th]=-1
    df[col][(df[col]>=-th)&(df[col]<=th)]=0
    df[col][df[col]>th]=1
    return df

def toy_df(mn,mx,l,w,col):
    df = pd.DataFrame(np.random.randint(mn,mx,size=(l,w)), columns=list(col))
    return df

def toy_df_fl(mn,mx,l,w,col):
    df = pd.DataFrame(np.random.uniform(mn,mx,size=(l,w)), columns=list(col))
    return df


def check_inf(d):
    col_name = d.columns.to_series()[np.isinf(d).any()]
    r = d.index[np.isinf(d).any(axis=1)]
    print('Rows\n{}\nCol\n{}'.format(r,col_name))
    return col_name,r

def pr_ty_sh(ele):
    print('Type: {} Shape: {}'.format(type(ele),ele.shape))


def hist_perc(df,c):
    plt.hist(df[c],density=True)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.show()






























"""

def hist(d):
    test_original = d['DFs'][1]
    num = 6
    for i in d['Titles'][1]:
        fig, ax = plt.subplots(figsize =(size,size))
        ax.hist(test_original[d['Columns_DFs'][2]][test_original[d['Columns_DFs'][5]]==i])
        plt.title(title_hist+i, fontsize = font)
        plt.savefig(graphs_png[num])
        graphs_save_excel.append(graphs_png[num])
        plt.show()
        num+=1


def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return round(np.mean(np.abs((actual - pred) / actual)) * 100,0)


def division(a,b):
    try:
        c = round(a/b,2)
    except:
        c = 'nan'
    return c


def summary(test_original,y_test,th,ts,y_train,y_val,first_date,val_date_words,ir):
    error = mape(test_original['Next Q '],test_original['pred_spreads'])
    corr = np.round_(np.corrcoef(test_original['Next Q '],test_original['pred_spreads']),decimals=2)
    df_pd = test_original[test_original['pred_dir']==d]
    df_pu = test_original[test_original['pred_dir']==u]
    df_prd = test_original[(test_original['Direction']==d)&(test_original['pred_dir']==d)]
    df_pru = test_original[(test_original['Direction']==u)&(test_original['pred_dir']==u)]
    df_pwd = test_original[(test_original['Direction']==u)&(test_original['pred_dir']==d)]
    df_pwu = test_original[(test_original['Direction']==d)&(test_original['pred_dir']==u)]
    prd=division(len(df_prd),len(df_pd))
    pwd =division(len(df_pwd),len(df_pd))
    ratio_d=division(prd,pwd)
    pru=division(len(df_pru),len(df_pu))
    pwu=division(len(df_pwu),len(df_pu))
    ratio_u=division(pru,pwu)
    # Sum of corners
    total_right = len(df_prd)+len(df_pru)-len(df_pwd)-len(df_pwu)
    # dataframe with values to export
    col_summary = ['threshold_neutral','timesteps ('+str(frequency_date)+' days)','samples_train', 'samples_val','samples_test','train from','validation date','test from','%_pred_down_right','%_pred_down_wrong','down_Right/Wrong','%_pred_up_right','%_pred_up_wrong','up_Right/Wrong','Q Right - Wrong','IR','MAPE','Corr']
    summary_data = [th,ts,y_train.shape[0], y_val.shape[0],y_test.shape[0],first_date,val_date_words,date_split,prd,pwd,ratio_d, pru,pwu,ratio_u,total_right,ir,error,corr[0][1]]
    summary_df = pd.DataFrame([summary_data],columns=col_summary)
    return summary_df


def df_res(df,picked,th,pred,df_mov_aver,returns):
    test_original = df[(df['Date']>date_split)&(df['Explanation Index']==picked)][columns_keep_graphs]
    test_original.reset_index(drop=True, inplace=True)
    test_original = threshold_changes_spreads(test_original,th)
    #test_original = test_original.iloc[ts-1:,:]
    test_original['pred_spreads']=pred.round()
    test_original['dif']=test_original['pred_spreads']-test_original['Next Q ']
    test_original['abs_dif']=abs(test_original['dif'])
    test_original['pred_always_long']='Down'
    for i in range(df_mov_aver.shape[1]):
        aver_spread = df_mov_aver.iloc[:,i]
        test_original['aver_spreads_'+aver_spread.name]=aver_spread
    test_original = calculate_directions(test_original,'pred_spreads','pred_direction','pred_dir',th)   
    test_original = calculate_signals(test_original,'pred_signal','pred_dir')
    test_original = calculate_signals(test_original,'actual_signal','Direction')
    for ma in [3,6,12,24]:
            test_original = calculate_directions(test_original,'aver_spreads_'+str(ma),'pred_aver_'+str(ma),'pred_average_'+str(ma),th)
    test_original = pd.concat([test_original,returns.iloc[:,-1]], axis=1)
    for i in returns_titles:
        col = 'indiv_'+i
        test_original[col]=0
        test_original[col][test_original[i]=="Down"]=test_original[picked+'_Returns']/100
        test_original[col][test_original[i]=="Neutral"]=0
        test_original[col][test_original[i]=="Up"]=-1*test_original[picked+'_Returns']/100
        col2 =  'cum_'+i
        # check if returns divided by 100
        test_original[col2] = (test_original[col] + 1).cumprod()
        test_original['Vs_'+i]=test_original['indiv_pred_dir'] - test_original[col]
    test_original = average_signals(test_original,'pred_signal',picked+'_Returns','mov_aver_pred_signal','indiv_aver_signal_return',6,'cum_aver_signal_return')
    ir = IR(test_original,'pred_dir','pred_always_long')
    ir_aver_signal = IR(test_original,'aver_signal_return','pred_always_long')
    return test_original, ir,ir_aver_signal


def average_signals(df,my_pred,index_ret,ma_signal,ma_return,steps,cum_ret):
    df[ma_signal]=df[my_pred].rolling(steps).mean()
    df[ma_return]=(df[index_ret]/100)*-1*df[ma_signal]
    df[cum_ret]=(df[ma_return]+1).cumprod()
    return df

def IR(df,portf,bench):
    periods = len(df)
    dif_CAGR = (df['cum_'+portf].iloc[-1]**(1/periods)-1)-(df['cum_'+bench].iloc[-1]**(1/periods)-1)
    std_dev_returns = np.std(df['indiv_'+portf]-df['indiv_'+bench])
    ir = round(dif_CAGR/std_dev_returns,2)
    return ir

def make_all_windows(one,two,three,four,df):
    df_aver_spreads = window(one,df)   
    aver_spreads_3 = df_aver_spreads['aver_spreads'][df_aver_spreads['Date']>date_split]
    #aver_spreads_3 = aver_spreads_3[ts-1:]
    aver_spreads_3.reset_index(drop=True, inplace=True)
    df_aver_spreads = window(two,df)
    aver_spreads_6 = df_aver_spreads['aver_spreads'][df_aver_spreads['Date']>date_split]
    #aver_spreads_6 = aver_spreads_6[ts-1:]
    aver_spreads_6.reset_index(drop=True, inplace=True)
    df_aver_spreads = window(three,df)
    aver_spreads_12 = df_aver_spreads['aver_spreads'][df_aver_spreads['Date']>date_split]
    #aver_spreads_12 = aver_spreads_12[ts-1:]
    aver_spreads_12.reset_index(drop=True, inplace=True)
    df_aver_spreads = window(four,df)
    aver_spreads_24 = df_aver_spreads['aver_spreads'][df_aver_spreads['Date']>date_split]
    #aver_spreads_24 = aver_spreads_24[ts-1:]
    aver_spreads_24.reset_index(drop=True, inplace=True)
    return aver_spreads_3,aver_spreads_6,aver_spreads_12,aver_spreads_24


def predict(m,y_ts,x_ts,col,sc ):
    fcst = m.predict(x_ts)
    y_ts = [[i] for i in y_ts]
    y_test_copy = np.repeat(y_ts,len(col),axis=1)
    fcst_copy = np.repeat(fcst,len(col),axis=1)
    pred = sc.inverse_transform(fcst_copy)[:,0]
    y_ts = sc.inverse_transform(y_test_copy)[:,0]
    return y_test,pred

  
def window(months,df):
    df_aver_spreads = df[['Date','Spreads today']]
    df_aver_spreads['aver_spreads'] = 0
    w = months*intervals
    for i in range(len(df_aver_spreads)-w+1):
        init = i+w-1
        df_aver_spreads.iloc[init,2] = round(np.mean(df_aver_spreads.iloc[i:init+1,1]),0)
    return df_aver_spreads[['Date','aver_spreads']]


def df_settings_display():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

def analyze(df,rows,col):
    print(df.iloc[:rows,:col],'\n********************')
    print(df.iloc[-rows:,:col],'\n********************')
    print(df.shape,'\n********************')
    print(df.isna().values.sum(),'\n********************')
    #print(df.columns.tolist(),'\n********************')
    
def pick(possibility_order,returns):    
    picked = possibilities[possibility_order]
    title = 'predictions 3 months ahead - '+picked
    ts = ts_possibilities[possibility_order]
    units = units_possibilities[possibility_order]
    date_filter = date_filter_possibilities[possibility_order]
    epochs = epochs_possibilities[possibility_order]
    optimizer = optimizer_possibilities[possibility_order]
    th = th_possibilities[possibility_order]
    drop = drop_possibilities[possibility_order]
    returns_columns = returns.columns.tolist()
    returns = returns[[returns_columns[0],returns_columns[possibility_order+1]]]
    pickle_mov_aver = picked+"_Moving_Average.pickle"
    return title, picked, ts, units, date_filter, epochs, optimizer, th, drop,returns,pickle_mov_aver


def preparation(df,returns,date_filter,index,th):
    returns = returns[returns['Date']>date_split].reset_index(drop=True)
    #returns = returns.iloc[ts-1:,:]
    df = df[(df['Date']>date_filter)&(df['Explanation Index']==index)]
    df.reset_index(drop=True, inplace=True)
    df = threshold_changes_spreads(df,th)
    print(pd.isnull(df).sum()[pd.isnull(df).sum()>0])
    df = df.drop(labels=to_drop, axis=1)
    return df, returns

  
"""
    







