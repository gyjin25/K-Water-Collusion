#!/usr/bin/env python
# coding: utf-8

# # New LS 220916 FINAL

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer
from scipy.stats import f

pd.set_option('Display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


# In[2]:


# dfReg = pd.read_excel("./교수님 전달 파일/0914 datafile excluded version.xlsx", sheet_name=None, header = 4)
dfReg = pd.read_excel("./교수님 전달 파일/0915 datafile excluded version dmaintain.xlsx", sheet_name=None, header = 4)


# In[3]:


dfReg.keys()


# In[4]:


raw = pd.concat(dfReg, ignore_index=True)


# In[5]:


df_raw = raw[['낙찰율(예대)','연도','기초금액','1순위투찰금액','업체수','담합','점검정비','공고명','공고번호','감정']]
df_raw.head(3)


# In[6]:


df_raw['점검정비'].sum()


# In[7]:


df_raw.tail(3)


# In[8]:


df_raw['담합'].sum() ,df_raw['점검정비'].sum() # 30 & 44 ok


# In[9]:


df_raw.isna().sum()


# In[10]:


df_raw.loc[(df_raw['점검정비'] == 1) & (df_raw['담합'] == 0) & (df_raw['연도'] != 2018)]


# ### 0. 유찰된 데이터 제거 및 확인

# In[11]:


df_temp1 = df_raw.loc[(df_raw['1순위투찰금액'] != 0) & (df_raw['낙찰율(예대)'].isna() != True), ['낙찰율(예대)','연도', '기초금액', '업체수', '담합', '점검정비','공고명','감정']]
df_temp1.head(3)
df_temp1.shape
df_temp1['담합'].sum() # 28


# In[12]:


df_temp1 = df_temp1.loc[(~(df_temp1['감정'].isna())) | (df_temp1['점검정비']==1), df_temp1.columns != '감정']
df_temp1.shape


# In[13]:


df_temp1['점검정비'].sum()


# In[14]:


df_temp1.loc[((df_temp1['담합'] == 1) & (df_temp1['낙찰율(예대)'] > 0.9))]


# In[15]:


""" 담합 데이터 2개 제거 """

df_temp1 = df_temp1.loc[~((df_temp1['담합'] == 1) & (df_temp1['낙찰율(예대)'] > 0.9))]
df_temp1.shape


# In[16]:


df_temp1.loc[df_temp1['기초금액'] < 1000000000] # 10억보다 작은 금액 존재 X


# In[17]:


df_temp1['연도'].unique()


# ### 1. 연도별 더미 생성

# In[28]:


df_temp2 = df_temp1.copy()
# df_temp2[['d2011','d2012','d2013','d2014','d2015','d2016','d2017','d2018','d2019','d2020']] = 0
df_temp2[['d2011','d2012','d2013','d2014','d2016','d2017','d2018','d2019','d2020']] = 0
df_temp2.loc[df_temp2['연도'] == 2011, 'd2011'] = 1 
df_temp2.loc[df_temp2['연도'] == 2012, 'd2012'] = 1 
df_temp2.loc[df_temp2['연도'] == 2013, 'd2013'] = 1 
df_temp2.loc[df_temp2['연도'] == 2014, 'd2014'] = 1 
df_temp2.loc[df_temp2['연도'] == 2015, 'd2014'] = 1 
df_temp2.loc[df_temp2['연도'] == 2016, 'd2016'] = 1 
df_temp2.loc[df_temp2['연도'] == 2017, 'd2017'] = 1 
df_temp2.loc[df_temp2['연도'] == 2018, 'd2018'] = 1 
df_temp2.loc[df_temp2['연도'] == 2019, 'd2019'] = 1 
df_temp2.loc[df_temp2['연도'] == 2020, 'd2020'] = 1 


# In[29]:


df_temp2.head(3)


# ### 2. 규모 더미 생성 (33% & 66% Quantile)

# In[30]:


df_temp2['기초금액'].sort_values().quantile([.33, .66])


# In[31]:


df_temp3 = df_temp2.copy()
df_temp3['d33'] = 0
df_temp3['d66'] = 0

df_temp3.loc[df_temp3['기초금액'] < df_temp2['기초금액'].sort_values().quantile(.33), 'd33'] = 1
df_temp3.loc[(df_temp3['기초금액'] > df_temp2['기초금액'].sort_values().quantile(.33)) & (df_temp3['기초금액'] < df_temp2['기초금액'].sort_values().quantile(.66)), 'd66'] = 1
df_temp3.shape


# ### 3. 업체수 규제 (완전 제거 & 20으로 고정)

# In[32]:


df_temp4 = df_temp3.copy()


# In[33]:


# 3-2. dnumfirm2 : 업체수 20 이상은 NAN 으로 값 줘서 추후 삭제
df_temp4['dnumfirm2'] = np.nan
df_temp4.loc[df_temp4['업체수'] <= 20, 'dnumfirm2'] = df_temp4.loc[df_temp4['업체수'] <= 20, '업체수']
df_temp4 = df_temp4.dropna()
df_temp4.shape


# ### 4. Linear Regression 준비 위한 dataframe 재정비

# In[36]:


# df_temp5 = df_temp4[['낙찰율(예대)','담합', '기초금액','dnumfirm2','d2011','d2012','d2013','d2014','d2015','d2016','d2017','d2018','d2019','d2020','d33','d66','점검정비','공고명','연도']]
df_temp5 = df_temp4[['낙찰율(예대)','담합', '기초금액','dnumfirm2','d2011','d2012','d2013','d2014','d2016','d2017','d2018','d2019','d2020','d33','d66','점검정비','공고명','연도']]
df_temp5 = df_temp5.copy()
df_temp5['낙찰율'] = df_temp5['낙찰율(예대)']*100
df_temp5['ln기초금액'] = np.log(df_temp5['기초금액'])
df_temp5 = df_temp5.rename(columns = {"낙찰율" : "tenderRatio", "담합" : "Collusion", "ln기초금액" : "LnCash", "점검정비" : "dmaintain"})
df_temp5.head()


# In[37]:


df_temp5['dnumfirm2'].isna().sum() #60
df_temp5.shape # 640


# In[38]:


df_temp5.isna().sum()


# In[39]:


df_temp5['Collusion'].sum()


# ### 4-1. 연도별 더미 추가 생성

# In[40]:


df_temp5.head(3)


# In[41]:


df_temp5.loc[df_temp5['공고명'].str.contains("점검정비")].shape


# In[43]:


df_temp6 = df_temp5.copy() 

df_temp6[['db2013','db2014','db2017_a', 'db2017_b']] = 0

df_temp6.loc[df_temp6['d2011'] == 1, 'db2013'] =1
df_temp6.loc[df_temp6['d2012'] == 1, 'db2013'] =1
df_temp6.loc[df_temp6['d2013'] == 1, 'db2013'] =1

df_temp6.loc[df_temp6['d2011'] == 1, 'db2014'] =1
df_temp6.loc[df_temp6['d2012'] == 1, 'db2014'] =1
df_temp6.loc[df_temp6['d2013'] == 1, 'db2014'] =1
df_temp6.loc[df_temp6['d2014'] == 1, 'db2014'] =1

df_temp6.loc[df_temp6['d2014'] == 1, 'db2017_a'] =1
# df_temp6.loc[df_temp6['d2015'] == 1, 'db2017_a'] =1
df_temp6.loc[df_temp6['d2016'] == 1, 'db2017_a'] =1
df_temp6.loc[df_temp6['d2017'] == 1, 'db2017_a'] =1

# df_temp6.loc[df_temp6['d2015'] == 1, 'db2017_b'] =1
df_temp6.loc[df_temp6['d2016'] == 1, 'db2017_b'] =1
df_temp6.loc[df_temp6['d2017'] == 1, 'db2017_b'] =1


# ### 6. Generating X matrix for fitted value

# In[44]:


df_xmatrix = df_temp6.loc[(df_temp6['Collusion'] == 1) & (df_temp6['tenderRatio'] < 90)]
#df_xmatrix = df_xmatrix.drop(['Collusion'], axis = 1)
df_xmatrix.shape


# ### 7. 연도 이전 이후 더미 Regression (528개의 경우 : 담합 제외 X)

# In[45]:


df_temp6.shape


# In[46]:


np.mean(df_temp6['tenderRatio'])


# ### ★ 이상치 제거 (4개) & "구매" 제거 (10개)

# In[47]:


original = df_temp6.loc[((df_temp6['Collusion']==0)) & ((df_temp6['dmaintain']==0))]
new = df_temp6.loc[((df_temp6['Collusion']==0) & (df_temp6['dmaintain']==1))]
collusion = df_temp6.loc[df_temp6['Collusion'] == 1]

original.shape, new.shape, collusion.shape


# In[48]:


np.mean(collusion['tenderRatio'])


# In[49]:


main = df_temp6.loc[((df_temp6['dmaintain']==1))]
main.shape
import matplotlib.pyplot as plt
plt.plot('연도', 'tenderRatio', 'o', data=main)


# In[50]:


new.head()


# In[51]:


import matplotlib.pyplot as plt

#plt.plot('연도', 'tenderRatio', 'o' ,data=original,)
plt.plot('연도', 'tenderRatio', 'or' ,data=new, label = "비담합")
plt.plot('연도', 'tenderRatio', 'og', data=collusion, label = "담합")
plt.xlabel("연도")
plt.ylabel("낙찰률")
plt.legend(loc = 'upper right', bbox_to_anchor=(1.25, 1))


# In[52]:


import matplotlib.pyplot as plt

plt.plot('tenderRatio', 'LnCash', 'o' ,data=original,)
plt.plot('tenderRatio', 'LnCash', 'or' ,data=new,)
plt.plot('tenderRatio', 'LnCash', 'og', data=collusion)


# In[53]:


df_temp6.loc[df_temp6['tenderRatio'] == 100].shape


# In[ ]:





# In[169]:


df_temp6.loc[((df_temp6['tenderRatio'] < 65) & (df_temp6['LnCash'] < 26)) | ((df_temp6['tenderRatio'] > 95) & (df_temp6['LnCash'] > 26))]


# In[170]:


# df_temp6.loc[(df_temp6['공고명'].str.contains("구매")), ['낙찰율(예대)','기초금액','공고명', 'dnumfirm2']].to_csv("./V Final_0920/a01.csv", index=False, encoding='EUC-KR')


# In[171]:


df_temp6.loc[~(df_temp6['공고명'].str.contains("구매")) & ((df_temp6['tenderRatio'] < 65) & ((df_temp6['LnCash'] < 26)) | ((df_temp6['tenderRatio'] > 95) & (df_temp6['LnCash'] > 26))), ['Collusion','낙찰율(예대)','기초금액','공고명','dnumfirm2']]


# In[54]:


df_temp6 = df_temp6.loc[~((df_temp6['tenderRatio'] < 65) & (df_temp6['LnCash'] < 26)) & 
             ~((df_temp6['tenderRatio'] > 95) & (df_temp6['LnCash'] > 26)) &
             ~(df_temp6['공고명'].str.contains("구매"))]

df_temp6.shape


# In[144]:


df_temp4.loc[df_temp4['공고명'].str.contains("실시설계")].shape


# In[45]:


df_raw.loc[df_raw['공고명'].str.contains("실시설계")].isna().sum() # 2개 실시설계 용역 낙찰률 X / 1개 실시설계 용역 송산그린시티 낙찰율 0.59 이상치 / 1개 실시설계 용역 


# In[46]:


np.mean(df_temp6['tenderRatio'])


# In[47]:


np.mean(df_temp6.loc[df_temp6['Collusion'] == 0, 'tenderRatio'])


# In[250]:


# """ 점검정비만 돌려봐 """
# df_temp6 = df_temp6.loc[(df_temp6['dmaintain']==1)]
# df_temp6.shape # 56개 (기존 58개에서 2개 제외)


# In[49]:


df_temp6.loc[(df_temp6['공고명'].str.contains("설계"))  & ~ (df_temp6['공고명'].str.contains("실시설계")), ['Collusion','dmaintain','tenderRatio','LnCash','공고명']]


# In[54]:


df_temp6.loc[df_temp6['공고명'].str.contains("설계")].append(df_temp6.loc[df_temp6['공고명'].str.contains("매설계")]).shape


# In[59]:


np.mean(df_temp6.loc[df_temp6['dmaintain'] == 0])


# In[56]:


np.mean(df_temp6.loc[(df_temp6['공고명'].str.contains("설계") & (~df_temp6['공고명'].str.contains("매설계"))),'tenderRatio'])


# In[55]:


df_temp6.loc[(df_temp6['공고명'].str.contains("설계") & (~df_temp6['공고명'].str.contains("매설계"))),'tenderRatio'].shape


# In[254]:


# """ "설계" 총 165개 중 "매설계" 1개 뺀 나머지 164개 제외 : 513 - 164 = 349개 """

# df_temp6 = df_temp6.loc[~(df_temp6['공고명'].str.contains("설계"))].append(df_temp6.loc[df_temp6['공고명'].str.contains("매설계")])
# df_temp6.shape


# In[228]:


""" 전체 - 실시설계 총 349개 평균 """
np.mean(df_temp6['tenderRatio'])


# In[ ]:


""" 156 감정인 데이터 중 실시설계 용역 1개 포함 (관망정비 현장조사) """

outlier = df_156.loc[~(df_156['공고명'].isin(df_temp6.loc[(df_temp6['공고명'].isin(df_156['공고명'])),'공고명']))]
outlier


# In[ ]:


df_temp6.loc[df_temp6['공고명'].isin(df_156['공고명'])].append(outlier).shape


# In[ ]:


df_temp6.loc[df_temp6['공고명'].str.contains("실시설계")]


# In[ ]:


df_temp6.loc[~(df_temp6['공고명'].isin(df_156['공고명'])) & (~df_temp6['dmaintain'] == 1)].shape


# In[ ]:


df_temp6.loc[(~(df_temp6['공고명'].isin(df_156['공고명']))) & (~(df_temp6['dmaintain']==1))].shape # .to_excel('./V Final_1001/139.xlsx', index = False)


# In[ ]:


np.mean(df_temp6.loc[(~(df_temp6['공고명'].isin(df_156['공고명']))) & (~(df_temp6['dmaintain']==1)),'tenderRatio'])


# In[ ]:


np.mean(df_temp6.loc[~(df_temp6['공고명'].isin(df_156['공고명'])) & (df_temp6['dmaintain'] == 1),'tenderRatio'])


# In[ ]:


df_temp6.loc[~(df_temp6['공고명'].isin(df_156['공고명']))].shape


# In[ ]:


np.mean(df_temp6.loc[~(df_temp6['공고명'].isin(df_156['공고명'])), 'tenderRatio'])


# In[140]:


df_temp6.loc[(df_temp6['Collusion'] == 0) & (df_temp6['dmaintain'] == 1)].shape


# In[141]:


np.mean(df_temp6['tenderRatio']), df_temp6.shape # 526개 평균


# In[230]:


df_temp6.loc[df_temp6['dnumfirm2'] != 2]


# In[260]:


""" dnumfirm == 2 인 55개 경우만 돌려봐 """
df_temp6 = df_temp6.loc[df_temp6['dnumfirm2'] == 2]


# In[259]:


# df_56 = df_temp6
df_56.shape


# In[266]:


df_temp6 = df_56
df_temp6.shape


# In[268]:


df_temp6.loc[df_temp6['dnumfirm2'] != 2].shape


# In[274]:


df_temp6['tenderRatio'].sort_values()


# In[56]:


results1 = smf.ols('tenderRatio ~ Collusion + LnCash + dmaintain + dnumfirm2', data=df_temp6).fit()
# results2 = smf.ols('tenderRatio ~ Collusion + LnCash + dmaintain + dnumfirm2 + d2011 + d2012 + d2013 + d2014 + d2015 + d2016 + d2017 + d2018 + d2019 + d2020', data=df_temp6).fit()
# results3 = smf.ols('tenderRatio ~ Collusion + LnCash + dmaintain + dnumfirm2 + d2011 + d2012 + d2013 + d2014 + d2015 + d2016 + d2017 + d2018 + d2019 + d2020 + d33 + d66', data=df_temp6).fit()
results2 = smf.ols('tenderRatio ~ Collusion + LnCash + dmaintain + dnumfirm2 + d2011 + d2012 + d2013 + d2014 + d2016 + d2017 + d2018 + d2019 + d2020', data=df_temp6).fit()
results3 = smf.ols('tenderRatio ~ Collusion + LnCash + dmaintain + dnumfirm2 + d2011 + d2012 + d2013 + d2014 + d2016 + d2017 + d2018 + d2019 + d2020 + d33 + d66', data=df_temp6).fit()

results4 = smf.ols('tenderRatio ~ Collusion + LnCash + dmaintain + dnumfirm2 + db2014 + db2017_b', data=df_temp6).fit()
results5 = smf.ols('tenderRatio ~ Collusion + LnCash + dmaintain + dnumfirm2 + d33 + d66 + db2014 + db2017_b', data=df_temp6).fit()


stargazer_tab = Stargazer([results1, results2, results3, results4, results5])
open('./V Final_1011/regression_132_dmaintain_2014.html', 'w').write(stargazer_tab.render_html())  # for latex


# In[270]:


# results1 = smf.ols('tenderRatio ~ Collusion + LnCash ', data=df_temp6).fit()

# results2 = smf.ols('tenderRatio ~ Collusion + LnCash  + d2013', data=df_temp6).fit()
# results3 = smf.ols('tenderRatio ~ Collusion + LnCash  + d2015', data=df_temp6).fit()
# results4 = smf.ols('tenderRatio ~ Collusion + LnCash  + d2017', data=df_temp6).fit()

# results5 = smf.ols('tenderRatio ~ Collusion + LnCash  + d33 + d66', data=df_temp6).fit()

# results6 = smf.ols('tenderRatio ~ Collusion + LnCash  + d2013 + d33 + d66', data=df_temp6).fit()
# results7 = smf.ols('tenderRatio ~ Collusion + LnCash  + d2015 + d33 + d66', data=df_temp6).fit()
# results8 = smf.ols('tenderRatio ~ Collusion + LnCash  + d2017 + d33 + d66', data=df_temp6).fit()

# stargazer_tab = Stargazer([results1, results2, results3, results4, results5, results6, results7, results8])
# open('./V Final_1006/regression_56_업체수미포함.html', 'w').write(stargazer_tab.render_html())  # for latex


# In[57]:


df_xmatrix = df_temp6.loc[(df_temp6['Collusion'] == 1) & (df_temp6['tenderRatio'] < 90)]
#df_xmatrix = df_xmatrix.drop(['Collusion'], axis = 1)
df_xmatrix.shape


# In[58]:


fitted = pd.concat([df_xmatrix['tenderRatio'],  results1.predict(df_xmatrix), results2.predict(df_xmatrix), results3.predict(df_xmatrix), results4.predict(df_xmatrix), 
                    results5.predict(df_xmatrix)], axis = 1)

fitted.columns = ['tenderRatio', 'fitted1', 'fitted2', 'fitted3', 'fitted4', 'fitted5']

fitted_final = fitted.copy()
fitted_final['diff1'] = fitted['tenderRatio'] - fitted['fitted1'] + results1.params['Collusion']
fitted_final['diff2'] = fitted['tenderRatio'] - fitted['fitted2'] + results2.params['Collusion']
fitted_final['diff3'] = fitted['tenderRatio'] - fitted['fitted3'] + results3.params['Collusion']
fitted_final['diff4'] = fitted['tenderRatio'] - fitted['fitted4'] + results4.params['Collusion']
fitted_final['diff5'] = fitted['tenderRatio'] - fitted['fitted5'] + results5.params['Collusion']

fitted_final.to_csv('./V Final_1011/fitted_132_dmaintain_2014.csv', index = False)


# In[277]:


# fitted = pd.concat([df_xmatrix['tenderRatio'],  results1.predict(df_xmatrix), results2.predict(df_xmatrix), results3.predict(df_xmatrix), results4.predict(df_xmatrix),
#                     results5.predict(df_xmatrix), results6.predict(df_xmatrix), results7.predict(df_xmatrix), results8.predict(df_xmatrix)], axis = 1)

# fitted.columns = ['tenderRatio', 'fitted1', 'fitted2', 'fitted3', 'fitted4', 'fitted5', 'fitted6', 'fitted7', 'fitted8']

# fitted_final = fitted.copy()
# fitted_final['diff1'] = fitted['tenderRatio'] - fitted['fitted1'] + results1.params['Collusion']
# fitted_final['diff2'] = fitted['tenderRatio'] - fitted['fitted2'] + results2.params['Collusion']
# fitted_final['diff3'] = fitted['tenderRatio'] - fitted['fitted3'] + results3.params['Collusion']
# fitted_final['diff4'] = fitted['tenderRatio'] - fitted['fitted4'] + results4.params['Collusion']
# fitted_final['diff5'] = fitted['tenderRatio'] - fitted['fitted5'] + results5.params['Collusion']
# fitted_final['diff6'] = fitted['tenderRatio'] - fitted['fitted6'] + results6.params['Collusion']
# fitted_final['diff7'] = fitted['tenderRatio'] - fitted['fitted7'] + results7.params['Collusion']
# fitted_final['diff8'] = fitted['tenderRatio'] - fitted['fitted8'] + results8.params['Collusion']

# fitted_final.to_csv('./V Final_1006/fitted_56_업체수미포함.csv', index = False)


# ### 7. Exclude collusion

# In[59]:


df_temp7 = df_temp6.loc[df_temp6['Collusion'] == 0]
np.mean(df_temp7['tenderRatio'])
df_temp7.shape


# In[60]:


df_temp7.shape


# In[61]:


# df_temp = df_temp6.loc[df_temp6['Collusion'] == 0]

results1 = smf.ols('tenderRatio ~ LnCash + dmaintain + dnumfirm2', data=df_temp7).fit()
# results2 = smf.ols('tenderRatio ~ LnCash + dmaintain + dnumfirm2 + d2011 + d2012 + d2013 + d2014 + d2015 + d2016 + d2017 + d2018 + d2019 + d2020', data=df_temp7).fit()
# results3 = smf.ols('tenderRatio ~ LnCash + dmaintain + dnumfirm2 + d2011 + d2012 + d2013 + d2014 + d2015 + d2016 + d2017 + d2018 + d2019 + d2020 + d33 + d66', data=df_temp7).fit()
results2 = smf.ols('tenderRatio ~ LnCash + dmaintain + dnumfirm2 + d2011 + d2012 + d2013 + d2014 + d2016 + d2017 + d2018 + d2019 + d2020', data=df_temp7).fit()
results3 = smf.ols('tenderRatio ~ LnCash + dmaintain + dnumfirm2 + d2011 + d2012 + d2013 + d2014 + d2016 + d2017 + d2018 + d2019 + d2020 + d33 + d66', data=df_temp7).fit()
results4 = smf.ols('tenderRatio ~ LnCash + dmaintain + dnumfirm2 + db2014 + db2017_b', data=df_temp7).fit()
results5 = smf.ols('tenderRatio ~ LnCash + dmaintain + dnumfirm2 + d33 + d66 + db2014 + db2017_b', data=df_temp7).fit()


stargazer_tab = Stargazer([results1, results2, results3, results4, results5])
open('./V Final_1011/regression_104_dmaintain_2014.html', 'w').write(stargazer_tab.render_html())  # for latex


# In[280]:


# results1 = smf.ols('tenderRatio ~ LnCash ', data=df_temp7).fit()

# results2 = smf.ols('tenderRatio ~ LnCash  + d2013', data=df_temp7).fit()
# results3 = smf.ols('tenderRatio ~ LnCash  + d2015', data=df_temp7).fit()
# results4 = smf.ols('tenderRatio ~ LnCash  + d2017', data=df_temp7).fit()

# results5 = smf.ols('tenderRatio ~ LnCash  + d33 + d66', data=df_temp7).fit()

# results6 = smf.ols('tenderRatio ~ LnCash  + d2013 + d33 + d66', data=df_temp7).fit()
# results7 = smf.ols('tenderRatio ~ LnCash  + d2015 + d33 + d66', data=df_temp7).fit()
# results8 = smf.ols('tenderRatio ~ LnCash  + d2017 + d33 + d66', data=df_temp7).fit()

# stargazer_tab = Stargazer([results1, results2, results3, results4, results5, results6, results7, results8])
# open('./V Final_1006/regression_28_업체수미포함.html', 'w').write(stargazer_tab.render_html())  # for latex


# In[62]:


fitted = pd.concat([df_xmatrix['tenderRatio'],  results1.predict(df_xmatrix), results2.predict(df_xmatrix), results3.predict(df_xmatrix), results4.predict(df_xmatrix),
                    results5.predict(df_xmatrix)], axis = 1)

fitted.columns = ['tenderRatio', 'fitted1', 'fitted2', 'fitted3', 'fitted4', 'fitted5']

fitted_final = fitted.copy()
fitted_final['diff1'] = fitted['tenderRatio'] - fitted['fitted1']
fitted_final['diff2'] = fitted['tenderRatio'] - fitted['fitted2']
fitted_final['diff3'] = fitted['tenderRatio'] - fitted['fitted3'] 
fitted_final['diff4'] = fitted['tenderRatio'] - fitted['fitted4'] 
fitted_final['diff5'] = fitted['tenderRatio'] - fitted['fitted5']


fitted_final.to_csv('./V Final_1011/fitted_104_dmaintain_2014.csv', index = False)


# In[193]:


# fitted = pd.concat([df_xmatrix['tenderRatio'],  results1.predict(df_xmatrix), results2.predict(df_xmatrix), results3.predict(df_xmatrix), results4.predict(df_xmatrix),
#                     results5.predict(df_xmatrix), results6.predict(df_xmatrix), results7.predict(df_xmatrix), results8.predict(df_xmatrix)], axis = 1)

# fitted.columns = ['tenderRatio', 'fitted1', 'fitted2', 'fitted3', 'fitted4', 'fitted5', 'fitted6', 'fitted7', 'fitted8']

# fitted_final = fitted.copy()
# fitted_final['diff1'] = fitted['tenderRatio'] - fitted['fitted1']
# fitted_final['diff2'] = fitted['tenderRatio'] - fitted['fitted2']
# fitted_final['diff3'] = fitted['tenderRatio'] - fitted['fitted3'] 
# fitted_final['diff4'] = fitted['tenderRatio'] - fitted['fitted4'] 
# fitted_final['diff5'] = fitted['tenderRatio'] - fitted['fitted5']
# fitted_final['diff6'] = fitted['tenderRatio'] - fitted['fitted6']
# fitted_final['diff7'] = fitted['tenderRatio'] - fitted['fitted7']
# fitted_final['diff8'] = fitted['tenderRatio'] - fitted['fitted8']

# fitted_final.to_csv('./V Final_1006/fitted_28_업체수미포함_check.csv', index = False)


# In[498]:


fitted = pd.concat([df_xmatrix['tenderRatio'], results1.predict(df_xmatrix), results2.predict(df_xmatrix), results3.predict(df_xmatrix), 
                    results5.predict(df_xmatrix), results6.predict(df_xmatrix)], axis = 1)

fitted.columns = ['tenderRatio','fitted1', 'fitted2', 'fitted3', 'fitted5', 'fitted6']

fitted_final = fitted.copy()
fitted_final['diff1'] = fitted['tenderRatio'] - fitted['fitted1'] 
fitted_final['diff2'] = fitted['tenderRatio'] - fitted['fitted2']
fitted_final['diff3'] = fitted['tenderRatio'] - fitted['fitted3']
fitted_final['diff5'] = fitted['tenderRatio'] - fitted['fitted5']
fitted_final['diff6'] = fitted['tenderRatio'] - fitted['fitted6']

fitted_final.to_csv('./V Final_1006/fitted_485_14년.csv', index = False)


# # New LS 220916 FINAL : 감정인데이터 기준으로 (2) 사용&제외

# ### 0916 1230 지하수 2개 관측치 제거
# ### 0916 1335 점검정비 55개? 포함

# In[48]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer
from scipy.stats import f

pd.set_option('Display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


# In[49]:


# dfReg = pd.read_excel("./교수님 전달 파일/0914 datafile excluded version.xlsx", sheet_name=None, header = 4)
dfReg = pd.read_excel("./교수님 전달 파일/0915 datafile excluded version dmaintain.xlsx", sheet_name=None, header = 4)


# In[50]:


dfReg.keys()


# In[51]:


raw = pd.concat(dfReg, ignore_index=True)


# In[52]:


Klist = pd.Series(['2011-0107' ,'2011-0892','2013-0036' ,'2013-0559' ,'2013-2386' ,'2016-4565' ,'2017-0119' ,'2018-0139' ,'B5201800526' ,'B5201900988' ,'B5201901079' ,'B5201901922' ,'B5202002509'])


# In[65]:


df_raw = raw[['공고명','낙찰율(예대)','연도','기초금액','1순위투찰금액','업체수','담합','점검정비','공고번호','감정']]

df_raw.shape


# In[66]:


df_raw.loc[df_raw['낙찰율(예대)'] == 1]


# In[67]:


df_raw['공고번호'].isin(Klist)


# In[68]:


df_raw.loc[df_raw['점검정비'] == 1].shape


# In[69]:


# df_raw.loc[df_raw['공고명'].str.contains("지하수 기초조사") | 
#            df_raw['공고명'].str.contains("시설 전수조사") | 
#            df_raw['공고명'].str.contains("관측망") | 
#            df_raw['공고명'].str.contains("정밀안전") | 
#            df_raw['공고명'].str.contains("GIS") |
#            df_raw['공고명'].str.contains("내진") |
#            df_raw['공고명'].str.contains("사후환경") |
#            df_raw['공고명'].str.contains("기술진단") | 
#            df_raw['공고명'].str.contains("시설안정화") |
#            df_raw['공고명'].str.contains("건설사업관리") |
#            df_raw['공고명'].str.contains("댐비상대처") |
#            df_raw['공고명'].str.contains("물 진단고도화") |
#            df_raw['공고명'].str.contains("환경영향평가") |
#            df_raw['공고명'].str.contains("관망정비 현장조사") |
#            df_raw['공고명'].str.contains("관로이설공사") |
#            df_raw['공고명'].str.contains("해양환경영향조사") |
#            df_raw['공고명'].str.contains("송산그린시티") |
#            df_raw['공고명'].str.contains("시공감리")]


# In[70]:


df_raw.loc[df_raw['공고번호'].isin(Klist)].shape


# In[71]:


from matplotlib import rc, font_manager, rcParams
font_name = font_manager.FontProperties(fname = 'c:/Windows/Fonts/malgun.ttf').get_name()
rc('font', family = font_name)
rcParams['axes.unicode_minus'] = False


# In[72]:


# df_raw_out = df_raw.loc[df_raw['공고명'].str.contains("지하수 기초조사") | 
#            df_raw['공고명'].str.contains("시설 전수조사") | 
#            df_raw['공고명'].str.contains("관측망") | 
#            df_raw['공고명'].str.contains("정밀안전") | 
#            df_raw['공고명'].str.contains("GIS") |
#            df_raw['공고명'].str.contains("내진") |
#            df_raw['공고명'].str.contains("사후환경") |
#            df_raw['공고명'].str.contains("기술진단") | 
#            df_raw['공고명'].str.contains("시설안정화") |
#            df_raw['공고명'].str.contains("건설사업관리") |
#            df_raw['공고명'].str.contains("댐비상대처") |
#            df_raw['공고명'].str.contains("물 진단고도화") |
#            df_raw['공고명'].str.contains("환경영향평가") |
#            df_raw['공고명'].str.contains("관망정비 현장조사") |
#            df_raw['공고명'].str.contains("관로이설공사") |
#            df_raw['공고명'].str.contains("해양환경영향조사") |
#            df_raw['공고명'].str.contains("송산그린시티") |
#            df_raw['공고명'].str.contains("시공감리")]

# df_raw_out.to_csv("df_out.csv", index = False)


# In[73]:


np.mean(df_raw.loc[(df_raw['공고명'].str.contains("설계")), '낙찰율(예대)'])


# In[74]:


df_raw.loc[(df_raw['공고명'].str.contains("설계")), '낙찰율(예대)'].dropna()


# In[ ]:





# In[75]:


df_raw.loc[((df_raw['공고명'].str.contains("송산그린시티")) & ((df_raw['공고명'].str.contains("설계"))))]


# In[77]:


"""@@@@@@@@@@@@@@@@@@@@@@@"""

df_raw2 = df_raw.loc[df_raw['공고명'].str.contains("지하수 기초조사") | 
           df_raw['공고명'].str.contains("시설 전수조사") | 
           df_raw['공고명'].str.contains("관측망") | 
           df_raw['공고명'].str.contains("정밀안전") | 
           df_raw['공고명'].str.contains("GIS") |
           df_raw['공고명'].str.contains("내진") |
           df_raw['공고명'].str.contains("사후환경") |
           df_raw['공고명'].str.contains("기술진단") | 
           df_raw['공고명'].str.contains("시설안정화") |
           df_raw['공고명'].str.contains("건설사업관리") |
           df_raw['공고명'].str.contains("비상대처") |
           df_raw['공고명'].str.contains("물 진단고도화") |
           df_raw['공고명'].str.contains("환경영향평가") |
           df_raw['공고명'].str.contains("관망정비 현장조사") |
           df_raw['공고명'].str.contains("관로이설공사") |
           df_raw['공고명'].str.contains("해양환경영향조사") |
           ((df_raw['공고명'].str.contains("송산그린시티")) & (~(df_raw['공고명'].str.contains("설계")))) |
           df_raw['공고명'].str.contains("시공감리") |
           df_raw['점검정비'] == 1 | 
           (~df_raw['감정'].isna())]

df_raw2.shape


# In[80]:


df_raw.loc[~df_raw['감정'].isna()].shape


# In[684]:


df_raw2.isna().sum()


# In[685]:


df_raw.loc[df_raw['공고번호'].isin(Klist)]


# In[686]:


df_raw2.loc[df_raw2['공고번호'].isin(Klist)]


# In[687]:


df_raw3 = df_raw2.append(df_raw.loc[df_raw['공고번호'].isin(Klist)]).drop_duplicates()


# In[688]:


df_raw3.shape


# In[689]:


df_raw3.loc[df_raw3['공고명'].str.contains("관망정비")]


# In[690]:


df_raw3['담합'].sum() ,df_raw3['점검정비'].sum() # 30 & 44 ok


# In[691]:


df_raw3.isna().sum()


# In[692]:


df_raw.loc[(df_raw['점검정비'] == 1) & (df_raw['담합'] == 0) & (df_raw['연도'] != 2018)] 


# In[693]:


df_raw3.shape


# In[694]:


""" 점검정비 6개 포함시킨거 (X)"""
# df_addtoraw = df_raw.loc[(df_raw['점검정비'] == 1) & (df_raw['담합'] == 0) & (df_raw['연도'] != 2018)] 
# df_raw4 = df_raw3.append(df_addtoraw)
# df_raw4.loc[~df_raw4['공고명'].isin(df_raw3['공고명'])]
# df_raw4.drop_duplicates().shape


# In[695]:


df_raw3['점검정비'].sum()


# ### 0. 유찰된 데이터 제거 및 확인 & 0919 이상치 제거

# In[696]:


df_temp1 = df_raw3.loc[(df_raw3['1순위투찰금액'] != 0) & (df_raw3['낙찰율(예대)'].isna() != True), ['낙찰율(예대)','연도', '기초금액', '업체수', '담합', '점검정비','공고번호','공고명']]
df_temp1.head(3)
df_temp1.shape


# In[697]:


(df_temp1['담합'] == df_temp1['점검정비']).sum()


# In[698]:


df_temp1.loc[((df_temp1['담합'] == 1) & (df_temp1['낙찰율(예대)'] > 0.9))]


# In[699]:


""" 담합 데이터 2개 제거 """

df_temp1 = df_temp1.loc[~((df_temp1['담합'] == 1) & (df_temp1['낙찰율(예대)'] > 0.9))]
df_temp1.shape


# In[700]:


df_temp1.loc[df_temp1['기초금액'] < 1000000000] # 10억보다 작은 금액 존재 X


# In[701]:


df_temp1['연도'].unique()


# ### 1. 연도별 더미 생성

# In[702]:


df_temp2 = df_temp1.copy()
df_temp2[['d2011','d2012','d2013','d2014','d2015','d2016','d2017','d2018','d2019','d2020']] = 0
df_temp2.loc[df_temp2['연도'] == 2011, 'd2011'] = 1 
df_temp2.loc[df_temp2['연도'] == 2012, 'd2012'] = 1 
df_temp2.loc[df_temp2['연도'] == 2013, 'd2013'] = 1 
df_temp2.loc[df_temp2['연도'] == 2014, 'd2014'] = 1 
df_temp2.loc[df_temp2['연도'] == 2015, 'd2015'] = 1 
df_temp2.loc[df_temp2['연도'] == 2016, 'd2016'] = 1 
df_temp2.loc[df_temp2['연도'] == 2017, 'd2017'] = 1 
df_temp2.loc[df_temp2['연도'] == 2018, 'd2018'] = 1 
df_temp2.loc[df_temp2['연도'] == 2019, 'd2019'] = 1 
df_temp2.loc[df_temp2['연도'] == 2020, 'd2020'] = 1 


# In[703]:


df_temp2.head(3)


# ### 2. 규모 더미 생성 (33% & 66% Quantile)

# In[704]:


df_temp2['기초금액'].sort_values().quantile([.33, .66])


# In[705]:


df_temp3 = df_temp2.copy()
df_temp3['d33'] = 0
df_temp3['d66'] = 0

df_temp3.loc[df_temp3['기초금액'] < df_temp2['기초금액'].sort_values().quantile(.33), 'd33'] = 1
df_temp3.loc[(df_temp3['기초금액'] > df_temp2['기초금액'].sort_values().quantile(.33)) & (df_temp3['기초금액'] < df_temp2['기초금액'].sort_values().quantile(.66)), 'd66'] = 1
df_temp3


# ### 3. 업체수 규제 (완전 제거 & 20으로 고정)

# In[706]:


df_temp4 = df_temp3.copy()


# In[707]:


# 3-2. dnumfirm2 : 업체수 20 이상은 NAN 으로 값 줘서 추후 삭제
df_temp4['dnumfirm2'] = np.nan
df_temp4.loc[df_temp4['업체수'] <= 20, 'dnumfirm2'] = df_temp4.loc[df_temp4['업체수'] <= 20, '업체수']
df_temp4 = df_temp4.dropna()
df_temp4.shape


# ### 4. Linear Regression 준비 위한 dataframe 재정비

# In[708]:


df_temp5 = df_temp4[['낙찰율(예대)','담합', '기초금액','dnumfirm2','d2011','d2012','d2013','d2014','d2015','d2016','d2017','d2018','d2019','d2020','d33', 'd66', '점검정비','공고번호','공고명']]
df_temp5 = df_temp5.copy()
df_temp5['낙찰율'] = df_temp5['낙찰율(예대)']*100
df_temp5['ln기초금액'] = np.log(df_temp5['기초금액'])
df_temp5 = df_temp5.rename(columns = {"낙찰율" : "tenderRatio", "담합" : "Collusion", "ln기초금액" : "LnCash", "점검정비" : "dmaintain"})
df_temp5.head()


# In[709]:


df_temp5['dnumfirm2'].isna().sum() #60
df_temp5.shape # 640


# In[710]:


df_temp5.isna().sum().sum()


# In[711]:


df_temp5['Collusion'].sum()


# ### 4-1. 연도별 더미 추가 생성

# In[712]:


df_temp5.head(3)


# In[713]:


df_temp6 = df_temp5.copy() 

df_temp6[['db2013','db2014','db2017_a', 'db2017_b']] = 0

df_temp6.loc[df_temp6['d2011'] == 1, 'db2013'] =1
df_temp6.loc[df_temp6['d2012'] == 1, 'db2013'] =1
df_temp6.loc[df_temp6['d2013'] == 1, 'db2013'] =1

df_temp6.loc[df_temp6['d2011'] == 1, 'db2014'] =1
df_temp6.loc[df_temp6['d2012'] == 1, 'db2014'] =1
df_temp6.loc[df_temp6['d2013'] == 1, 'db2014'] =1
df_temp6.loc[df_temp6['d2014'] == 1, 'db2014'] =1

df_temp6.loc[df_temp6['d2014'] == 1, 'db2017_a'] =1
df_temp6.loc[df_temp6['d2015'] == 1, 'db2017_a'] =1
df_temp6.loc[df_temp6['d2016'] == 1, 'db2017_a'] =1
df_temp6.loc[df_temp6['d2017'] == 1, 'db2017_a'] =1

df_temp6.loc[df_temp6['d2015'] == 1, 'db2017_b'] =1
df_temp6.loc[df_temp6['d2016'] == 1, 'db2017_b'] =1
df_temp6.loc[df_temp6['d2017'] == 1, 'db2017_b'] =1


# In[714]:


df_temp6.isna().sum().sum()


# ### 5. 연도 이전 이후 더미 Regression (528개의 경우 : 담합 제외 X)

# In[715]:


np.mean(df_temp6['tenderRatio'])


# In[716]:


original = df_temp6.loc[((df_temp6['Collusion']==0) & (df_temp6['dmaintain']==0))]
new = df_temp6.loc[((df_temp6['Collusion']==0) & (df_temp6['dmaintain']==1))]
collusion = df_temp6.loc[df_temp6['Collusion'] == 1]


# In[717]:


original.shape, new.shape, collusion.shape


# In[718]:


import matplotlib.pyplot as plt

plt.plot('tenderRatio', 'LnCash', 'o' ,data=original,)
plt.plot('tenderRatio', 'LnCash', 'or' ,data=new,)
plt.plot('tenderRatio', 'LnCash', 'og', data=collusion)


# In[719]:



""" 이상치 데이터 2개 제거 """


df_temp6 = df_temp6.loc[~(((df_temp6['LnCash'] > 26) & (df_temp6['tenderRatio'] > 95)) | ((df_temp6['LnCash'] < 23) & (df_temp6['tenderRatio'] < 65)))]
df_temp6.shape


# In[720]:


df_156 = df_temp6.loc[df_temp6['Collusion']==0]
df_156.shape


# In[721]:


np.mean(df_156['tenderRatio'])


# In[722]:


df_temp6.loc[(df_temp6['Collusion'] == 0) & (df_temp6['dmaintain'] == 1)].shape


# In[723]:


results1 = smf.ols('tenderRatio ~ Collusion + LnCash + dnumfirm2', data=df_temp6).fit()
results2 = smf.ols('tenderRatio ~ Collusion + LnCash + dnumfirm2 + d2011 + d2012 + d2013 + d2014 + d2015 + d2016 + d2017 + d2018 + d2019 + d2020', data=df_temp6).fit()
results3 = smf.ols('tenderRatio ~ Collusion + LnCash + dnumfirm2 + d2011 + d2012 + d2013 + d2014 + d2015 + d2016 + d2017 + d2018 + d2019 + d2020 + d33 + d66', data=df_temp6).fit()

results5 = smf.ols('tenderRatio ~ Collusion + LnCash + dnumfirm2 + db2014 + db2017_b', data=df_temp6).fit()
results6 = smf.ols('tenderRatio ~ Collusion + LnCash + dnumfirm2 + d33 + d66 + db2014 + db2017_b', data=df_temp6).fit()


stargazer_tab = Stargazer([results1, results2, results3, results5, results6])
open('./V Final_1006/regression_212_점검_14년.html', 'w').write(stargazer_tab.render_html())  # for latex


# In[724]:


"""FFFFFFFFFFFFFFIIIIIIIIIIIIIIIIIITTTTTTTTTTTTT"""

df_xmatrix = df_temp6.loc[(df_temp6['Collusion'] == 1) & (df_temp6['tenderRatio'] < 90)]
df_xmatrix.shape


# In[725]:


fitted = pd.concat([df_xmatrix['tenderRatio'],  results1.predict(df_xmatrix), results2.predict(df_xmatrix), results3.predict(df_xmatrix), 
                    results5.predict(df_xmatrix), results6.predict(df_xmatrix)], axis = 1)

fitted.columns = ['tenderRatio', 'fitted1', 'fitted2', 'fitted3', 'fitted5', 'fitted6']

fitted_final = fitted.copy()
fitted_final['diff1'] = fitted['tenderRatio'] - fitted['fitted1'] + results1.params['Collusion']
fitted_final['diff2'] = fitted['tenderRatio'] - fitted['fitted2'] + results2.params['Collusion']
fitted_final['diff3'] = fitted['tenderRatio'] - fitted['fitted3'] + results3.params['Collusion']
fitted_final['diff5'] = fitted['tenderRatio'] - fitted['fitted5'] + results5.params['Collusion']
fitted_final['diff6'] = fitted['tenderRatio'] - fitted['fitted6'] + results6.params['Collusion']


fitted_final.to_csv('./V Final_1006/fitted_212_점검_14년.csv', index = False)


# In[726]:


df_temp7 = df_temp6.loc[df_temp6['Collusion'] == 0]
df_temp7.shape


# In[727]:


df_temp6['dmaintain'].sum()


# ### 7. Exclude collusion

# In[728]:


df_temp7 = df_temp6.loc[df_temp6['Collusion'] == 0]

results1 = smf.ols('tenderRatio ~ LnCash + dnumfirm2', data=df_temp7).fit()
results2 = smf.ols('tenderRatio ~ LnCash + dnumfirm2 + d2011 + d2012 + d2013 + d2014 + d2015 + d2016 + d2017 + d2018 + d2019 + d2020', data=df_temp7).fit()
results3 = smf.ols('tenderRatio ~ LnCash + dnumfirm2 + d2011 + d2012 + d2013 + d2014 + d2015 + d2016 + d2017 + d2018 + d2019 + d2020 + d33 + d66', data=df_temp7).fit()
results5 = smf.ols('tenderRatio ~ LnCash + dnumfirm2 + db2014 + db2017_b', data=df_temp7).fit()
results6 = smf.ols('tenderRatio ~ LnCash + dnumfirm2 + d33 + d66 + db2014 + db2017_b', data=df_temp7).fit()



stargazer_tab = Stargazer([results1, results2, results3, results5, results6])
open('./V Final_1006/regression_184_점검_14년.html', 'w').write(stargazer_tab.render_html())  # for latex


# In[729]:


df_temp6['dmaintain'].sum()


# In[730]:


fitted = pd.concat([df_xmatrix['tenderRatio'], results1.predict(df_xmatrix), results2.predict(df_xmatrix), results3.predict(df_xmatrix), 
                    results5.predict(df_xmatrix), results6.predict(df_xmatrix)], axis = 1)

fitted.columns = ['tenderRatio','fitted1', 'fitted2', 'fitted3', 'fitted5', 'fitted6']

fitted_final = fitted.copy()
fitted_final['diff1'] = fitted['tenderRatio'] - fitted['fitted1'] 
fitted_final['diff2'] = fitted['tenderRatio'] - fitted['fitted2']
fitted_final['diff3'] = fitted['tenderRatio'] - fitted['fitted3']
fitted_final['diff5'] = fitted['tenderRatio'] - fitted['fitted5']
fitted_final['diff6'] = fitted['tenderRatio'] - fitted['fitted6']


fitted_final.to_csv('./V Final_1006/fitted_184_점검_14년.csv', index = False)


# In[ ]:


df_temp6.columns


# ### ★ 공고명 각각 이름 몇 개 존재하는지 확인

# df_raw87 = df_temp6.loc[df_raw['공고명'].str.contains("지하수 기초조사") | 
#            df_temp6['공고명'].str.contains("시설 전수조사") | 
#            df_temp6['공고명'].str.contains("관측망") | 
#            df_temp6['공고명'].str.contains("정밀안전") | 
#            df_temp6['공고명'].str.contains("GIS") |
#            df_temp6['공고명'].str.contains("내진") |
#            df_temp6['공고명'].str.contains("사후환경") |
#            df_temp6['공고명'].str.contains("기술진단") | 
#            df_temp6['공고명'].str.contains("시설안정화") |
#            df_temp6['공고명'].str.contains("건설사업관리") |
#            df_temp6['공고명'].str.contains("댐비상대처") |
#            df_temp6['공고명'].str.contains("물 진단고도화") |
#            df_temp6['공고명'].str.contains("환경영향평가") |
#            df_temp6['공고명'].str.contains("관망정비 현장조사") |
#            df_temp6['공고명'].str.contains("관로이설공사") |
#            df_temp6['공고명'].str.contains("해양환경영향조사") |
#            ((df_temp6['공고명'].str.contains("송산그린시티")) & (~(df_temp6['공고명'].str.contains("설계")))) |
#            df_temp6['공고명'].str.contains("시공감리") |
#            df_temp6['점검정비'] == 1]
# 
# df_raw2.to_csv("df_raw2.csv", index = False, encoding='EUC-KR')

# In[ ]:


df_raw.loc[df_raw['공고명'].str.contains("송산그린시티")].shape


# In[ ]:


df_temp6.loc[df_temp6['공고명'].str.contains("송산그린시티")].shape


# In[ ]:


df_temp6.to_excel("df_temp6.xlsx", index=False, encoding='EUC-KR')


# In[ ]:


# df_raw.loc[df_raw['공고명'].str.contains("지하수 기초조사")| (df_raw['공고명'].str.contains("시설 전수조사")) | (df_raw['공고명'].str.contains("관측망"))].to_csv("키워드1.csv", index=False, encoding='EUC-KR')
# df_raw.loc[df_raw['공고명'].str.contains("기술진단")].to_csv("키워드2.csv", index=False, encoding='EUC-KR')
# df_raw.loc[df_raw['공고명'].str.contains("관측망")].to_csv("키워드3.csv", index=False, encoding='EUC-KR')
# df_raw.loc[df_raw['공고명'].str.contains("정밀안전")].to_csv("키워드4.csv", index=False, encoding='EUC-KR')
# df_raw.loc[df_raw['공고명'].str.contains("GIS")].to_csv("키워드5.csv", index=False, encoding='EUC-KR')
# df_raw.loc[df_raw['공고명'].str.contains("내진")].to_csv("키워드5.csv", index=False, encoding='EUC-KR')
# df_raw.loc[df_raw['공고명'].str.contains("기술진단")].to_csv("키워드7.csv", index=False, encoding='EUC-KR')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


df_temp6.shape


# In[ ]:


asd = df_temp6.rename({'공고번호' : '입찰공고번호'})
asd


# In[ ]:


dfReg = pd.read_excel("./교수님 전달 파일/회귀분석원자료.xlsx")

# temp1 : 연도별 더미 생성
df_temp1 = dfReg[['tenderRatio','techScore','techGap','낙찰여부','largeBusiness','입찰공고번호','계약명']]
df_temp1.head()


# In[ ]:


df_temp1 = df_temp1.loc[df_temp1['낙찰여부'] == '낙찰', ['tenderRatio', '입찰공고번호','계약명']]


# In[ ]:


df_temp6.loc[df_temp6['공고명'].str.contains('건설사업관리')].shape


# In[ ]:


df_temp1.loc[df_temp1['계약명'].str.contains("시공감리")]


# In[ ]:


df_temp1.loc[((df_temp1['계약명'].str.contains("송산그린시티")) & (~(df_temp1['계약명'].str.contains("설계"))))]


# In[ ]:


df_temp1.loc[(df_temp1['계약명'].str.contains("송산그린시티"))]


# In[ ]:


# df_temp1.to_csv("df_temp1.csv", index = False, encoding='EUC-KR')


# In[ ]:


df_temp6.columns


# In[ ]:


df_tempplot  = df_temp6.copy()
df_tempplot['감정인사용'] = 0
df_tempplot.loc[(df_temp6['dmaintain'] == 0 ) & (df_temp6['Collusion'] == 0) & (df_tempplot['공고번호'].isin(df_temp1['입찰공고번호'])), '감정인사용'] = 1
df_tempplot.loc[(df_temp6['dmaintain'] == 0 ) & (df_temp6['Collusion'] == 0) & (~df_tempplot['공고번호'].isin(df_temp1['입찰공고번호'])), '감정인사용'] = 2
df_tempplot.loc[df_tempplot['감정인사용'] == 2].shape


# In[ ]:


df_tempplot.loc[(df_tempplot['감정인사용'] == 0) & (df_tempplot['Collusion'] == 1)]


# In[ ]:


in76 = df_tempplot.loc[((df_tempplot['Collusion']==0) & (df_tempplot['dmaintain']==0) & (df_tempplot['감정인사용']==1))]
in80 = df_tempplot.loc[((df_tempplot['Collusion']==0) & (df_tempplot['dmaintain']==0) & (df_tempplot['감정인사용']==2))]
main = df_tempplot.loc[(df_tempplot['Collusion']==0) & (df_tempplot['dmaintain']==1)]
coll = df_tempplot.loc[df_tempplot['Collusion'] == 1]

in76.shape, in80.shape, main.shape, coll.shape


# In[ ]:


np.mean(in80['tenderRatio'])


# In[ ]:


np.mean(in76['tenderRatio'])


# In[ ]:


df_tempplot.loc[~((df_tempplot['Collusion']==0) & (df_tempplot['dmaintain']==0)), 'tenderRatio'].shape


# In[ ]:


np.mean(df_tempplot['tenderRatio'])


# In[ ]:


np.mean(df_tempplot.loc[((df_tempplot['Collusion']==0) & (df_tempplot['dmaintain']==0)) | ((df_tempplot['Collusion']==0) & (df_tempplot['dmaintain']==1)), 'tenderRatio'])


# In[ ]:


np.mean(df_tempplot.loc[((df_tempplot['Collusion']==0) & (df_tempplot['dmaintain']==0)) | ((df_tempplot['Collusion']==1) & (df_tempplot['dmaintain']==1)), 'tenderRatio'])


# In[ ]:


df_tempplot.loc[((df_tempplot['Collusion']==0) & (df_tempplot['dmaintain']==0)) | ((df_tempplot['Collusion']==1) & (df_tempplot['dmaintain']==1)), 'tenderRatio'].shape


# In[ ]:


df_tempplot.loc[(df_tempplot['Collusion'] == 0) & (df_tempplot['dmaintain']==1)].shape


# In[ ]:


np.mean(df_tempplot.loc[((df_tempplot['Collusion']==0) & (df_tempplot['dmaintain']==0)), 'tenderRatio'])


# In[ ]:


""" tech score 의 경우 156개 중 67개 만 존재하고 나머지 존재하지 X 도출 불가 """
df_tempplot.columns


# In[ ]:


import matplotlib.pyplot as plt

plt.plot('tenderRatio', 'LnCash', 'o' ,data=in76, label = 'In 76')
plt.plot('tenderRatio', 'LnCash', 'o' ,data=in80, label = 'In 80')
plt.plot('tenderRatio', 'LnCash', 'or' ,data=main, label = '점검정비(담합X)')
plt.plot('tenderRatio', 'LnCash', 'og', data=coll, label = '담합')
plt.legend(loc = 'upper right', bbox_to_anchor=(1.374, 1))
plt.xlabel('투찰율')
plt.ylabel("로그기초금액")


# In[ ]:





# In[ ]:


df_156 = df_temp6.loc[(df_temp6['dmaintain'] == 0 ) & (df_temp6['Collusion'] == 0)]
df_156.shape # 156
df_temp1.shape # 76


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


df_80 = df_156.loc[~df_156['공고번호'].isin(df_temp1['입찰공고번호'])]
df_80.shape


# In[ ]:


qq= asd.loc[~asd['공고번호'].isin(df_temp1['입찰공고번호'])]


# In[ ]:


np.mean(qq['tenderRatio'])


# In[ ]:


(~df_temp1['입찰공고번호'].isin(asd['공고번호'])).sum()


# In[ ]:


df_temp1.loc[~df_temp1['입찰공고번호'].isin(asd['공고번호'])]


# In[ ]:


asd.loc[asd['공고번호'].str.contains('2013-0559')]


# In[ ]:


df_temp1.loc[df_temp1['입찰공고번호'] == '2013-0559']


# In[ ]:


asd.loc[asd['공고번호'] == '2011-0107']


# In[ ]:


asd

