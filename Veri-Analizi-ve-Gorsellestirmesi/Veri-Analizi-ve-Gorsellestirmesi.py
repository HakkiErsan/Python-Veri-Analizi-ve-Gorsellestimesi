#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Kütüphaneler, import komutu ile yüklenmektedir.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="darkgrid") 


# In[ ]:


#Veri setimizi okuyup bir değişkene atayalım

salary = pd.read_csv(r"C:\Users\hersann\Desktop\employeeearningscy.csv", encoding="latin-1")


# In[ ]:


#İlk 5 satırı incelemek için head() metodunu kullanabiliriz.

salary.head()


# In[ ]:


#Son 5 satırı incelemek için ise tail() metodunu kullanırız.

salary.tail()


# In[ ]:


#İlk olarak, salary DataFrame'inde columns özelliğini görüntüleyerek sütunlara bakalım.

salary.columns


# In[ ]:


#Dizindeki dizeleri küçük harfe dönüştürmek için str.lower () işlevini kullanırız.
#İşimizi kolaylaştırmak için dizindeki dizelerde boşluk yerine "_" işaretini koyarız.

salary.columns= salary.columns.str.lower()

salary.rename(columns={'total earnings': 'total_earnings','quinn/education incentive':'quinn_incentive'}, inplace=True)


# In[ ]:


#Sorun, 'total.earnings' sütununun sayısal olmayan karakter(",") içermesidir.
#Burada dtypes kullanıp total_earnings'in veri türünün bir "nesne" olarak tanındığını görebiliriz.

salary.dtypes


# In[ ]:


#'total_earnings' ve diğer sayısal karakter içeren sütunların içindeki ',' öğesini bulmamız ve kaldırmamız gerekir. 
#str.replace () işlevi bunu yapmamıza izin verir.

salary['regular']= salary['regular'].str.replace(',','')
salary['retro']= salary['retro'].str.replace(',','')
salary['other']= salary['other'].str.replace(',','')
salary['overtime']= salary['overtime'].str.replace(',','')
salary['injured']= salary['injured'].str.replace(',','')
salary['detail']= salary['detail'].str.replace(',','')
salary['quinn_incentive']= salary['quinn_incentive'].str.replace(',','')
salary['total_earnings']= salary['total_earnings'].str.replace(',','')


# In[ ]:


#Veri tipimiz nesne olduğundan hala sıralı değil.
#Veri tipini değişmeyi pd.to_numeric() ile yapabiliriz.
#errors='coerce' geçersiz değerleri null değerine döndürür.

salary['regular'] = pd.to_numeric(salary['regular'], errors='coerce')
salary['retro'] = pd.to_numeric(salary['retro'], errors='coerce')
salary['other'] = pd.to_numeric(salary['other'], errors='coerce')
salary['overtime'] = pd.to_numeric(salary['overtime'], errors='coerce')
salary['injured'] = pd.to_numeric(salary['injured'], errors='coerce')
salary['detail'] = pd.to_numeric(salary['detail'], errors='coerce')
salary['quinn_incentive'] = pd.to_numeric(salary['quinn_incentive'], errors='coerce')
salary['total_earnings'] = pd.to_numeric(salary['total_earnings'], errors='coerce')


# In[ ]:


#Kayıp verileri doldurmak için fillna komutu kullanılır.
#Kayıp verilerin yerine 0 değeri girilir.

salary.fillna({'regular':0}, inplace=True)
salary.fillna({'retro':0}, inplace=True)
salary.fillna({'other':0}, inplace=True)
salary.fillna({'overtime':0}, inplace=True)
salary.fillna({'injured':0}, inplace=True)
salary.fillna({'detail':0}, inplace=True)
salary.fillna({'quinn_incentive':0}, inplace=True)
salary.fillna({'total_earnings':0}, inplace=True)


# In[ ]:


#Kayıp veri kalmadıgı görülür.

salary.isnull().sum().sum()


# In[ ]:


#Tekrar eden değer var mı diye kontrol edilir.

salary.duplicated().sum()


# In[ ]:


#Tekar eden değerleri kaldırılır.

salary.drop_duplicates(keep=False , inplace=True)


# In[ ]:


#Toplam kazanç 0'dan büyük olmalı.

salary=salary[salary.total_earnings>0]


# In[ ]:


#'selected_salary' adlı yeni bir DataFrame oluşturup seçilen sütunlar bu DataFrame'e kaydedilir.

selected_salary = salary[['name','department_name','title', 'total_earnings']].copy()


# In[ ]:


#Verileri pandastaki sort_values() işlevini kullanarak toplam maaşa göre büyükten küçüğe doğru sıralanır:

salary_sort = selected_salary.sort_values('total_earnings', ascending = False)


# In[ ]:


#İndex kısmı dağınık duruyor.
#reset_index() komutunu kullanarak index kısmı düzeltilir.

salary_sort = salary_sort.reset_index(drop = True)


# In[ ]:


#groupby ve mean() işlevlerini kullanarak "salary_average" adını vereceğimiz bölüme göre ortalama kazançları bulabiliriz.

salary_average = salary_sort.groupby('department_name').mean()


# In[ ]:


#Baktığımızda departman isimlerinin satır dizini olarak kullanıldığını görürüz.
#Bundan kurtulmak için reset_index() fonksiyonunu kullanırız.

salary_average = salary_average.reset_index() # reset_index


# In[ ]:


#Burada kullandığımız 'total_earnings' aslında departmanların ortalama kazançlarıdır.
#Karışıklığı önlemek için sütun ismimizi 'average_earnings' olarak değiştiririz.

salary_average.rename(columns = {'total_earnings': 'dept_average'},inplace=True)


# In[ ]:


#İki ana veri setimiz var.Biri maaş sıralamamızın olduğu "salary_sort" diğeri maaş ortalamamızın olduğu "salary_average". 
#Bu iki veri setini bir araya getirerek departmanların ortalamasına kıyasla her bireyin maaşını görebiliriz.

#Her iki veri kümesinde de tutarlı olduğundan "department_name" değişkenine katılırız. 
#Birleştirilmiş verileri "salary_merged" adlı yeni bir veri çerçevesine koyalım.

salary_merged = pd.merge(salary_sort, salary_average, on = 'department_name')


# In[ ]:


#Anormal tespitimiz için IsolationForest import edilir.

from sklearn.ensemble import IsolationForest


# In[ ]:


#Toplam maaştaki anormallikler için modelimizi tanımlarız

model1 = IsolationForest(n_estimators=100,behaviour='new',contamination=float(0.05))


min_max = np.linspace(salary_merged['total_earnings'].min(),
                      salary_merged['total_earnings'].max(), 
                      len(salary_merged)).reshape(-1, 1)

Y = salary_merged['total_earnings'].values.reshape(-1,1)

#Modelimizi eğitiriz
model1.fit(min_max)

#Eğitimli modeli çağırarak puanlar sütununun değerlerini bulabiliriz.
salary_merged['anomaly_score'] = model1.decision_function(Y)

#Benzer şekilde eğitimli modeli çağırarak anomali sütununun değerlerini bulabiliriz.
salary_merged['outlier'] = model1.predict(Y)


# In[ ]:


#Anormal değerleri olanları görebiliriz.

salary_merged[salary_merged.outlier==-1].sort_values(by='total_earnings',ascending=False)


# In[ ]:


#Normal olarak tespit edilen değerleri görebiliriz.

salary_merged[salary_merged.outlier==1].sort_values(by='total_earnings',ascending=False)


# In[ ]:


#Anormal olmayan değerlerimizi yeni oluşturduğumuz DataFrame'e atarız.

salary_merged1 = salary_merged[(salary_merged.outlier==1)].sort_values(by='total_earnings',ascending=False)

salary_merged1 = salary_merged1[['name','department_name','title', 'total_earnings']].copy()


# In[ ]:


#Yeni departman ortalamaları oluşturulur.

salary_average1=(salary_merged1.groupby('department_name')
 .mean()
 .reset_index()
 .rename(columns = {'total_earnings':'dept_average'}))

#Şimdi departman ortalamasını ve bireyin maaşını tekrar birleştiririz.

salary_merged1 = pd.merge(salary_merged1, salary_average1, on = 'department_name')


# In[ ]:


#Toplam maaş ve departman arasındaki anormallik tespiti için ikinci modelimizi oluştururuz.

model2 = IsolationForest(n_estimators=100,behaviour='new',contamination=float(0.05))

X1=salary_merged1['dept_average'].values.reshape(-1,1)
X2=salary_merged1['total_earnings'].values.reshape(-1,1)
X = np.concatenate((X1,X2),axis=1)

#Modelimizi eğitiriz
model2.fit(X)

#Eğitimli modeli çağırarak puanlar sütununun değerlerini bulabiliriz.
salary_merged1['anomaly_score'] = model2.decision_function(X)

#Benzer şekilde eğitimli modeli çağırarak anomali sütununun değerlerini bulabiliriz.
salary_merged1['outlier'] = model2.predict(X)


# In[ ]:


#Anormallik tespit edilenleri görebiliriz.

salary_merged1[salary_merged1.outlier==-1].sort_values(by='total_earnings',ascending=False)


# ### Veri Analizi

# In[ ]:


#Normal değerde tespit edilenleri görebiliriz.

salary_merged1[salary_merged1.outlier==1].sort_values(by='total_earnings',ascending=False)


# In[ ]:


#Anormal olmayan değerlerimizi yeni oluşturduğumuz DataFrame'e atarız.

salary_merged2=salary_merged1[(salary_merged1.outlier == 1)].sort_values(by='total_earnings',ascending=False)

salary_merged2 = salary_merged2[['name','department_name','title', 'total_earnings']].copy()


# In[ ]:


#Yeni departman ortalamaları oluşturulur.

salary_average2=(salary_merged2.groupby('department_name')
 .mean()
 .reset_index()
 .rename(columns = {'total_earnings':'dept_average'}))

#Şimdi departman ortalamasını ve bireyin maaşını tekrar birleştiririz.

salary_merged2 = pd.merge(salary_merged2, salary_average2, on = 'department_name')


# In[ ]:


#Tespit edilen anormallikler sonrası tabloyu inceleyelim.

salary_merged2.sort_values(by='total_earnings',ascending=False)



# In[ ]:


#Departman sayısı

len(salary_merged2['dept_average'].unique())


# In[ ]:


#En yüksek maaşa sahip birey bilgileri

salary_merged2.iloc[salary_merged2['total_earnings'].idxmax()]


# In[ ]:


#En fazla çalışanı bulunan 10 departman

salary_merged2['department_name'].value_counts()[:10]


# In[ ]:


#En az çalışanı bulunan 10 departman

salary_merged2[salary_merged2.department_name=='Boston Police Department'].head(5)


# In[ ]:


#Veri kümesine genel bakış

salary_merged2[['total_earnings', 'dept_average']].groupby(['dept_average']).describe()


# In[ ]:


#Bireyin toplam kazancı ve departmanının ortalama maaşı arasındaki korelasyon

salary_merged2.corr()


# ### Veri Görselleştime

# In[ ]:


#Kutu Grafiği

fig, ax = plt.subplots(figsize=(18,40))
ax.set_ylabel('dept_average')
# 2 veri arasındaki kutu grafiği
_a = salary_merged2[['total_earnings', 'dept_average']].boxplot(by='dept_average',vert=False, figsize=(6,4), sym='b.', ax=ax)


# In[ ]:


#Dağılım Grafiği

plt.figure(figsize=(20,10)) 
__ = sns.regplot(data=salary_merged2, x='total_earnings', y='dept_average')


# In[ ]:


#Tablosal korelasyon matrisini görebiliriz.

corr = salary_merged2.corr()
___ , ax = plt.subplots(figsize=(13,10)) 

___ = sns.heatmap(corr, ax=ax,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                )


# In[ ]:


#En yüksek ortalamaya sahip 5 departman

higher_values = salary_merged2.groupby(['department_name'])['dept_average'].mean().nlargest(5).values
higher_index = salary_merged2.groupby(['department_name'])['dept_average'].mean().nlargest(5).index


# In[ ]:


#En yüksek ortalamaya sahip 5 departmanın grafiğini bakarız.

plt.figure(figsize=(12,5))
plt.ylabel("title")
plt.xlabel("total_earnings")
pall =sns.color_palette("hls", 8)
sns.barplot(y=higher_index,x=higher_values,palette=pall)


# In[ ]:


#En düşük ortalamaya sahip 5 departman

lower_values = salary_merged2.groupby(['department_name'])['dept_average'].mean().nsmallest(5).values
lower_index = salary_merged2.groupby(['department_name'])['dept_average'].mean().nsmallest(5).index


# In[ ]:


#En düşük ortalamaya sahip 5 departmana bakarız.

plt.figure(figsize=(12,5))
plt.ylabel("department_name")
plt.xlabel("total_earnings")
pall =sns.color_palette("hls", 8)
sns.barplot(y=lower_index,x=lower_values,palette=pall)


# In[ ]:


#Departman ortalamasının dağılım grafiği

sns.distplot(salary_merged2['dept_average'],color='black',)
plt.title('dept_average')
sns.despine()


# In[ ]:


#Toplam kazancın dağılım grafiği

sns.distplot(salary_merged2['total_earnings'],color='black',)
plt.title('total_earnings')
sns.despine()


# In[ ]:


#Toplam maaş ve departman ortalamasının ilişkisel grafiği

sns.relplot(x="dept_average",y="total_earnings",
            data=salary_merged2,
            alpha=0.5,
           edgecolor=None)


# In[ ]:


#Departman ortalaması 100.000'den fazla olan departmanların keman grafiği

sal=salary_merged[salary_merged.dept_average>100000]
sns.catplot(x="dept_average",
            y="total_earnings",
            data=sal,
           inner = "box", 
           kind="violin",
           palette="Pastel1")


# In[ ]:


#Toplam kazanç ve departman ortalamasının birleştirilmiş grafik ile gösterilmesi

sns.jointplot(x="dept_average",y="total_earnings",kind='kde',data=salary_merged2)


# In[ ]:


#Yeni görselleştirmelerde kullanmak için verimizi departman ortalamasına göre gruplara ayıralım.

sal5=salary_merged2[salary_merged2.dept_average<=60000]
sal6=salary_merged2[(salary_merged2.dept_average>60000)&(salary_merged2.dept_average<=80000)] 
sal7=salary_merged2[(salary_merged2.dept_average>80000)&(salary_merged2.dept_average<=100000)] 
sal8=salary_merged2[salary_merged2.dept_average>100000]


# In[ ]:


#Farklı analiz yöntemleriyle analiz yapalım.
#Ortalaması 60.000'den küçük departmanlar

print(sal5['department_name'].unique())


# In[ ]:


#Departman ortalamalarına göre ayrılmış tablolardaki farklı meslek sayıları

print('Farklı iş ünvanı sayısı:', len(salary_merged2['title'].unique()) - 1)
print('sal5 -> farklı iş ünvanı sayısı:', len(sal5['title'].unique()) - 1)
print('sal6 -> farklı iş ünvanı sayısı:', len(sal6['title'].unique()) - 1)
print('sal7 -> farklı iş ünvanı sayısı:', len(sal7['title'].unique()) - 1)
print('sal8 -> farklı iş ünvanı sayısı:', len(sal8['title'].unique()) - 1)


# In[ ]:


#Ortalaması 60.000-80.000 arasında en fazla olan 20 iş

sal6['title'].value_counts()[:20]


# In[ ]:


#En fazla çalışanı olan mesleklerin gösterimi.

print("En fazla elemanı olan 3 meslek\n",salary_merged2['title'].value_counts().head(3),sep="\n",
      end="\n------------------------------------------------------------------------------\n")
print("60.000'den düşük bütçeli departmanlarda en fazla elemanı olan 3 meslek\n",sal5['title'].value_counts().head(3),sep="\n",
      end="\n------------------------------------------------------------------------------\n")
print("60.000-80.000 arası bütçeye sahip departmanlarda en fazla elemanı olan 3 meslek\n",sal6['title'].value_counts().head(3),sep="\n",
      end="\n------------------------------------------------------------------------------\n")
print("80.000-100.000 arası bütçeye sahip departmanlarda en fazla elemanı olan 3 meslek\n",sal7['title'].value_counts().head(3),sep="\n",
      end="\n------------------------------------------------------------------------------\n")
print("100.000'den yüksek bütçeli departmanlarda en fazla elemanı olan 3 meslek\n",sal8['title'].value_counts().head(3),sep="\n",
      end="\n------------------------------------------------------------------------------\n")


# In[ ]:


#Departman başına benzersiz ünvan sayısı.

sal5.groupby('department_name')['title'].nunique().plot(kind='bar')
plt.show()


# In[ ]:


#Maaş ve departman ortalama kazancının çift yönlü ilişkisinin gösterimi.

g = sns.pairplot(sal6[['total_earnings','dept_average']])


# In[ ]:


#Departmanlara göre kazanılan maaşların sürü grafiği ile gösterimi.

sns.set(style="whitegrid")
plt.figure(figsize=(20,8))
ax = sns.swarmplot(x = "department_name",
              y = 'total_earnings', 
              data = sal7,
              size = 3)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()


# In[ ]:


#Departmanlara göre kazançların grafiği.

sns.relplot(x="dept_average", y="total_earnings", data = sal8,hue='department_name')


# In[ ]:


#60.000-80.000 arası departmanların kümülatif histogram grafiği.


sal6.dept_average.hist(cumulative=True)


# In[ ]:


#80.000-100.000 arası ortalama maaşı olan departmanların grafiği

sns.set(style="whitegrid")
plt.figure(figsize=(20,8))

ax = sns.barplot(x="department_name", y="dept_average", data=sal7,palette='muted')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title('Departman Ortalamaları')


# In[ ]:


#Maaş yoğunluğu grafiği.

x0 = sal5['total_earnings']
x1 = sal6['total_earnings']
x2 = sal7['total_earnings']
x3 = sal8['total_earnings']
fig, ax = plt.subplots(figsize=(20, 6))
sns.kdeplot(x0, label="60.000'den Düşük Ortalamaya Sahip Departman Maaşları", shade=True, ax=ax)
sns.kdeplot(x1, label="60.000-80.000 Arası Ortalamaya Sahip Departman Maaşları", shade=True, ax=ax)
sns.kdeplot(x2, label="80.000-100.00 Arası Ortalamaya Sahip Departman Maaşları", shade=True, ax=ax)
sns.kdeplot(x3, label="100.000'den Yüksek Ortalamaya Sahip Departman Maaşları", shade=True, ax=ax)
plt.xlabel('Alınan Maaş')
plt.ylabel('Yoğunluk')
title = plt.title('Maaşlar Dağılımı')


# In[ ]:


#En fazla işçisi olan 30 meslek çubuk grafiği gösterimi.

plt.figure(figsize=(13,8))
sectors = salary_merged2['title'].value_counts()[0:30]
sns.barplot(y=sectors.index, x=sectors.values, alpha=0.6)
plt.xlabel('İş Sayısı', fontsize=16)
plt.ylabel("İş Ünvanı", fontsize=16)
plt.title("Sektör İşçi Sayısı")
plt.show();


# In[ ]:


#Mesleklerin adlarını küçük harflere dönüştürürüz.

salary_merged2.title = salary_merged2.title.str.lower()

#250 değerini kritik olarak ayarlıyoruz.
criteria = salary_merged2.title.value_counts()>250

#250'den fazla olanları alırız.
jobtitlelist = salary_merged2.title.value_counts()[criteria].reset_index()

#Meslek ve kazançlardan oluşan bir dataframe oluşturalım.
df = salary_merged2[['title', 'total_earnings']]

df = df[df.title.isin(jobtitlelist['index'])]

pivoted_data = df.pivot_table('total_earnings', index='title' , aggfunc=np.mean)

sorted_salaries = pivoted_data.sort_values(by='total_earnings', ascending= False)


# In[ ]:


#250'den fazla işçisi olan mesleklerin ortalama maaş grafiği.

sorted_salaries=sorted_salaries.reset_index()
sns.set(style="whitegrid")
plt.figure(figsize=(12,8))

ax = sns.barplot(x="title", y="total_earnings", data=sorted_salaries,palette='muted')
plt.xlabel('Meslek', fontsize=16)
plt.ylabel("Ortalama Maaş", fontsize=16)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title('Meslek Ortalama Maaşları')


# In[ ]:









#Lightning kütüphanesini import edip görselleştirmelerini inceleyelim.

from lightning import Lightning

lgn = Lightning(ipython=True,local=True)


# In[ ]:


#Dağılım grafiği gösterimi.
#Shift tuşuyla fırçalama yapılabilir

c = [100,200,100]
lgn.scatter(sal7['dept_average'], sal7['total_earnings'],alpha=0.8,brush=True,color=c,size=5)


# In[ ]:


#Dağılım grafiğinin farklı gösterimi.

from numpy import random
v = random.rand(8751)
lgn.scatter(sal6['total_earnings'], sal6['dept_average'], values=v, alpha=0.6, colormap='YlOrRd',
            xaxis='Toplam Maaş', yaxis='Departman Ortalaması')


# In[ ]:


#Maaş ve departman ortalama maaşı çizgi grafiği

g1=salary_merged2['dept_average'].sort_values().unique()
g2=salary_merged2['total_earnings'].sort_values().unique()


# In[ ]:


#Departman ortalaması artış grafiği.

lgn.line(g1, thickness=8, color=[0,0,0])


# In[ ]:


#Alınan maaş artışı grafiği

lgn.line(g2, thickness=10, color=[25,25,112])


# In[ ]:


#Bokeh kütüphanesini import edelim.

from bokeh.plotting import figure 
from bokeh.io import push_notebook,output_notebook, show
TOOLS = "crosshair,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select"


# In[ ]:


output_notebook()


# In[ ]:


#Departman ortalamalarına göre kazançların grafiği
p = figure(tools=TOOLS,plot_width = 400, plot_height = 400, 
           title = 'Departman Ortalamalarına Göre Kazançlar',
           x_axis_label = 'Departman Ortalaması', y_axis_label = 'Toplam Kazanç')
p.circle(sal7['dept_average'], sal7['total_earnings'], size = 12, color = 'red', alpha = 0.6,line_color='yellow')
p.circle(sal6['dept_average'], sal6['total_earnings'], size = 12, color = 'navy',alpha = 0.6,line_color="yellow")
show(p)


# In[ ]:


#Fareyle üzerinde geldiğimiz departmanın maaşları kırmızı renge dönüşür.

from bokeh.models import HoverTool

hover = HoverTool(tooltips=None,mode='hline')
x=sal8['total_earnings']
y=sal8['dept_average']

plot = figure(tools=[hover, 'crosshair,pan,wheel_zoom,box_zoom,reset'],plot_height = 500, plot_width = 1000)
plot.xaxis.axis_label = 'Maaş'
plot.yaxis.axis_label = 'Departman Ortalama Maaş'
plot.circle(x,y, size = 12, 
            color = 'navy', alpha = 0.6,line_color='yellow' , hover_color='red')

#Gösterim

show(plot)


# In[ ]:


#En yüksek işçi sayısına sahip 20 departmanı ve işçi sayılarını alalım.

dpname = salary_merged2.groupby(['department_name'], as_index=False).count()
top_dpname = dpname.sort_values(by='dept_average', ascending=False).head(20)
top_dpname = top_dpname.reset_index().drop(['index', 'name','title','total_earnings'], axis=1)
top_dpname_list = top_dpname.department_name.values.tolist()
top_dpname_count = top_dpname.dept_average.astype(float).values.tolist()

print(top_dpname_list)
print(top_dpname_count)


# In[ ]:


#En yüksek işçi sayısına sahip 20 departmanın çalıştırdığı işçi sayısının grafiği
p = figure(x_range=top_dpname_list, plot_height = 500, plot_width = 500)
p.xgrid.visible = False
p.xaxis.major_label_orientation = 3.14/4
p.xaxis.axis_label = 'Departmanlar'
p.ygrid.visible = False
p.yaxis.axis_label = 'Departman İşçi Sayısı'
p.circle(y=top_dpname_count, x=top_dpname_list, size=15, fill_color="black")
show(p)


# In[ ]:


#Departman ortalamasını çizgi grafiği şeklinde,ortalamaya göre maaşı daire şeklinde göster

k = figure(tools=TOOLS,plot_width=800, plot_height=600)
k.circle(sal5['dept_average'], sal5['total_earnings'], size=4, color='orange')
k.line(sal5['dept_average'], sal5['dept_average'], color='black')

#Özellikleri değiştirebiliriz

k.grid.grid_line_alpha = 0
k.xaxis.axis_label = 'Departman Ortalaması'
k.yaxis.axis_label = 'Maaş'
k.ygrid.band_fill_color = "grey"
k.ygrid.band_fill_alpha = 0.5

#Gösterim

show(k)


# In[ ]:


#Elmas çizimi şeklinde bir gösterim

s = figure(tools=TOOLS,plot_width = 600, plot_height = 600, 
           title = 'Departman Ortalaması-Toplam Kazanç',
           x_axis_label = 'Departman Ortalaması', y_axis_label = 'Toplam Kazanç')
s.diamond(sal7['dept_average'], sal7['total_earnings'],color = 'black', alpha = 0.6,line_color='yellow',size=12)
show(s)


# In[ ]:


# En fazla işçisi bulunan 10 departmanın pasta grafiği.

from math import pi
from bokeh.palettes import Category20c
from bokeh.transform import cumsum



# In[ ]:


# En fazla işçisi bulunan 10 departmanın pasta grafiği.
d_name = salary_merged2['department_name'].value_counts()[:10]
data = pd.Series(d_name).reset_index(name='value').rename(columns={'index':'department_name'})
data['angle'] = data['value']/data['value'].sum() * 2*pi
data['color'] = Category20c[len(d_name)]

p = figure(plot_height=350, title="Pasta Grafiği", toolbar_location=None,
           tools="hover", tooltips="@department_name: @value", x_range=(-0.5, 1.0))
p.wedge(x=0, y=1, radius=0.4,
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="black", fill_color='color',legend='department_name', source=data)
p.axis.axis_label=None
p.axis.visible=False
p.grid.grid_line_color = None
show(p)


# In[ ]:


df = salary_merged2.copy()


# #### MACHINE LEARNING

# In[ ]:


# Makine öğrenmesi kullanarak çalışanların maaşlarının tahmini yapılacaktır.
# Maaş tahmini için linear regression ve RandomForest Regression algoritmaları kullanılacaktır. 
# Tahminlerimizi yapmak için sklearn kütüphanesinden algoritmaları eklenir.
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


# Algoritmaların performanslarını ölçmek için metrikler eklerinir.
from sklearn.metrics import r2_score , explained_variance_score , mean_absolute_error
from sklearn.metrics import mean_squared_error , max_error


# In[ ]:


# Makine öğrenimi için nesne türündeki departman ve meslek sütunlarını önce kategorik değişkene çevirilir.

df['department_name']=pd.Categorical(df['department_name'])
df['title']=pd.Categorical(df['title'])


# In[ ]:


# Dataframe'deki yeni tipler.

df.dtypes


# In[ ]:


# Daha başarılı sonuçlar elde etmek için kategorik değişkenler kukla değişkenlere dönüştürülür.

df= pd.get_dummies(df)


# In[ ]:


# Veri tablosunun yeni halinin görünüşü.


df.head()


# ### LİNEAR REGRESSİON İLE MAAŞ TAHMİNİ

# In[ ]:


# Linear Regression için kullanılacak veriler seçilir.
# X maaş haricindeki tabloda olan verileri içerir.
X = df[df.loc[ : , df.columns != 'total_earnings'].columns]

#y hedef verisidir.Çalışanların maaşını içerir.
y = df['total_earnings']


# In[ ]:


#Linear Regression modeli tanımlanır.
lm = LinearRegression()
#Tanımlanan model eğitilir.
lm.fit(X, y)
lm


# In[ ]:


#Çalışan maaşlarının tahmini yapılır
yhat = lm.predict(X)


# In[ ]:


#Linear Regression ile yapılan tahmin tabloya sütun olarak eklenir.
df['linear_predict']=yhat


# In[ ]:


#Departman ortalaması çalışanın kazandığı maaş ve yapılan tahminler genel bakış.
df[['dept_average','total_earnings','linear_predict']]


# ### RANDOM FOREST ALGORİTMASI İLE MAAŞ TAHMİNİ

# In[ ]:


#Random Forest için kullanılacak veriler seçilir.
#x maaş haricindeki tabloda olan verileri içerir.
x = df[df.loc[ : , df.columns != 'total_earnings'].columns]
#Y hedef veridir.Çalışan maaşını içerir.
Y = df['total_earnings']


# In[ ]:


#Model tanımlanır ve eğitilir.
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(x, Y)


# In[ ]:


#Random forest algoritmasıyla çalışan maaşlarının tahmini yapılır.
y_predict = regressor.predict(x)


# In[ ]:


#Yapılan tahminler tabloya eklenir.
df['random_predict'] = y_predict


# In[ ]:


#Yapılan tahminler ve maaşlara genel bakış.
df[['dept_average','total_earnings','random_predict']]


# In[ ]:


#Linear regression ve random forest algoritmalarıyla yapılan tahminler tabloya kalıcı olarak eklenir.
df = df[['dept_average','total_earnings','linear_predict','random_predict']]


# In[ ]:


#Başta kopyası alınan dataframe(df2) ile yeni tahminleri içeren dataframe birleştirilir.
df2.drop(columns=['total_earnings','dept_average'],inplace=True)


# In[ ]:


#İki tablonun birleşmesinin sonucu dataframe son halini alır.
salary = pd.concat([df2, df], axis=1, sort=False)


# In[ ]:


#Toplam maaş ve algoritmaların tahminleri
salary.head()


# In[ ]:


#Polis memurlarının aldığı maaş ve tahmini maaşları
salary[salary.title=='Police Officer'].head()


# ### Performans Metrikleri

# In[ ]:


#Algoritmaların performansını ölçmek için metrikler kullanılır.
#r kare puanının 1 olması mümkün olan en iyi puandır ve tahminin kuvvetli olduğunu gösterir.
print("linear regresyon r kare puanı :",r2_score(y,yhat))
print("random forest r kare puanı    :",r2_score(Y,y_predict))


# In[ ]:


#Elde edilen tahminlerin ortalama kare hatası
print("linear regresyon ortalama kare hatası :",mean_squared_error(y , yhat))
print("random forest ortalama kare hatası    :",mean_squared_error(Y,y_predict))


# In[ ]:


#Tahminleri ortalama mutlak hatası
print("linear regresyon ortalama mutlak hatası :",mean_absolute_error(y , yhat))
print("random forest ortalama mutlak hatası    :",mean_absolute_error(Y,y_predict))



# In[ ]:


#max_error maksimum kalan hatayı gösterir.
print("linear regresyon maksimum hatası :",max_error(y , yhat))
print("random forest maksimum hatası    :",max_error(Y,y_predict))


# In[ ]:


#Açıklanan varyans puanları
#Puanların 1'e yakın olması tahminlerin güçlü olduğunu gösterir.
print("linear regresyon varyans puanı :",explained_variance_score(y, yhat))
print("random forest varyans puanı    :",explained_variance_score(Y,y_predict))


# In[ ]:


#Maaşların ve makine tahminlerinin görselleştirilerek karşılaştırılması.
x0 = salary['total_earnings']
x1 = salary['linear_predict']
x2 = salary['random_predict']

fig, ax = plt.subplots(figsize=(14, 9))

sns.kdeplot(x0, label="Veri seti alınan maaşlar", shade=True, ax=ax)
sns.kdeplot(x1, label="Linear regresyona göre tahmin edilen maaşlar", shade=True, ax=ax)
sns.kdeplot(x2, label="Random foreste göre tahmin edilen maaşlar", shade=True, ax=ax)

plt.xlabel('Maaş')
plt.ylabel('Yoğunluk')
title = plt.title('Maaşlar Dağılımı')


# ### Dask Kütüphanesinin İncelenmesi

# In[ ]:


salary.to_csv('bostonsalaries.csv')


# In[ ]:


#Kütüphane import edilir.
import dask.dataframe as dd


# In[ ]:


ddf = dd.read_csv(r"C:\Users\hersann\bostonsalaries.csv", encoding="latin-1")


# In[ ]:


#Pandasın aksine Dask tembeldir ve buraya hiçbir veri yazdırmaz.
ddf


# In[ ]:


#Veri türlerine bakılabilir.
ddf.dtypes


# In[ ]:


#Standart pandas komutları kullanılabilir.
ddf.head()


# In[ ]:


#Sonuç pandas veri çerçevesinde görülmek istendiğinde .compute() kullanılır.
computed_ddf = ddf.compute()
type(computed_ddf)


# In[ ]:


computed_ddf


# In[ ]:


#Dask istemcisi isteğe bağlı başlatılabilir
#Hesaplamalar hakkında kullanışlı bir gösterge tablosu sağlar.
from dask.distributed import Client, progress
client = Client(n_workers=4)
client
#Yeni sekmede açılır.


# In[ ]:


get_ipython().run_cell_magic('time', '', '#Görev akışını bir bağlam bloğunda toplama\n#Bu, bu bloğun etkin olduğu sırada çalıştırılan her görev hakkında tanılama bilgileri sağlar.\n\nfrom dask.distributed import get_task_stream\nwith get_task_stream() as ts:\n    ddf.compute()\nts.data')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Dask için bir ilerleme çubuğu.\nfrom dask.diagnostics import ProgressBar\n#Veri setinin toplam maaşının ortalaması\nwith ProgressBar():\n    display(ddf.total_earnings.mean().compute())')


# In[ ]:


get_ipython().run_cell_magic('time', '', '#Veri seti için linear regresyonla tahmin edilen maaşın ortalaması\nwith ProgressBar():\n    display(ddf.linear_predict.mean().compute())')


# In[ ]:


get_ipython().run_cell_magic('time', '', '#Veri seti için linear regresyonla tahmin edilen maaşın ortalaması\nwith ProgressBar():\n    display(ddf.random_predict.mean().compute())')


# In[ ]:


#Bu, gelecekteki hesaplamaların çok daha hızlı olmasını sağlar.
#Dask tüm verilerin nerede yaşadığını bilir ve isimle temiz bir şekilde endeksler. 
#Sonuç olarak, rasgele erişim gibi faaliyetler ucuz ve verimlidir.
ddf = ddf.persist()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'ddf.linear_predict.mean().compute()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '#Veri tablosunun kaç satırdan oluştuğu sorgulanır.\nddf.shape[0].compute()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '#Departmanların kaç çalışanı olduğu gösterilir\nddf.department_name.value_counts().compute()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '#Dask dataframe hakkında bilgi.\nddf.compute().info()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '#Departmanlarda çalışan maaşlarının standart sapması.\nddf.groupby("department_name").total_earnings.std().compute()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '#Departmanlarda çalışanların maaş tahminlerinin standart sapması.\nddf.groupby("department_name").linear_predict.std().compute()')


# In[ ]:


get_ipython().run_cell_magic('time', '', "#Departman ortalamaları ve maaş korelasyonu.\nddf[['dept_average','total_earnings']].corr().compute()")


# In[ ]:


get_ipython().run_cell_magic('time', '', "#linear regresyon tahminine göre departman ortalamaları ve tahmini maaş korelasyonu.\nddf[['dept_average','linear_predict']].corr().compute()")


# In[ ]:


get_ipython().run_cell_magic('time', '', "#random forest tahminine göre departman ortalamaları ve tahmini maaş korelasyonu.\nddf[['dept_average','random_predict']].corr().compute()")


# In[ ]:


get_ipython().run_cell_magic('time', '', "#linear ve random forest algoritma tahminleri karşılaştırması\nwith ProgressBar():\n    display(ddf[['linear_predict','random_predict']].head())")


# In[ ]:


get_ipython().run_cell_magic('time', '', "#Mesleklere göre alınan maaşlar ve linear regresyona göre tahmin edilen maaşların karşılaştırılması.\nwith ProgressBar():\n    display(ddf[['title','total_earnings','linear_predict']].compute())")


# In[ ]:


get_ipython().run_cell_magic('time', '', "#Mesleklere göre alınan maaşlar ve random foreste göre tahmin edilen maaşların karşılaştırılması. \nwith ProgressBar():\n    display(ddf[['title', 'total_earnings' , 'random_predict']].tail())")


# In[ ]:


get_ipython().run_cell_magic('time', '', "#Departmanlarda çalışan sayısı\nwith ProgressBar():\n    dept_count = ddf['department_name'].value_counts().compute()\ndept_count")


# In[ ]:


get_ipython().run_cell_magic('time', '', "#Departman ortalması 60.000-90.000 olan çalışanları filtreleme\nwith ProgressBar():\n    condition = (ddf['dept_average'] > 60000) & (ddf['dept_average'] < 90000)    \n    ddf_filtered = ddf[condition]\n  \n\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', '#Filtrelenmiş çalışan listesi\nddf_filtered.head()')


# In[ ]:


get_ipython().run_cell_magic('time', '', "#Mesleklere göre maaş ortalaması.\nddf.groupby('title').total_earnings.mean().compute()")


# In[ ]:


get_ipython().run_cell_magic('time', '', "#Mesleklere göre linear regresyon algoritması tahmini maaş ortalaması.\nddf.groupby('title').linear_predict.mean().compute()")


# In[ ]:


get_ipython().run_cell_magic('time', '', "#Mesleklere göre random forest algoritması tahmini maaş ortalaması.\nddf.groupby('title').random_predict.mean().compute()")

