
# coding: utf-8

# ### 英伦航空公司

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from stocker import Stocker


# In[4]:


britishmidland = Stocker('BMI')


# In[5]:


stock_history = britishmidland.stock
stock_history.head()


# In[6]:


f = open('bmi1.txt','w')
f.write(str(stock_history.head()))
f.close()


# In[7]:


britishmidland.plot_stock()


# In[9]:


f = open('bmi2.txt','w')
f.write(str(britishmidland.plot_stock()))
f.close()


# In[10]:


britishmidland.plot_stock(start_date = '2000-01-03', end_date = '2018-01-16', 
                     stats = ['Daily Change'], plot_type='pct')


# In[11]:


f = open('./BMI/bmi3.txt','w')
f.write(str(britishmidland.plot_stock(start_date = '2015-01-01', end_date = None, 
                     stats = ['Daily Change'], plot_type='pct')))
f.close()


# In[12]:


britishmidland.plot_stock(start_date = '2000-01-03', end_date = '2018-01-16', 
                     stats = ['Adj. Volume'], plot_type='pct')


# In[13]:


f = open('./BMI/bmi4.txt','w')
f.write(str(britishmidland.plot_stock(start_date = '2015-01-01', end_date = None, 
                     stats = ['Adj. Volume'], plot_type='pct')))
f.close()


# In[14]:


britishmidland.buy_and_hold(start_date='1986-03-13', end_date='2018-01-16', nshares=100)


# In[15]:


f = open('./BMI/bmi5.txt','w')
f.write(str(britishmidland.buy_and_hold(start_date='1986-03-13', end_date='2018-01-16', nshares=100)))
f.close()


# In[16]:


britishmidland.buy_and_hold(start_date='1999-01-05', end_date='2002-01-03', nshares=100)


# In[17]:


f = open('./BMI/bmi6.txt','w')
f.write(str(britishmidland.buy_and_hold(start_date='1999-01-05', end_date='2002-01-03', nshares=100)))
f.close()


# In[18]:


britishmidland.changepoint_date_analysis()


# In[19]:


f = open('./BMI/bmi7.txt','w')
f.write(str(britishmidland.changepoint_date_analysis()))
f.close()


# In[20]:


model, future = britishmidland.create_prophet_model(days=30)


# In[21]:


f = open('./BMI/bmi8.txt','w')
f.write(str(britishmidland.create_prophet_model(days=30)))
f.close()

