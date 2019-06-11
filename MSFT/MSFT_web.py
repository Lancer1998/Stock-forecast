
# coding: utf-8

# In[23]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


from stocker import Stocker


# In[25]:


microsoft = Stocker('MSFT')


# In[26]:


stock_history = microsoft.stock
stock_history.head()


# In[27]:


f = open('msft1.txt','w')
f.write(str(stock_history.head()))
f.close()


# In[28]:


microsoft.plot_stock()


# In[29]:


f = open('msft2.txt','w')
f.write(str(microsoft.plot_stock()))
f.close()


# In[30]:


microsoft.plot_stock(start_date = '2000-01-03', end_date = '2018-01-16', 
                     stats = ['Daily Change'], plot_type='pct')


# In[32]:


f = open('msft3.txt','w')
f.write(str(microsoft.plot_stock(start_date = '2015-01-01', end_date = None, 
                     stats = ['Daily Change'], plot_type='pct')))
f.close()


# In[33]:


microsoft.plot_stock(start_date = '2000-01-03', end_date = '2018-01-16', 
                     stats = ['Adj. Volume'], plot_type='pct')


# In[34]:


f = open('msft4.txt','w')
f.write(str(microsoft.plot_stock(start_date = '2015-01-01', end_date = None, 
                     stats = ['Adj. Volume'], plot_type='pct')))
f.close()


# In[35]:


microsoft.buy_and_hold(start_date='1986-03-13', end_date='2018-01-16', nshares=100)


# In[36]:


f = open('msft5.txt','w')
f.write(str(microsoft.buy_and_hold(start_date='1986-03-13', end_date='2018-01-16', nshares=100)))
f.close()


# In[37]:


microsoft.buy_and_hold(start_date='1999-01-05', end_date='2002-01-03', nshares=100)


# In[38]:


f = open('msft6.txt','w')
f.write(str(microsoft.buy_and_hold(start_date='1999-01-05', end_date='2002-01-03', nshares=100)))
f.close()


# In[39]:


microsoft.changepoint_date_analysis()


# In[40]:


f = open('msft7.txt','w')
f.write(str(microsoft.changepoint_date_analysis()))
f.close()


# In[41]:


model, future = microsoft.create_prophet_model(days=30)


# In[42]:


f = open('msft8.txt','w')
f.write(str(microsoft.create_prophet_model(days=30)))
f.close()

