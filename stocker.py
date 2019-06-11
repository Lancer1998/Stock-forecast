
# coding: utf-8

# In[2]:


import quandl
import pandas as pd
import numpy as np
import fbprophet
import pytrends
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
import matplotlib


# ### 用于分析和尝试预测未来价格的类

# In[3]:


class Stocker():
    
    # 初始化需要股票代码
    def __init__(self, ticker, exchange='WIKI'):
        
        # 强制资本化
        ticker = ticker.upper()
        
        # 符号用来标记图
        self.symbol = ticker
        
        # 使用Personal Api Key
        # quandl.ApiConfig.api_key = 'YourKeyHere'

        # 检索财务数据
        try:
            stock = quandl.get('%s/%s' % (exchange, ticker))
        
        except Exception as e:
            print('Error Retrieving Data.')
            print(e)
            return
        
        # S将索引设置为名为Date的列
        stock = stock.reset_index(level=0)
        
        # 需要的专栏
        stock['ds'] = stock['Date']

        if ('Adj. Close' not in stock.columns):
            stock['Adj. Close'] = stock['Close']
            stock['Adj. Open'] = stock['Open']
        
        stock['y'] = stock['Adj. Close']
        stock['Daily Change'] = stock['Adj. Close'] - stock['Adj. Open']
        
        # 分配为类属性的数据
        self.stock = stock.copy()
        
        # 范围内最小和最大日期
        self.min_date = min(stock['Date'])
        self.max_date = max(stock['Date'])
        
        # 查询最大和最小价格以及发生的日期
        self.max_price = np.max(self.stock['y'])
        self.min_price = np.min(self.stock['y'])
        
        self.min_price_date = self.stock[self.stock['y'] == self.min_price]['Date']
        self.min_price_date = self.min_price_date[self.min_price_date.index[0]]
        self.max_price_date = self.stock[self.stock['y'] == self.max_price]['Date']
        self.max_price_date = self.max_price_date[self.max_price_date.index[0]]
        
        # 开盘价格
        self.starting_price = float(self.stock.loc[0, 'Adj. Open'])
        
        # 最近价格
        self.most_recent_price = float(self.stock.loc[self.stock.index[-1], 'y'])

        # 是否圆形日期
        self.round_dates = True
        
        # 需要训练的数据年龄
        self.training_years = 3

        # Prophet参数，默认先于库
        self.changepoint_prior_scale = 0.05 
        self.weekly_seasonality = False
        self.daily_seasonality = False
        self.monthly_seasonality = True
        self.yearly_seasonality = True
        self.changepoints = None
        
        print('{} Stocker Initialized. Data covers {} to {}.'.format(self.symbol,
                                                                     self.min_date,
                                                                     self.max_date))
    
    """
    确保开始日期和结束日期在范围内，并且可以转换为pandas日期时间。 以正确的格式返回日期
    """
    def handle_dates(self, start_date, end_date):
        
        
        # 默认开始和结束日期是数据的开始和结束
        if start_date is None:
            start_date = self.min_date
        if end_date is None:
            end_date = self.max_date
        
        try:
            # 转换为pandas datetime以索引数据帧
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
        
        except Exception as e:
            print('Enter valid pandas date format.')
            print(e)
            return
        
        valid_start = False
        valid_end = False
        
        # 用户将继续输入日期
        while (not valid_start) & (not valid_end):
            valid_end = True
            valid_start = True
            
            if end_date < start_date:
                print('结束日期必须晚于开始日期')
                start_date = pd.to_datetime(input('输入新的开始日期：'))
                end_date= pd.to_datetime(input('输入新的结束日期：'))
                valid_end = False
                valid_start = False
            
            else: 
                if end_date > self.max_date:
                    print('结束日期超出数据范围')
                    end_date= pd.to_datetime(input('输入洗的结束日期：'))
                    valid_end = False

                if start_date < self.min_date:
                    print('开始日期在数据范围前')
                    start_date = pd.to_datetime(input('输入新的开始日期：'))
                    valid_start = False
                
        
        return start_date, end_date
        
    """
    返回修剪到指定范围的数据帧
    """
    def make_df(self, start_date, end_date, df=None):
        
        # 默认是使用对象库存数据
        if not df:
            df = self.stock.copy()
        
        
        start_date, end_date = self.handle_dates(start_date, end_date)
        
        # 记录开始和结束日期是否在数据中
        start_in = True
        end_in = True

        # 如果用户想要舍入日期
        if self.round_dates:
            # 记录开始和结束日期是否为df
            if (start_date not in list(df['Date'])):
                start_in = False
            if (end_date not in list(df['Date'])):
                end_in = False

            # 如果两者都不在数据框中，则舍入两者
            if (not end_in) & (not start_in):
                trim_df = df[(df['Date'] >= start_date) & 
                             (df['Date'] <= end_date)]
            
            else:
                # 如果两者都在数据框中，则不会四舍五入
                if (end_in) & (start_in):
                    trim_df = df[(df['Date'] >= start_date) & 
                                 (df['Date'] <= end_date)]
                else:
                    # 如果只缺少start，则开始循环
                    if (not start_in):
                        trim_df = df[(df['Date'] > start_date) & 
                                     (df['Date'] <= end_date)]
                    # 如果只是结束了圆形结束
                    elif (not end_in):
                        trim_df = df[(df['Date'] >= start_date) & 
                                     (df['Date'] < end_date)]

        
        else:
            valid_start = False
            valid_end = False
            while (not valid_start) & (not valid_end):
                start_date, end_date = self.handle_dates(start_date, end_date)
                
                # 没有圆形日期，如果数据不在，则打印消息并返回
                if (start_date in list(df['Date'])):
                    valid_start = True
                if (end_date in list(df['Date'])):
                    valid_end = True
                    
                # 检查以确保日期在数据中
                if (start_date not in list(df['Date'])):
                    print('开始日期不在数据中（超出范围或不在交易日）。')
                    start_date = pd.to_datetime(input(prompt='输入新的开始日期：'))
                    
                elif (end_date not in list(df['Date'])):
                    print('结束日期不在数据中（超出范围或不在交易日）。')
                    end_date = pd.to_datetime(input(prompt='输入新的结束日期：') )

            # 日期不是四舍五入
            trim_df = df[(df['Date'] >= start_date) & 
                         (df['Date'] <= end_date.date)]

        
            
        return trim_df


    # 基本历史图和基本统计
    def plot_stock(self, start_date=None, end_date=None, stats=['Adj. Close'], plot_type='basic'):
        
        self.reset_plot()
        
        if start_date is None:
            start_date = self.min_date
        if end_date is None:
            end_date = self.max_date
        
        stock_plot = self.make_df(start_date, end_date)

        colors = ['r', 'b', 'g', 'y', 'c', 'm']
        
        for i, stat in enumerate(stats):
            
            stat_min = min(stock_plot[stat])
            stat_max = max(stock_plot[stat])

            stat_avg = np.mean(stock_plot[stat])
            
            date_stat_min = stock_plot[stock_plot[stat] == stat_min]['Date']
            date_stat_min = date_stat_min[date_stat_min.index[0]]
            date_stat_max = stock_plot[stock_plot[stat] == stat_max]['Date']
            date_stat_max = date_stat_max[date_stat_max.index[0]]
            
#             print('Maximum {} = {:.2f} on {}.'.format(stat, stat_max, date_stat_max))
#             print('Minimum {} = {:.2f} on {}.'.format(stat, stat_min, date_stat_min))
#             print('Current {} = {:.2f} on {}.\n'.format(stat, self.stock.loc[self.stock.index[-1], stat], self.max_date))
            
        return stats,stat_max, date_stat_max,stat_min,date_stat_min,self.stock.loc[self.stock.index[-1], stat], self.max_date
            
            
            
#             # Percentage y轴
#             if plot_type == 'pct':
#                 # 简单情节
#                 plt.style.use('fivethirtyeight');
#                 if stat == 'Daily Change':
#                     plt.plot(stock_plot['Date'], 100 * stock_plot[stat],
#                          color = colors[i], linewidth = 2.4, alpha = 0.9,
#                          label = stat)
#                 else:
#                     plt.plot(stock_plot['Date'], 100 * (stock_plot[stat] -  stat_avg) / stat_avg,
#                          color = colors[i], linewidth = 2.4, alpha = 0.9,
#                          label = stat)

#                 plt.xlabel('Date'); plt.ylabel('Change Relative to Average (%)'); plt.title('%s Stock History' % self.symbol); 
#                 plt.legend(prop={'size':10})
#                 plt.grid(color = 'k', alpha = 0.4); 

#             # Stat y轴
#             elif plot_type == 'basic':
#                 plt.style.use('fivethirtyeight');
#                 plt.plot(stock_plot['Date'], stock_plot[stat], color = colors[i], linewidth = 3, label = stat, alpha = 0.8)
#                 plt.xlabel('Date'); plt.ylabel('US $'); plt.title('%s Stock History' % self.symbol); 
#                 plt.legend(prop={'size':10})
#                 plt.grid(color = 'k', alpha = 0.4); 
      
#         plt.show();
        
    #重置绘图参数以清除样式格式
    # 不确定这是否应该是静态方法
    @staticmethod
    def reset_plot():
        
        #恢复默认参数
        matplotlib.rcdefaults()
        
        #调整参数
        matplotlib.rcParams['figure.figsize'] = (8, 5)
        matplotlib.rcParams['axes.labelsize'] = 10
        matplotlib.rcParams['xtick.labelsize'] = 8
        matplotlib.rcParams['ytick.labelsize'] = 8
        matplotlib.rcParams['axes.titlesize'] = 14
        matplotlib.rcParams['text.color'] = 'k'
    
    # 在周末线性插入价格的方法
    def resample(self, dataframe):
        # Change the index and resample at daily level
        dataframe = dataframe.set_index('ds')
        dataframe = dataframe.resample('D')
        
        # 重置索引并插入nan值
        dataframe = dataframe.reset_index(level=0)
        dataframe = dataframe.interpolate()
        return dataframe
    
    #在数据框中删除周末
    def remove_weekends(self, dataframe):
        
        #重置索引以使用ix
        dataframe = dataframe.reset_index(drop=True)
        
        weekends = []
        
        # 查找所有周末
        for i, date in enumerate(dataframe['ds']):
            if (date.weekday()) == 5 | (date.weekday() == 6):
                weekends.append(i)
            
        #舍弃周末
        dataframe = dataframe.drop(weekends, axis=0)
        
        return dataframe
    
    
    #计算并计划在指定日期范围内购买和持有股票的利润
    def buy_and_hold(self, start_date=None, end_date=None, nshares=1):
        self.reset_plot()
        
        start_date, end_date = self.handle_dates(start_date, end_date)
            
        # 查找股票的起始和结束价格
        start_price = float(self.stock[self.stock['Date'] == start_date]['Adj. Open'])
        end_price = float(self.stock[self.stock['Date'] == end_date]['Adj. Close'])
        
        #创建利润数据框并计算利润列
        profits = self.make_df(start_date, end_date)
        profits['hold_profit'] = nshares * (profits['Adj. Close'] - start_price)
        
        #总利润
        total_hold_profit = nshares * (end_price - start_price)
        
#         print('{} Total buy and hold profit from {} to {} for {} shares = ${:.2f}'.format
#               (self.symbol, start_date, end_date, nshares, total_hold_profit))
        
        return self.symbol, start_date, end_date, nshares, total_hold_profit
        
#         #绘图
#         plt.style.use('dark_background')
        
#         # 利润数量的位置
#         text_location = (end_date - pd.DateOffset(months = 1))
        
#         #绘制利润随时间变化的图像
#         plt.plot(profits['Date'], profits['hold_profit'], 'b', linewidth = 3)
#         plt.ylabel('Profit ($)'); plt.xlabel('Date'); plt.title('Buy and Hold Profits for {} {} to {}'.format(
#                                                                 self.symbol, start_date, end_date))
        
#         #在图上显示最终值
#         plt.text(x = text_location, 
#              y =  total_hold_profit + (total_hold_profit / 40),
#              s = '$%d' % total_hold_profit,
#             color = 'g' if total_hold_profit > 0 else 'r',
#             size = 14)
        
#         plt.grid(alpha=0.2)
#         plt.show();
        
    # 创建一个没有训练过的模型
    def create_model(self):

        #制作模型
        model = fbprophet.Prophet(daily_seasonality=self.daily_seasonality,  
                                  weekly_seasonality=self.weekly_seasonality, 
                                  yearly_seasonality=self.yearly_seasonality,
                                  changepoint_prior_scale=self.changepoint_prior_scale,
                                  changepoints=self.changepoints)
        
        if self.monthly_seasonality:
            # 添加每月季节性
            model.add_seasonality(name = 'monthly', period = 30.5, fourier_order = 5)
        
        return model
    
    #绘制改变先前比例变化点（cps）的效果
    def changepoint_prior_analysis(self, changepoint_priors=[0.001, 0.05, 0.1, 0.2], colors=['b', 'r', 'grey', 'gold']):
    
        #使用指定年份的数据进行培训和绘图
        train = self.stock[(self.stock['Date'] > (max(self.stock['Date']) - pd.DateOffset(years=self.training_years)))]
        
        #迭代所有的变更点并制作模型
        for i, prior in enumerate(changepoint_priors):
            # 选择变更点
            self.changepoint_prior_scale = prior
            
            #使用指定的cps创建并训练模型
            model = self.create_model()
            model.fit(train)
            future = model.make_future_dataframe(periods=180, freq='D')
            
            #制作一个数据框来保存预测
            if i == 0:
                predictions = future.copy()
                
            future = model.predict(future)
            
            #填写预测数据框
            predictions['%.3f_yhat_upper' % prior] = future['yhat_upper']
            predictions['%.3f_yhat_lower' % prior] = future['yhat_lower']
            predictions['%.3f_yhat' % prior] = future['yhat']
         
        # 移除周末
        predictions = self.remove_weekends(predictions)
        
        # 绘图设置
        self.reset_plot()
        plt.style.use('fivethirtyeight')
        fig, ax = plt.subplots(1, 1)
        
        # 实际观察
        ax.plot(train['ds'], train['y'], 'ko', ms = 4, label = 'Observations')
        color_dict = {prior: color for prior, color in zip(changepoint_priors, colors)}

        # 绘制每个变更点的预测
        for prior in changepoint_priors:
            # 绘制预测本身
            ax.plot(predictions['ds'], predictions['%.3f_yhat' % prior], linewidth = 1.2,
                     color = color_dict[prior], label = '%.3f prior scale' % prior)
            
            # 不确定区间
            ax.fill_between(predictions['ds'].dt.to_pydatetime(), predictions['%.3f_yhat_upper' % prior],
                            predictions['%.3f_yhat_lower' % prior], facecolor = color_dict[prior],
                            alpha = 0.3, edgecolor = 'k', linewidth = 0.6)
                            
        # 标签
        plt.legend(loc = 2, prop={'size': 10})
        plt.xlabel('Date'); plt.ylabel('Stock Price ($)'); plt.title('Effect of Changepoint Prior Scale');
        plt.show()
            
    # 指定天数的基本模型
    def create_prophet_model(self, days=0, resample=False):
        
        self.reset_plot()
        
        model = self.create_model()
        
        # 适合self.training_years的股票历史记录
        stock_history = self.stock[self.stock['Date'] > (self.max_date - pd.DateOffset(years = self.training_years))]
        
        if resample:
            stock_history = self.resample(stock_history)
        
        model.fit(stock_history)
        
        # 使用未来的数据框制作并预测明年
        future = model.make_future_dataframe(periods = days, freq='D')
        future = model.predict(future)
        
        if days > 0:
            # 打印预测价格
#             print('Predicted Price on {} = ${:.2f}'.format(
#                 future.loc[future.index[-1], 'ds'], future.loc[future.index[-1], 'yhat']))

#             title = '%s Historical and Predicted Stock Price'  % self.symbol
            
            return future.loc[future.index[-1], 'ds'], future.loc[future.index[-1], 'yhat']
        else:
            return None
        
#         #设置图
#         fig, ax = plt.subplots(1, 1)

#         # 实际值
#         ax.plot(stock_history['ds'], stock_history['y'], 'ko-', linewidth = 1.4, alpha = 0.8, ms = 1.8, label = 'Observations')
        
#         # 预测值
#         ax.plot(future['ds'], future['yhat'], 'forestgreen',linewidth = 2.4, label = 'Modeled');

#         #不定区间绘制为色带
#         ax.fill_between(future['ds'].dt.to_pydatetime(), future['yhat_upper'], future['yhat_lower'], alpha = 0.3, 
#                        facecolor = 'g', edgecolor = 'k', linewidth = 1.4, label = 'Confidence Interval')

#         # 格式
#         plt.legend(loc = 2, prop={'size': 10}); plt.xlabel('Date'); plt.ylabel('Price $');
#         plt.grid(linewidth=0.6, alpha = 0.6)
#         plt.title(title);
#         plt.show()
        
#         return model, future
      
    #评估预测模型一年
    def evaluate_prediction(self, start_date=None, end_date=None, nshares = None):
        
        # 默认开始日期比数据的结束日期早一年，结束日期是数据的结束日期
        if start_date is None:
            start_date = self.max_date - pd.DateOffset(years=1)
        if end_date is None:
            end_date = self.max_date
            
        start_date, end_date = self.handle_dates(start_date, end_date)
        
        # 培训数据在开始日期之前的几年开始self.training_years，并开始上升到开始日期
        train = self.stock[(self.stock['Date'] < start_date) & 
                           (self.stock['Date'] > (start_date - pd.DateOffset(years=self.training_years)))]
        
        #测试数据在范围内指定
        test = self.stock[(self.stock['Date'] >= start_date) & (self.stock['Date'] <= end_date)]
        
        #创建并训练模型
        model = self.create_model()
        model.fit(train)
        
        #预测数据额的数据框和预测
        future = model.make_future_dataframe(periods = 365, freq='D')
        future = model.predict(future)
        
        #使用已知值合并预测
        test = pd.merge(test, future, on = 'ds', how = 'inner')

        train = pd.merge(train, future, on = 'ds', how = 'inner')
        
        #计算连续测量之间的差异
        test['pred_diff'] = test['yhat'].diff()
        test['real_diff'] = test['y'].diff()

        #当我们预测正确时，显示正确
        test['correct'] = (np.sign(test['pred_diff'][1:]) == np.sign(test['real_diff'][1:])) * 1
        
        # 预测增加或者减少的准确性
        increase_accuracy = 100 * np.mean(test[test['pred_diff'] > 0]['correct'])
        decrease_accuracy = 100 * np.mean(test[test['pred_diff'] < 0]['correct'])

        # 平均绝对误差
        test_errors = abs(test['y'] - test['yhat'])
        test_mean_error = np.mean(test_errors)

        train_errors = abs(train['y'] - train['yhat'])
        train_mean_error = np.mean(train_errors)

        #计算预测范围内实际值的时间百分比
        test['in_range'] = False

        for i in test.index:
            if (test.loc[i, 'y'] < test.loc[i, 'yhat_upper']) & (test.loc[i, 'y'] > test.loc[i, 'yhat_lower']):
                test.loc[i, 'in_range'] = True

        in_range_accuracy = 100 * np.mean(test['in_range'])

        if not nshares:

            # 日期预测范围
            print('\n Prediction Range: {} to {}.'.format(start_date,
                end_date))

            # 最终预测和实际值
            print('\n Predicted price on {} = ${:.2f}.'.format(max(future['ds']), future.loc[future.index[-1], 'yhat']))
            print('Actual price on    {} = ${:.2f}.\n'.format(max(test['ds']), test.loc[test.index[-1], 'y']))

            print('Average Absolute Error on Training Data = ${:.2f}.'.format(train_mean_error))
            print('Average Absolute Error on Testing  Data = ${:.2f}.\n'.format(test_mean_error))

            #精度
            print('When the model predicted an increase, the price increased {:.2f}% of the time.'.format(increase_accuracy))
            print('When the model predicted a  decrease, the price decreased  {:.2f}% of the time.\n'.format(decrease_accuracy))

            print('The actual value was within the {:d}% confidence interval {:.2f}% of the time.'.format(int(100 * model.interval_width), in_range_accuracy))


             #重置
            self.reset_plot()
            
            #设置
            fig, ax = plt.subplots(1, 1)

            #实际值
            ax.plot(train['ds'], train['y'], 'ko-', linewidth = 1.4, alpha = 0.8, ms = 1.8, label = 'Observations')
            ax.plot(test['ds'], test['y'], 'ko-', linewidth = 1.4, alpha = 0.8, ms = 1.8, label = 'Observations')
            
            #预测值
            ax.plot(future['ds'], future['yhat'], 'navy', linewidth = 2.4, label = 'Predicted');

            #不确定区域为灰色地带
            ax.fill_between(future['ds'].dt.to_pydatetime(), future['yhat_upper'], future['yhat_lower'], alpha = 0.6, 
                           facecolor = 'gold', edgecolor = 'k', linewidth = 1.4, label = 'Confidence Interval')

            # 在预测开始处放置一条垂直线
            plt.vlines(x=min(test['ds']), ymin=min(future['yhat_lower']), ymax=max(future['yhat_upper']), colors = 'r',
                       linestyles='dashed', label = 'Prediction Start')

            # 格式
            plt.legend(loc = 2, prop={'size': 8}); plt.xlabel('Date'); plt.ylabel('Price $');
            plt.grid(linewidth=0.6, alpha = 0.6)
                       
            plt.title('{} Model Evaluation from {} to {}.'.format(self.symbol,
                start_date, end_date));
            plt.show();

        
        # 指定了多个份额
        elif nshares:
            
            #只有这个股票会增长的时候我们会炒股
            test_pred_increase = test[test['pred_diff'] > 0]
            
            test_pred_increase.reset_index(inplace=True)
            prediction_profit = []
            
            #迭代所有的预测并且计算炒股的利润
            for i, correct in enumerate(test_pred_increase['correct']):
                
                #如果我们预测到并且价格上涨，我们就会获得差额
                if correct == 1:
                    prediction_profit.append(nshares * test_pred_increase.loc[i, 'real_diff'])
                # 如果我们预测到价格下降，我们会失去差额
                else:
                    prediction_profit.append(nshares * test_pred_increase.loc[i, 'real_diff'])
            
            test_pred_increase['pred_profit'] = prediction_profit
            
            # 把利润导入测试数据框中
            test = pd.merge(test, test_pred_increase[['ds', 'pred_profit']], on = 'ds', how = 'left')
            test.loc[0, 'pred_profit'] = 0
        
            #任何一种方法在任何日期的利润
            test['pred_profit'] = test['pred_profit'].cumsum().ffill()
            test['hold_profit'] = nshares * (test['y'] - float(test.loc[0, 'y']))
            
            #显示信息
            print('You played the stock market in {} from {} to {} with {} shares.\n'.format(
                self.symbol, start_date, end_date, nshares))
            
            print('When the model predicted an increase, the price increased {:.2f}% of the time.'.format(increase_accuracy))
            print('When the model predicted a  decrease, the price decreased  {:.2f}% of the time.\n'.format(decrease_accuracy))

            # 显示有关股票市场风险的一些良性信息
            print('The total profit using the Prophet model = ${:.2f}.'.format(np.sum(prediction_profit)))
            print('The Buy and Hold strategy profit =         ${:.2f}.'.format(float(test.loc[test.index[-1], 'hold_profit'])))
            print('\nThanks for playing the stock market!\n')
            
           
            
            #绘制预测和实际利润随时间变化的情况
            self.reset_plot()
            
            #最终的利润和最终的智能用于定位文本
            final_profit = test.loc[test.index[-1], 'pred_profit']
            final_smart = test.loc[test.index[-1], 'hold_profit']

            #文本
            last_date = test.loc[test.index[-1], 'ds']
            text_location = (last_date - pd.DateOffset(months = 1))

            plt.style.use('dark_background')

            #智能利润
            plt.plot(test['ds'], test['hold_profit'], 'b',
                     linewidth = 1.8, label = 'Buy and Hold Strategy') 

            #预测利润
            plt.plot(test['ds'], test['pred_profit'], 
                     color = 'g' if final_profit > 0 else 'r',
                     linewidth = 1.8, label = 'Prediction Strategy')

            # 显示最终值
            plt.text(x = text_location, 
                     y =  final_profit + (final_profit / 40),
                     s = '$%d' % final_profit,
                    color = 'g' if final_profit > 0 else 'r',
                    size = 18)
            
            plt.text(x = text_location, 
                     y =  final_smart + (final_smart / 40),
                     s = '$%d' % final_smart,
                    color = 'g' if final_smart > 0 else 'r',
                    size = 18);

            plt.ylabel('Profit  (US $)'); plt.xlabel('Date'); 
            plt.title('Predicted versus Buy and Hold Profits');
            plt.legend(loc = 2, prop={'size': 10});
            plt.grid(alpha=0.2); 
            plt.show()
        
    def retrieve_google_trends(self, search, date_range):
        
        # 设置趋势提取对象
        pytrends = TrendReq(hl='en-US', tz=360)
        kw_list = [search]

        try:
        
            # 创建搜索对象
            pytrends.build_payload(kw_list, cat=0, timeframe=date_range[0], geo='', gprop='news')
            
            #随时检索
            trends = pytrends.interest_over_time()

            related_queries = pytrends.related_queries()

        except Exception as e:
            print('\nGoogle Search Trend retrieval failed.')
            print(e)
            return
        
        return trends, related_queries
        
    def changepoint_date_analysis(self, search=None):
        self.reset_plot()

        model = self.create_model()
        
        #使用过去self.training_years年的数据
        train = self.stock[self.stock['Date'] > (self.max_date - pd.DateOffset(years = self.training_years))]
        model.fit(train)
        
        #训练数据的预测（没有未来时期）
        future = model.make_future_dataframe(periods=0, freq='D')
        future = model.predict(future)
    
        train = pd.merge(train, future[['ds', 'yhat']], on = 'ds', how = 'inner')
        
        changepoints = model.changepoints
        train = train.reset_index(drop=True)
        
        # 仅创建变更点的数据框
        change_indices = []
        for changepoint in (changepoints):
            change_indices.append(train[train['ds'] == changepoint].index[0])
        
        c_data = train.loc[change_indices, :]
        deltas = model.params['delta'][0]
        
        c_data['delta'] = deltas
        c_data['abs_delta'] = abs(c_data['delta'])
        
        # 按最大变化对值进行排序
        c_data = c_data.sort_values(by='abs_delta', ascending=False)

        #限制为10个最大的变更点
        c_data = c_data[:10]

        # 分为负面和正面变化点
        cpos_data = c_data[c_data['delta'] > 0]
        cneg_data = c_data[c_data['delta'] < 0]

        # 变更点和数据
        if not search:
        
#             print('\nChangepoints sorted by slope rate of change (2nd derivative):\n')
#             print(c_data.loc[:, ['Date', 'Adj. Close', 'delta']][:5])
            return c_data.loc[:, ['Date', 'Adj. Close', 'delta']][:5]

            # 线图显示实际值，估计值和变更点
            self.reset_plot()
            
#             # 设置线图
#             plt.plot(train['ds'], train['y'], 'ko', ms = 4, label = 'Stock Price')
#             plt.plot(future['ds'], future['yhat'], color = 'navy', linewidth = 2.0, label = 'Modeled')
            
#             # 将点更改为垂直线
#             plt.vlines(cpos_data['ds'].dt.to_pydatetime(), ymin = min(train['y']), ymax = max(train['y']), 
#                        linestyles='dashed', color = 'r', 
#                        linewidth= 1.2, label='Negative Changepoints')

#             plt.vlines(cneg_data['ds'].dt.to_pydatetime(), ymin = min(train['y']), ymax = max(train['y']), 
#                        linestyles='dashed', color = 'darkgreen', 
#                        linewidth= 1.2, label='Positive Changepoints')

#             plt.legend(prop={'size':10});
#             plt.xlabel('Date'); plt.ylabel('Price ($)'); plt.title('Stock Price with Changepoints')
#             plt.show()
        
        #在Google新闻中搜索搜索字词,显示相关查询，上升相关查询,图形变化点，搜索频率，股票价格
        if search:
            date_range = ['%s %s' % (str(min(train['Date'])), str(max(train['Date'])))]

            #获取指定字词的Google趋势并加入培训数据框
            trends, related_queries = self.retrieve_google_trends(search, date_range)

            if (trends is None)  or (related_queries is None):
                print('No search trends found for %s' % search)
                return

#             print('\n Top Related Queries: \n')
#             print(related_queries[search]['top'].head())

#             print('\n Rising Related Queries: \n')
#             print(related_queries[search]['rising'].head())
            
            return related_queries[search]['top'].head(),related_queries[search]['rising'].head()

            #上传用于加入训练数据的数据
            trends = trends.resample('D')

            trends = trends.reset_index(level=0)
            trends = trends.rename(columns={'date': 'ds', search: 'freq'})

            # 插值频率
            trends['freq'] = trends['freq'].interpolate()

            #合并训练数据
            train = pd.merge(train, trends, on = 'ds', how = 'inner')

            # 标准化
            train['y_norm'] = train['y'] / max(train['y'])
            train['freq_norm'] = train['freq'] / max(train['freq'])
            
            self.reset_plot()

#             #绘制标准化股票价格并规范搜索频率
#             plt.plot(train['ds'], train['y_norm'], 'k-', label = 'Stock Price')
#             plt.plot(train['ds'], train['freq_norm'], color='goldenrod', label = 'Search Frequency')

#             # 将点更改为垂直线
#             plt.vlines(cpos_data['ds'].dt.to_pydatetime(), ymin = 0, ymax = 1, 
#                        linestyles='dashed', color = 'r', 
#                        linewidth= 1.2, label='Negative Changepoints')

#             plt.vlines(cneg_data['ds'].dt.to_pydatetime(), ymin = 0, ymax = 1, 
#                        linestyles='dashed', color = 'darkgreen', 
#                        linewidth= 1.2, label='Positive Changepoints')

#             plt.legend(prop={'size': 10})
#             plt.xlabel('Date'); plt.ylabel('Normalized Values'); plt.title('%s Stock Price and Search Frequency for %s' % (self.symbol, search))
#             plt.show()
        
    # 预测给定天数的未来价格
    def predict_future(self, days=30):
        
        #使用过去的self.training_years年份进行培训
        train = self.stock[self.stock['Date'] > (max(self.stock['Date']) - pd.DateOffset(years=self.training_years))]
        
        model = self.create_model()
        
        model.fit(train)
        
        # 具有指定预测天数的未来数据帧
        future = model.make_future_dataframe(periods=days, freq='D')
        future = model.predict(future)
        
        # 仅关注未来日期
        future = future[future['ds'] >= max(self.stock['Date'])]
        
        #移出周末
        future = self.remove_weekends(future)
        
        #计算是否增加
        future['diff'] = future['yhat'].diff()
    
        future = future.dropna()

        # 找到预测方向并创建单独的数据帧
        future['direction'] = (future['diff'] > 0) * 1
        
        #重命名列以进行演示
        future = future.rename(columns={'ds': 'Date', 'yhat': 'estimate', 'diff': 'change', 
                                        'yhat_upper': 'upper', 'yhat_lower': 'lower'})
        
        future_increase = future[future['direction'] == 1]
        future_decrease = future[future['direction'] == 0]
        
        # 打印出日期
        print('\nPredicted Increase: \n')
        print(future_increase[['Date', 'estimate', 'change', 'upper', 'lower']])
        
        print('\nPredicted Decrease: \n')
        print(future_decrease[['Date', 'estimate', 'change', 'upper', 'lower']])
        
        self.reset_plot()
        
        plt.style.use('fivethirtyeight')
        matplotlib.rcParams['axes.labelsize'] = 10
        matplotlib.rcParams['xtick.labelsize'] = 8
        matplotlib.rcParams['ytick.labelsize'] = 8
        matplotlib.rcParams['axes.titlesize'] = 12
        
        # 绘制预测并指出是增加还是减少
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # 绘制估算值
        ax.plot(future_increase['Date'], future_increase['estimate'], 'g^', ms = 12, label = 'Pred. Increase')
        ax.plot(future_decrease['Date'], future_decrease['estimate'], 'rv', ms = 12, label = 'Pred. Decrease')

        #errorbars
        ax.errorbar(future['Date'].dt.to_pydatetime(), future['estimate'], 
                    yerr = future['upper'] - future['lower'], 
                    capthick=1.4, color = 'k',linewidth = 2,
                   ecolor='darkblue', capsize = 4, elinewidth = 1, label = 'Pred with Range')

        plt.legend(loc = 2, prop={'size': 10});
        plt.xticks(rotation = '45')
        plt.ylabel('Predicted Stock Price (US $)');
        plt.xlabel('Date'); plt.title('Predictions for %s' % self.symbol);
        plt.show()
        
    def changepoint_prior_validation(self, start_date=None, end_date=None,changepoint_priors = [0.001, 0.05, 0.1, 0.2]):


        # 默认开始日期是数据结束前两年，结束日期是数据结束前一年
        if start_date is None:
            start_date = self.max_date - pd.DateOffset(years=2)
        if end_date is None:
            end_date = self.max_date - pd.DateOffset(years=1)
            
        # 转换为pandas datetime以索引数据帧
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        start_date, end_date = self.handle_dates(start_date, end_date)
                               
        # 选择self.training_years年数
        train = self.stock[(self.stock['Date'] > (start_date - pd.DateOffset(years=self.training_years))) & 
        (self.stock['Date'] < start_date)]
        
        # 测试数据由范围指定
        test = self.stock[(self.stock['Date'] >= start_date) & (self.stock['Date'] <= end_date)]

        eval_days = (max(test['Date']) - min(test['Date'])).days
        
        results = pd.DataFrame(0, index = list(range(len(changepoint_priors))), 
            columns = ['cps', 'train_err', 'train_range', 'test_err', 'test_range'])

        print('\nValidation Range {} to {}.\n'.format(min(test['Date']),
            max(test['Date'])))
            
        
        # 迭代所有变更点并制作模型
        for i, prior in enumerate(changepoint_priors):
            results.loc[i, 'cps'] = prior
            
            #变更点
            self.changepoint_prior_scale = prior
            
            # 使用指定的cps创建并训练模型
            model = self.create_model()
            model.fit(train)
            future = model.make_future_dataframe(periods=eval_days, freq='D')
                
            future = model.predict(future)
            
            # 培训结果和指标
            train_results = pd.merge(train, future[['ds', 'yhat', 'yhat_upper', 'yhat_lower']], on = 'ds', how = 'inner')
            avg_train_error = np.mean(abs(train_results['y'] - train_results['yhat']))
            avg_train_uncertainty = np.mean(abs(train_results['yhat_upper'] - train_results['yhat_lower']))
            
            results.loc[i, 'train_err'] = avg_train_error
            results.loc[i, 'train_range'] = avg_train_uncertainty
            
            #测试结果和指标
            test_results = pd.merge(test, future[['ds', 'yhat', 'yhat_upper', 'yhat_lower']], on = 'ds', how = 'inner')
            avg_test_error = np.mean(abs(test_results['y'] - test_results['yhat']))
            avg_test_uncertainty = np.mean(abs(test_results['yhat_upper'] - test_results['yhat_lower']))
            
            results.loc[i, 'test_err'] = avg_test_error
            results.loc[i, 'test_range'] = avg_test_uncertainty

        print(results)


        
        #培训和测试平均错误
        self.reset_plot()
        
        plt.plot(results['cps'], results['train_err'], 'bo-', ms = 8, label = 'Train Error')
        plt.plot(results['cps'], results['test_err'], 'r*-', ms = 8, label = 'Test Error')
        plt.xlabel('Changepoint Prior Scale'); plt.ylabel('Avg. Absolute Error ($)');
        plt.title('Training and Testing Curves as Function of CPS')
        plt.grid(color='k', alpha=0.3)
        plt.xticks(results['cps'], results['cps'])
        plt.legend(prop={'size':10})
        plt.show();
        
        #培训和测试平均不确定性
        self.reset_plot()

        plt.plot(results['cps'], results['train_range'], 'bo-', ms = 8, label = 'Train Range')
        plt.plot(results['cps'], results['test_range'], 'r*-', ms = 8, label = 'Test Range')
        plt.xlabel('Changepoint Prior Scale'); plt.ylabel('Avg. Uncertainty ($)');
        plt.title('Uncertainty in Estimate as Function of CPS')
        plt.grid(color='k', alpha=0.3)
        plt.xticks(results['cps'], results['cps'])
        plt.legend(prop={'size':10})
        plt.show();

