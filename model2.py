# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:41:07 2023

@author: Chenhao Sun
"""

import scipy.optimize as so
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

df = pd.read_excel('Price Data TIPS.xlsx')

def YTM(PV, k, d, n, M, TS, t, r, T, MI):
 def f(y):
  coupon=[]
  for i in np.arange(0, n):
            coupon.append(MI * T * r / pow(1 + y / k, d / TS + i))
  return np.sum(coupon) + M / pow(1 + y / k, d / TS + n - 1) - PV

 return so.fsolve(f, 0)

k = 2 # Frequency of coupon payments per year
m = 100 # Face value
r = 0.00125 / 2 # Interest rate
Mi = [m * 1.00453, m * 1.01529, m * 1.05790, m * 1.0930925, 
      m * 1.1468366666666667, m * 1.1512366666666667, m * 1.1858033333333333] # List of 'Face Value * Index Ratio' on each Issue Date

coupon_dates = ['10/15/2020', '04/15/2021', '10/15/2021', '04/15/2022', '10/15/2022', '04/15/2023', '10/15/2023', '04/15/2024', '10/15/2024', '04/15/2025']
issue_date = datetime.strptime('04/15/2020', '%m/%d/%Y')

estimate_ytms = []
for date in df['Date']:
    for a, coupon_date in enumerate(coupon_dates):
        coupon_date = datetime.strptime(coupon_date, '%m/%d/%Y')
        next_coupon_date = datetime.strptime(coupon_dates[a + 1], '%m/%d/%Y') if a < len(coupon_dates) - 1 else None

        if date < coupon_date or (next_coupon_date and date >= coupon_date and date < next_coupon_date):
            if date < coupon_date:
                t = (date - issue_date).days # Time periods
                TS = (coupon_date - issue_date).days # Time from the last coupon to the next
                n = 10 # Number of remaining coupons
            else:
                t = (date - coupon_date).days # Time periods
                if next_coupon_date:
                    TS = (next_coupon_date - coupon_date).days # Time from the last coupon to the next
                n = 10 - a # Number of remaining coupons
            d = (coupon_date - date).days # Time to next coupon
            MI = Mi[10 - n] # Choose MI for each period in list Mi
            break
    
    T = TS / 365
    I = df[df['Date'] == date]['Index Ratio'].values # Index Ratio
    M = m * I # 'Face Value * Index Ratio' on each day
    PV = df[df['Date'] == date]['Last Price Clean Data'].values # Present value
    Bond_yield = YTM(PV, k, d, n, M, TS, t, r, T, MI)[0]
    estimate_ytms.append(np.round(Bond_yield, 10))

df['Estimate_YTM'] = estimate_ytms
df['Error'] = df['Estimate_YTM'] - df['Yield']
print(df)
df.to_excel('model2.xlsx', index = False)
        
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Estimate_YTM'], label='Estimate YTM', linestyle='-', color='b')
plt.plot(df['Date'], df['Yield'], label='Actual YTM', linestyle='-', color='r')
plt.xlabel('Date')
plt.ylabel('YTM')
plt.title('Estimate YTM vs. Actual YTM')
plt.legend()
plt.xticks(rotation = 45)
plt.tight_layout()

plt.show()