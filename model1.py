# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 19:29:04 2023

@author: Chenhao Sun
"""

import scipy.optimize as so
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

df = pd.read_excel('Data (Shared).xlsx')

def YTM(PV, C, k, d, n, M, TS, t, AI):
 def f(y):
  coupon=[]
  for i in np.arange(0, n):
            coupon.append((C / k) / pow(1 + y / k, d / TS + i))
  return np.sum(coupon) + M / pow(1 + y / k, d / TS + n - 1) - PV - AI
 return so.fsolve(f, 0)

k = 2 # Frequency of coupon payments per year
M = 100 # Face value
C = 0.125 # Annual coupon rate

coupon_dates = ['04/30/2021', '10/31/2021', '04/30/2022', '10/31/2022']
issue_date = datetime.strptime('10/31/2020', '%m/%d/%Y')

estimate_ytms = []
for date in df['Date']:
    for i, coupon_date in enumerate(coupon_dates):
        coupon_date = datetime.strptime(coupon_date, '%m/%d/%Y')
        next_coupon_date = datetime.strptime(coupon_dates[i + 1], '%m/%d/%Y') if i <= len(coupon_dates) - 1 else None

        if date < coupon_date or (next_coupon_date and date >= coupon_date and date < next_coupon_date):
            if date < coupon_date:
                t = (date - issue_date).days # Time periods
                TS = (coupon_date - issue_date).days # Time from the last coupon to the next
                n = 4 # Number of remaining coupons
            else:
                t = (date - coupon_date).days # Time periods
                if next_coupon_date:
                    TS = (next_coupon_date - coupon_date).days # Time from the last coupon to the next
                n = 4 - i # Number of remaining coupons
            d = (coupon_date - date).days # Time to next coupon
            AI = C * t / k / TS # Accrued Interest
            break
    
    PV = df[df['Date'] == date]['Prices'].values # Present value
    Bond_yield = YTM(PV, C, k, d, n, M, TS, t, AI)[0]
    estimate_ytms.append(np.round(Bond_yield, 10))

df['Estimate_YTM'] = estimate_ytms
        
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Estimate_YTM'], label='Estimate YTM', linestyle='-', color='b')
plt.plot(df['Date'], df['YTM (in %)'], label='Actual YTM', linestyle='-', color='r')
plt.xlabel('Date')
plt.ylabel('YTM')
plt.title('Estimate YTM vs. Actual YTM')
plt.legend()
plt.xticks(rotation = 45)
plt.tight_layout()

plt.show()