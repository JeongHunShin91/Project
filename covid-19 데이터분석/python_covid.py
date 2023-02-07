# 0920_covid SJH
import numpy as np
import pandas as pd

text = pd.read_csv(r'C:\Users\tj-bu\PycharmProjects\pythonProject\csse_covid_19_daily_reports\12-31-2020.csv')
textt = pd.read_csv(r'C:\Users\tj-bu\PycharmProjects\pythonProject\csse_covid_19_daily_reports\12-31-2021.csv')
text1 = text.groupby('Country_Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum()
textt1 = textt.groupby('Country_Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum()
text1  # 12-31-2020 데이터
textt1  # 12-31-2021 데이터

# 1)
# 일별 국가별 코로나 발생자수와 사망자 수를 기준으로 전처리 하시오. 일부
# 국가는 지역별로 코로나 발생자수와 사망자 수가 분리되어 있으니 국가별로
# 집계하고 국가, 총발생자수, 총사망자수, 일평균 발생자수, 일평균 사망자수 리
# 스트를 제시하시오.
textt2 = textt1[['Confirmed']]  # 12-31-2021월 총발생자수
textt3 = textt1[['Deaths']]  # 12-31-2021월 총사망자수
text2 = text1[['Confirmed']]  # 12-31-2020 총발생자수
text2
text3 = text1[['Deaths']]  # 12-31-2020 총사망자수
text3
textt4 = (textt2 - text2) // 365  # 일평균 발생자수
textt4.columns = ['Confirmedmean']
textt4
textt5 = (textt3 - text3) // 365  # 일평균 사망자수
textt5.columns = ['Deathsmean']
textt5

# 2)
# 데이터가 0인 경우(코로나 환자 0)와 데이터가 없는 경우를 구분하여 전처
# 리하고 전처리 시 data가 없는 국가는 제외하고 제외된 국가 리스트를 제시하
# 시오.
full = pd.concat([textt2, textt3, textt4, textt5], axis=1)
zerodata = full[(full['Confirmed']==0)]
zerodata
nandata = full.dropna()
nandata

# 3 총발생자수, 총사망수, 일평균 발생자수, 일평균 사방자수
full = pd.concat([textt2, textt3, textt4, textt5], axis=1)  # 총 데이터합
full
full1 = full.sort_values(['Confirmed'], ascending=False).head(20)  # 총발생자수 내림차순추출
Confirmedlist = full1.sort_values(['Confirmed'])  # 총발생자수 20안에 오름차순
print(Confirmedlist)
full1 = full.sort_values(['Deaths'], ascending=False).head(20)  # 총사망자수 내림차순추출
Deathslist = full1.sort_values(['Deaths'])
print(Deathslist)
full1 = full.sort_values(['Confirmedmean'], ascending=False).head(20)  # 일평균 발생자수 내림차순추출
Confirmedmeanlist = full1.sort_values(['Confirmedmean'])
print(Confirmedmeanlist)
full1 = full.sort_values(['Deathsmean'], ascending=False).head(20)  # 일평균 사망자수 내림차순추출
Deathsmeanlist = full1.sort_values(['Deathsmean'])
print(Deathsmeanlist)

# 4 한국꺼만 추출
full.loc[['Korea, South'], :]

