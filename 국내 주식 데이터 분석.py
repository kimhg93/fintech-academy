#kaggle
import pandas as pd

kor_ticker = pd.read_csv('./kor_ticker.csv')
kor_sector = pd.read_csv('./kor_sector.csv')

# 국내 종목 티커의 경우 6자리지만 int 타입으로 제공된 데이터는 앞에 0을 채워 6자리로 만들어야 함.
kor_ticker['종목코드'] = kor_ticker['종목코드'].str.zfill(6)
kor_sector['CMP_CD'] = kor_sector['CMP_CD'].astype(str).str.zfill(6)

# 컬럼명이 영문, 한글로 다르기 때문에 일치 시켜 join 한다.
data_market = kor_ticker.merge(kor_sector,
                               left_on = ['종목코드', '종목명', '기준일'],
                               right_on = ['CMP_CD', 'CMP_KOR', '기준일'],
                               how = 'left')

# 섹터의 종류
print(list(data_market['SEC_NM_KOR'].unique()))

# 섹터별 종목 개수
print(data_market['SEC_NM_KOR'].value_counts())

# 섹터별 비중
print(data_market['SEC_NM_KOR'].value_counts(normalize=True))

# 종목명 (특정 컬럼의 데이터)
print(data_market['종목명'])

# 컬럼명이 "시" 로 시작하는 컬럼의 데이터
print(data_market.loc[:, data_market.columns.str.startswith('시')])

# 컬럼명이 "S"로 끝나는 컬럼의 데이터
print(data_market.loc[:, data_market.columns.str.endswith('S')])

# 컬럼명에 "p"가 포함된 컬럼의 데이터
print(data_market.loc[:, data_market.columns.str.contains('P')])

# 필요한 컬럼을 만들어 저장
data_market['PBR'] = data_market['종가'] / data_market['BPS']
data_market['PER'] = data_market['종가'] / data_market['EPS']
data_market['ROE'] = data_market['PBR'] / data_market['PER']

print(data_market)

# pbr이 1보다 작은거
print(data_market.loc[data_market['PBR'] < 1, ['종목명', 'PBR']])
# 다중 조건
print(data_market.loc[(data_market['PBR'] < 1) & (data_market['ROE'] > 0.2) , ['종목명', 'PBR', 'ROE']])
#pbr 최소 및 최대
print(data_market['PBR'].agg(['min', 'max']))
#pbr 기준 정렬
print(data_market[['종목명', 'PBR']].sort_values(['PBR']))
# 섹터별 pbr 평균
print(data_market.groupby(['SEC_NM_KOR'])['PBR'].agg('median').sort_values())
# 시장, 섹터별 pbr 평균
group_pbr = data_market.groupby(['시장구분', 'SEC_NM_KOR'])['PBR'].agg('median').sort_values().reset_index()
print(group_pbr)
# 피벗
print(group_pbr.pivot_table(index = 'SEC_NM_KOR', columns = '시장구분', values = 'PBR'))



# print(kor_ticker)
# print(kor_sector)


