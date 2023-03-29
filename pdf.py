import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

def geodistance(lng1, lat1, lng2, lat2):
    '公式计算两点间距离（m）'
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 经纬度转换成弧度
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
    distance = round(distance / 1000, 3)  # 得到单位km
    return distance

def cal_h(region_df):
    city_ind_h = {} #计算每个产业的最优的h值和距离对
    # n = region_df.shape[0]
    city_ind_dis = {}
    citylist = list(region_df['citycode'].unique())
    indlist = list(region_df['CIC3_2002'].unique())
    error_city_ind_list = []
    for city in citylist:
        for ind in indlist:
            try:
                one_city_ind_df = region_df[(region_df['CIC3_2002'] == ind) & (region_df['citycode'] == city)]
                one_city_ind_df = one_city_ind_df.reset_index(drop=True)
                n = one_city_ind_df.shape[0]
                city_ind_dis_list = []
                for i in range(len(one_city_ind_df)-1):
                    lng1 = one_city_ind_df.loc[i,'WGS_lon']
                    lat1 = one_city_ind_df.loc[i,'WGS_lat']
                    for j in range(i+1,len(one_city_ind_df)):
                        lng2 = one_city_ind_df.loc[j, 'WGS_lon']
                        lat2 = one_city_ind_df.loc[j, 'WGS_lat']
                        dis = geodistance(lng1, lat1, lng2, lat2)
                        city_ind_dis_list.append(dis)
                city_ind_dis_list = np.array(city_ind_dis_list)
                h = 0.04 * city_ind_dis_list.std() * ((4/3*n) ** 0.2)
                city_ind_h[city,ind] = h
                # print(city_ind_h[city,ind])
            except Exception as error:
                error_city_ind_list.append((city,ind))
    return city_ind_h,error_city_ind_list

def cal_k(h, df):
    '计算k值'
    d = np.linspace(0.5,100,200).T
    k = np.linspace(0,0,200).T
    df = df.reset_index(drop=True)
    n = df.shape[0]
    employ_sum = 0
    for i in range(1, n-1):
        employ1 = df.loc[i, 'staff']
        lng1 = df.loc[i, 'WGS_lon']
        lat1 = df.loc[i, 'WGS_lat']
        for j in range(i+1, n):
            employ2 = df.loc[j, 'staff']
            lng2 = df.loc[j, 'WGS_lon']
            lat2 = df.loc[j, 'WGS_lat']
            employ_sum = employ_sum + (employ1 + employ2)
            try:
                dis = geodistance(lng1, lat1, lng2, lat2)
                k_hat = (employ1 + employ2) * (np.exp(-np.square(d - dis)/2*(h**2)) + np.exp(-np.square(d + dis)/2*(h**2)))
                k = k + k_hat
            except Exception as error:
                print(error)
    k = k/(h * employ_sum)

    return k


if __name__ == '__main__':
    file_path = r'E:\数据\DO\DO指数测算数据集\DO少量指标\DO07.csv'
    df = pd.read_csv(file_path)
    city_list = list(df['citycode'].unique())
    ind_list = list(df['CIC3_2002'].unique())
    d,e = cal_h(df)
    pdf = {}
    for city in city_list:
        for ind in ind_list:
            city_ind_df = df[(df['CIC3_2002'] == ind) & (df['citycode'] == city)]
            city_ind_df = city_ind_df.reset_index(drop=True)
            k = cal_k(d[city,ind],city_ind_df)
            pdf[city, ind] = np.sum(k[:60]) / np.sum(k)
            print(pdf[city,ind])

    df = pd.DataFrame(pdf.items(), columns=['key', 'value'])
    df[['city', 'ind']] = pd.DataFrame(df['key'].tolist(), index=df.index)
    df[['city', 'ind', 'value']].to_excel('PDF07.xlsx', index=False)














