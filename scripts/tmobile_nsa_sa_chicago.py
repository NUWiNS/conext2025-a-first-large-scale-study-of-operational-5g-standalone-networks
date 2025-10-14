import re
import sys
import os
import glob
import numpy as np 
import pandas as pd 
import pickle as pkl 
import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
import random 
import datetime
import xyzservices as xyz
import geopy.distance
from pprint import pprint
from geopy.distance import distance
from geopy.distance import geodesic
from shapely.geometry import Point
from collections import Counter
from collections import defaultdict
from earfcn.convert import earfcn2freq, earfcn2band, freq2earfcn

def check_distance_feasible(prev_lat_lon, current_lat_lon, prev_ts, current_ts):
    time_diff = abs(current_ts - prev_ts)
    if time_diff > 2:
        return 0
    distance = geopy.distance.geodesic((prev_lat_lon[0], prev_lat_lon[1]), (current_lat_lon[0], current_lat_lon[1])).miles

    if time_diff == 0:
        # since these are 5G metrics, then probably they are around 100 to 200 ms apart
        # choose a random number between 0.1 and 0.2
        time_diff = random.uniform(0.1, 0.2)
    # assume max of 130 miles/hr 
    max_distance = (130 / 3600) * time_diff
    max_distance = 0.1
    # if distance > max_distance:
    #     return 0
    # else:
    return distance

def calculate_percentage_of_occurrence(lst):
    total_elements = len(lst)
    
    # Use Counter to count the occurrence of each element
    element_counts = Counter(lst)

    # Calculate the percentage for each unique element
    percentage_dict = {element: round((count / total_elements) * 100, 2) for element, count in element_counts.items()}

    return percentage_dict

def datetime_to_timestamp(datetime_str):
    int(datetime_str.astimezone(datetime.timezone.utc).timestamp())
    return datetime_str.astimezone(datetime.timezone.utc).timestamp()


def get_iperf_data_length(data):
    count = 0
    for d in data:
        if '[  5] ' in d and 'sec' in d and 'sender' not in d and 'receiver' not in d:
            count+=1 
    return count

def get_icmp_ping_data_length(data):
    count = 0
    for d in data:
        if 'icmp_seq=' in d and 'ttl=' in d:
            count+=1 
    return count


tmobile_nsa_app_dir = "../raw_data/small_study_chic_bos/chicago/tmobile_nsa_chicago_drive"
tmobile_sa_app_dir = "../raw_data/small_study_chic_bos/chicago/tmobile_sa_chicago_drive"

tmobile_nsa_xcal_file = "../raw_data/small_study_chic_bos/chicago/tmobile_nsa_chicago_drive/TMOBILE_NSA_100MS.xlsx"
tmobile_sa_xcal_file = "../raw_data/small_study_chic_bos/chicago/tmobile_sa_chicago_drive/TMOBILE_SA_100MS.xlsx"

# NSA
if 1:
    print("NSA")
    dl_unix_ts_list = []
    ul_unix_ts_list = []
    rtt_unix_ts_list = []

    for day in glob.glob(tmobile_nsa_app_dir + "/*"):
        for folder in glob.glob(day + "/*"):
            outfile = glob.glob(folder + "/*out")[0]
            fh = open(outfile, "r")
            data = fh.readlines()
            fh.close()

            start_time = None 
            end_time = None 

            for d in data:
                if 'Start time:' in d:
                    start_time = float(d.split(':')[-1].strip()) / 1000 
                elif 'End time:' in d:
                    end_time = float(d.split(':')[-1].strip()) / 1000 

            if 'downlink' in folder:
                if start_time == None or end_time == None:
                    if start_time == None:
                        continue 
                    else:
                        iperf_log_count = get_iperf_data_length(data)
                        end_time = start_time + (0.1 * iperf_log_count)
                dl_unix_ts_list.append((start_time, end_time))
            elif 'uplink' in folder:
                if start_time == None or end_time == None:
                    if start_time == None:
                        continue 
                    else:
                        iperf_log_count = get_iperf_data_length(data)
                        end_time = start_time + (0.1 * iperf_log_count)
                ul_unix_ts_list.append((start_time, end_time))
            elif 'rtt' in folder:
                if start_time == None or end_time == None:
                    if start_time == None:
                        continue 
                    else:
                        ping_log_count = get_icmp_ping_data_length(data)
                        end_time = start_time + (0.1 * ping_log_count)
                rtt_unix_ts_list.append((start_time, end_time))
    
    if not os.path.exists("../pkls/data_chicago_april_2025/tmobile_nsa_xcal_df.pkl"):
        tmobile_nsa_xcal_df = pd.read_excel(tmobile_nsa_xcal_file)

        tmobile_nsa_xcal_df.drop(tmobile_nsa_xcal_df.tail(8).index,inplace=True)
        tmobile_nsa_xcal_df['TIME_STAMP'] = tmobile_nsa_xcal_df['TIME_STAMP'].apply(datetime_to_timestamp)
        tmobile_nsa_xcal_df = tmobile_nsa_xcal_df.rename(columns={'TIME_STAMP' : 'Timestamp'})
        tmobile_nsa_xcal_df = tmobile_nsa_xcal_df.sort_values("Timestamp").reset_index(drop=True)

        fh = open("../pkls/data_chicago_april_2025/tmobile_nsa_xcal_df.pkl", "wb")
        pkl.dump(tmobile_nsa_xcal_df, fh)
        fh.close()
    else:
        fh = open("../pkls/data_chicago_april_2025/tmobile_nsa_xcal_df.pkl", "rb")
        tmobile_nsa_xcal_df = pkl.load(fh)
        fh.close()

    # important columns
    mac_cell_dl_xput_cols = [i for i in tmobile_nsa_xcal_df.columns if 'MAC DL Throughput' in i and '5G' in i and 'Total' not in i]
    mac_cell_ul_xput_cols = [i for i in tmobile_nsa_xcal_df.columns if 'MAC UL Throughput' in i and '5G' in i and 'Total' not in i]
    bw_cols = [i for i in tmobile_nsa_xcal_df.columns if 'RF BandWidth' in i and '5G' in i]
    band_cols = [i for i in tmobile_nsa_xcal_df.columns if 'RF Band' in i and '5G' in i and 'RF BandWidth' not in i]

    all_cols = ['Lat', 'Lon', 'Timestamp', 'Event Technology', '5G KPI PCell RF Serving PCI', '5G KPI PCell RF NR-ARFCN', '5G KPI Total Info Layer2 MAC DL Throughput [Mbps]', '5G KPI Total Info Layer2 MAC UL Throughput [Mbps]', '5G KPI Total Info DL CA Type', '5G KPI Total Info UL CA Type', '5G KPI PCell RF Serving SS-RSRP [dBm]', '5G KPI PCell RF PUSCH Power [dBm]', '5G KPI PCell RF PUCCH Power [dBm]', '5G KPI PCell RF Pathloss [dB]', 'Smart Phone Smart Throughput Mobile Network DL Throughput [Mbps]', 'Smart Phone Smart Throughput Mobile Network UL Throughput [Mbps]']
    all_cols.extend(mac_cell_dl_xput_cols)
    all_cols.extend(mac_cell_ul_xput_cols)
    all_cols.extend(bw_cols)
    all_cols.extend(band_cols)

    tmobile_nsa_xcal_df = tmobile_nsa_xcal_df[all_cols]
    dl_df_list_nsa = []
    ul_df_list_nsa = []
    ping_df_list_nsa = [] 
    total_event_tech_nsa = []
    # downlink first 
    for dl_test_start_end in dl_unix_ts_list:
        sub_df = tmobile_nsa_xcal_df[(tmobile_nsa_xcal_df['Timestamp'] >= dl_test_start_end[0]) & (tmobile_nsa_xcal_df['Timestamp'] <= dl_test_start_end[1])]

        total_event_tech_nsa.extend(sub_df['Event Technology'].dropna())
        dl_df_list_nsa.append(sub_df)

    # uplink second 
    for ul_test_start_end in ul_unix_ts_list:
        sub_df = tmobile_nsa_xcal_df[(tmobile_nsa_xcal_df['Timestamp'] >= ul_test_start_end[0]) & (tmobile_nsa_xcal_df['Timestamp'] <= ul_test_start_end[1])]

        total_event_tech_nsa.extend(sub_df['Event Technology'].dropna())
        ul_df_list_nsa.append(sub_df)

    # rtt third 
    for rtt_test_start_end in rtt_unix_ts_list:
        sub_df = tmobile_nsa_xcal_df[(tmobile_nsa_xcal_df['Timestamp'] >= rtt_test_start_end[0]) & (tmobile_nsa_xcal_df['Timestamp'] <= rtt_test_start_end[1])]

        total_event_tech_nsa.extend(sub_df['Event Technology'].dropna())
        ping_df_list_nsa.append(sub_df)


# SA
if 1:
    print("SA")
    dl_unix_ts_list = []
    ul_unix_ts_list = []
    rtt_unix_ts_list = []

    for day in glob.glob(tmobile_sa_app_dir + "/*"):
        for folder in glob.glob(day + "/*"):
            outfile = glob.glob(folder + "/*out")[0]
            fh = open(outfile, "r")
            data = fh.readlines()
            fh.close()

            start_time = None 
            end_time = None 

            for d in data:
                if 'Start time:' in d:
                    start_time = float(d.split(':')[-1].strip()) / 1000 
                elif 'End time:' in d:
                    end_time = float(d.split(':')[-1].strip()) / 1000 

            if 'downlink' in folder:
                if start_time == None or end_time == None:
                    if start_time == None:
                        continue 
                    else:
                        iperf_log_count = get_iperf_data_length(data)
                        end_time = start_time + (0.1 * iperf_log_count)
                dl_unix_ts_list.append((start_time, end_time))
            elif 'uplink' in folder:
                if start_time == None or end_time == None:
                    if start_time == None:
                        continue 
                    else:
                        iperf_log_count = get_iperf_data_length(data)
                        end_time = start_time + (0.1 * iperf_log_count)
                ul_unix_ts_list.append((start_time, end_time))
            elif 'rtt' in folder:
                if start_time == None or end_time == None:
                    if start_time == None:
                        continue 
                    else:
                        ping_log_count = get_icmp_ping_data_length(data)
                        end_time = start_time + (0.1 * ping_log_count)
                rtt_unix_ts_list.append((start_time, end_time))

    if not os.path.exists("../pkls/data_chicago_april_2025/tmobile_sa_xcal_df.pkl"):
        tmobile_sa_xcal_df = pd.read_excel(tmobile_sa_xcal_file)

        tmobile_sa_xcal_df.drop(tmobile_sa_xcal_df.tail(8).index,inplace=True)
        tmobile_sa_xcal_df['TIME_STAMP'] = tmobile_sa_xcal_df['TIME_STAMP'].apply(datetime_to_timestamp)
        tmobile_sa_xcal_df = tmobile_sa_xcal_df.rename(columns={'TIME_STAMP' : 'Timestamp'})
        tmobile_sa_xcal_df = tmobile_sa_xcal_df.sort_values("Timestamp").reset_index(drop=True)

        fh = open("../pkls/data_chicago_april_2025/tmobile_sa_xcal_df.pkl", "wb")
        pkl.dump(tmobile_sa_xcal_df, fh)
        fh.close()
    else:
        fh = open("../pkls/data_chicago_april_2025/tmobile_sa_xcal_df.pkl", "rb")
        tmobile_sa_xcal_df = pkl.load(fh)
        fh.close()
    # get 5G MAC xput
    mac_cell_dl_xput_cols = [i for i in tmobile_sa_xcal_df.columns if 'MAC DL Throughput' in i and '5G' in i and 'Total' not in i]
    mac_cell_ul_xput_cols = [i for i in tmobile_sa_xcal_df.columns if 'MAC UL Throughput' in i and '5G' in i and 'Total' not in i]
    bw_cols = [i for i in tmobile_sa_xcal_df.columns if 'RF BandWidth' in i and '5G' in i]
    band_cols = [i for i in tmobile_sa_xcal_df.columns if 'RF Band' in i and '5G' in i and 'RF BandWidth' not in i]

    all_cols = ['Lat', 'Lon', 'Timestamp', 'Event Technology', '5G KPI PCell RF Serving PCI', '5G KPI PCell RF NR-ARFCN', '5G KPI Total Info Layer2 MAC DL Throughput [Mbps]', '5G KPI Total Info Layer2 MAC UL Throughput [Mbps]', '5G KPI Total Info DL CA Type', '5G KPI Total Info UL CA Type', '5G KPI PCell RF Serving SS-RSRP [dBm]', '5G KPI PCell RF PUSCH Power [dBm]', '5G KPI PCell RF PUCCH Power [dBm]', '5G KPI PCell RF Pathloss [dB]', 'Smart Phone Smart Throughput Mobile Network DL Throughput [Mbps]', 'Smart Phone Smart Throughput Mobile Network UL Throughput [Mbps]']
    all_cols.extend(mac_cell_dl_xput_cols)
    all_cols.extend(mac_cell_ul_xput_cols)
    all_cols.extend(bw_cols)
    all_cols.extend(band_cols)

    tmobile_sa_xcal_df = tmobile_sa_xcal_df[all_cols]
    total_event_tech_sa = []
    dl_df_list_sa = []
    ul_df_list_sa = []
    ping_df_list_sa = [] 

    # downlink first 
    for dl_test_start_end in dl_unix_ts_list:
        sub_df = tmobile_sa_xcal_df[(tmobile_sa_xcal_df['Timestamp'] >= dl_test_start_end[0]) & (tmobile_sa_xcal_df['Timestamp'] <= dl_test_start_end[1])]

        total_event_tech_sa.extend(sub_df['Event Technology'].dropna())
        dl_df_list_sa.append(sub_df)

    # uplink second 
    for ul_test_start_end in ul_unix_ts_list:
        sub_df = tmobile_sa_xcal_df[(tmobile_sa_xcal_df['Timestamp'] >= ul_test_start_end[0]) & (tmobile_sa_xcal_df['Timestamp'] <= ul_test_start_end[1])]

        total_event_tech_sa.extend(sub_df['Event Technology'].dropna())
        ul_df_list_sa.append(sub_df)

    # rtt third 
    for rtt_test_start_end in rtt_unix_ts_list:
        sub_df = tmobile_sa_xcal_df[(tmobile_sa_xcal_df['Timestamp'] >= rtt_test_start_end[0]) & (tmobile_sa_xcal_df['Timestamp'] <= rtt_test_start_end[1])]

        total_event_tech_sa.extend(sub_df['Event Technology'].dropna())
        ping_df_list_sa.append(sub_df)

ts_diff_list_1 = list(tmobile_nsa_xcal_df['Timestamp'].diff())
ts_diff_list_1 = [i for i in ts_diff_list_1 if i <= 10]
total_mins_1 = sum(ts_diff_list_1) / 60

ts_diff_list_3 = list(tmobile_nsa_xcal_df['Timestamp'].diff())
ts_diff_list_3 = [i for i in ts_diff_list_3 if i <= 10]
total_mins_3 = sum(ts_diff_list_3) / 60

nsa_event_distance_dict = {}
nsa_arfcn_list = []

sa_event_distance_dict = {}
sa_arfcn_list = []

nsa_sa_same_row_arfcn = []
nsa_sa_same_diff_arfcn = []
nsa_sa_same_diff_pci = []

same_pci_same_arfcn = 0
same_pci_diff_arfcn = 0
diff_pci_same_arfcn = 0
diff_pci_diff_arfcn = 0

nsa_same_pci_same_arfcn_dl_xput = []
nsa_same_pci_diff_arfcn_dl_xput = []
nsa_diff_pci_same_arfcn_dl_xput = []
nsa_diff_pci_diff_arfcn_dl_xput = []

sa_same_pci_same_arfcn_dl_xput = []
sa_same_pci_diff_arfcn_dl_xput = []
sa_diff_pci_same_arfcn_dl_xput = []
sa_diff_pci_diff_arfcn_dl_xput = []

nsa_same_pci_same_arfcn_ul_xput = []
nsa_same_pci_diff_arfcn_ul_xput = []
nsa_diff_pci_same_arfcn_ul_xput = []
nsa_diff_pci_diff_arfcn_ul_xput = []

sa_same_pci_same_arfcn_ul_xput = []
sa_same_pci_diff_arfcn_ul_xput = []
sa_diff_pci_same_arfcn_ul_xput = []
sa_diff_pci_diff_arfcn_ul_xput = []
nsa_same_pci_same_arfcn_dl_pl = []
nsa_same_pci_diff_arfcn_dl_pl = []
sa_same_pci_same_arfcn_dl_pl = []
sa_same_pci_diff_arfcn_dl_pl = []

nsa_same_pci_same_arfcn_dl_power = []
nsa_same_pci_diff_arfcn_dl_power = []
sa_same_pci_same_arfcn_dl_power = []
sa_same_pci_diff_arfcn_dl_power = []

nsa_same_pci_same_arfcn_ul_pl = []
nsa_same_pci_diff_arfcn_ul_pl = []
sa_same_pci_same_arfcn_ul_pl = []
sa_same_pci_diff_arfcn_ul_pl = []

nsa_same_pci_same_arfcn_ul_power = []
nsa_same_pci_diff_arfcn_ul_power = []
sa_same_pci_same_arfcn_ul_power = []
sa_same_pci_diff_arfcn_ul_power = []

nsa_same_pci_same_arfcn_dl_bw = []
nsa_same_pci_diff_arfcn_dl_bw = []
sa_same_pci_same_arfcn_dl_bw = []
sa_same_pci_diff_arfcn_dl_bw = []

nsa_same_pci_same_arfcn_ul_bw = []
nsa_same_pci_diff_arfcn_ul_bw = []
sa_same_pci_same_arfcn_ul_bw = []
sa_same_pci_diff_arfcn_ul_bw = []
# downlink 
if 1:
    # coverage
    if 1:
        for df in dl_df_list_nsa:
            if len(df['Lat'].dropna()) == 0:
                continue
            ts =  list(df['Timestamp'])
            lat = list(df['Lat'])
            lon = list(df['Lon'])

            prev_lat = lat[0]
            prev_lon = lon[0]
            distance = []
            for lat_curr, lon_curr in zip(lat, lon):
                distance.append(geopy.distance.geodesic((prev_lat, prev_lon), (lat_curr, lon_curr)).miles)
                prev_lat = lat_curr
                prev_lon = lon_curr
            
            df['distance'] = distance 
            df['Event Technology'] = df['Event Technology'].fillna(method='ffill')
            sub_df = df[['Event Technology', 'distance']].dropna()
            distance_dict = sub_df.groupby('Event Technology')['distance'].sum().to_dict()
            for k, v in distance_dict.items():
                if k not in nsa_event_distance_dict.keys():
                    nsa_event_distance_dict[k] = 0
                nsa_event_distance_dict[k]+=v

        for df in dl_df_list_sa:
            if len(df['Lat'].dropna()) == 0:
                continue
            ts =  list(df['Timestamp'])
            lat = list(df['Lat'])
            lon = list(df['Lon'])

            prev_lat = lat[0]
            prev_lon = lon[0]
            distance = []
            for lat_curr, lon_curr in zip(lat, lon):
                distance.append(geopy.distance.geodesic((prev_lat, prev_lon), (lat_curr, lon_curr)).miles)
                prev_lat = lat_curr
                prev_lon = lon_curr
            
            df['distance'] = distance 
            df['Event Technology'] = df['Event Technology'].fillna(method='ffill')
            sub_df = df[['Event Technology', 'distance']].dropna()
            distance_dict = sub_df.groupby('Event Technology')['distance'].sum().to_dict()
            for k, v in distance_dict.items():
                if k not in sa_event_distance_dict.keys():
                    sa_event_distance_dict[k] = 0
                sa_event_distance_dict[k]+=v

    # performance CDFs
    if 1:
        merged_nsa_dl_df = pd.concat(dl_df_list_nsa)
        merged_nsa_dl_df = merged_nsa_dl_df.sort_values("Timestamp").reset_index(drop=True)
        merged_nsa_dl_df = merged_nsa_dl_df[['Timestamp', 'Lat', 'Lon', 'Event Technology', '5G KPI PCell RF Serving PCI', '5G KPI PCell RF NR-ARFCN', '5G KPI PCell RF Band', '5G KPI PCell RF BandWidth', '5G KPI Total Info Layer2 MAC DL Throughput [Mbps]', '5G KPI PCell Layer2 MAC DL Throughput [Mbps]', '5G KPI PCell RF Serving SS-RSRP [dBm]', '5G KPI PCell RF Pathloss [dB]']]
        merged_sa_dl_df = pd.concat(dl_df_list_sa)
        merged_sa_dl_df = merged_sa_dl_df.sort_values("Timestamp").reset_index(drop=True)
        merged_sa_dl_df = merged_sa_dl_df[['Timestamp', 'Lat', 'Lon', 'Event Technology', '5G KPI PCell RF Serving PCI', '5G KPI PCell RF NR-ARFCN', '5G KPI PCell RF Band', '5G KPI PCell RF BandWidth', '5G KPI Total Info Layer2 MAC DL Throughput [Mbps]', '5G KPI PCell Layer2 MAC DL Throughput [Mbps]', '5G KPI PCell RF Serving SS-RSRP [dBm]', '5G KPI PCell RF Pathloss [dB]']]

        # Perform asof merge with 1-second tolerance
        merged_df = pd.merge_asof(
            merged_nsa_dl_df,
            merged_sa_dl_df,
            on='Timestamp',
            direction='nearest',
            tolerance=0.2  # 1 second tolerance
        )

        # Drop duplicate rows if any
        merged_df = merged_df.drop_duplicates()
        merged_df = merged_df.dropna(
            subset=[
                '5G KPI Total Info Layer2 MAC DL Throughput [Mbps]_x',
                '5G KPI Total Info Layer2 MAC DL Throughput [Mbps]_y'
            ]
        )
        
        nsa_dl_xput = list(merged_df['5G KPI Total Info Layer2 MAC DL Throughput [Mbps]_x'].dropna())
        sa_dl_xput =  list(merged_df['5G KPI Total Info Layer2 MAC DL Throughput [Mbps]_y'].dropna())

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(np.sort(nsa_dl_xput), np.linspace(0, 1, len(nsa_dl_xput)), label='NSA', color='salmon')
        ax.plot(np.sort(sa_dl_xput), np.linspace(0, 1, len(sa_dl_xput)), label='SA', color='slategrey')

        ax.set_xlabel("Throughput (Mbps)", fontsize=22)
        ax.set_ylabel("CDF", fontsize=22)

        ax.legend(loc='best', fontsize=16)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1000)
        ax.grid(True)

        plt.tight_layout()
        plt.savefig('../plots/yearwise/tmobile_nsa_sa_dl_xput.pdf')
        plt.close()
        
        nsa_arfcn_list.extend(list(merged_df['5G KPI PCell RF NR-ARFCN_x'].dropna()))
        sa_arfcn_list.extend(list(merged_df['5G KPI PCell RF NR-ARFCN_y'].dropna()))

        temp = merged_df[['5G KPI PCell RF NR-ARFCN_x', '5G KPI PCell RF NR-ARFCN_y']].dropna()
        nsa_arfcn_list = list(temp['5G KPI PCell RF NR-ARFCN_x'])
        sa_arfcn_list =  list(temp['5G KPI PCell RF NR-ARFCN_y'])

        for nsa, sa in zip(nsa_arfcn_list, sa_arfcn_list):
            nsa_sa_same_row_arfcn.append("%s-%s" %(str(int(nsa)), str(int(sa))))
            if nsa == sa:
                nsa_sa_same_diff_arfcn.append("Same ARFCN")
            else:
                nsa_sa_same_diff_arfcn.append("Different ARFCN")



        # temp = merged_df[['5G KPI PCell RF Serving PCI_x', '5G KPI PCell RF Serving PCI_y', '5G KPI PCell RF NR-ARFCN_x', '5G KPI PCell RF NR-ARFCN_y', '5G KPI Total Info Layer2 MAC DL Throughput [Mbps]_x', '5G KPI Total Info Layer2 MAC DL Throughput [Mbps]_y', '5G KPI PCell Layer2 MAC DL Throughput [Mbps]_x', '5G KPI PCell Layer2 MAC DL Throughput [Mbps]_y']].dropna()
        temp = merged_df[['5G KPI PCell RF Serving PCI_x', '5G KPI PCell RF Serving PCI_y', '5G KPI PCell RF NR-ARFCN_x', '5G KPI PCell RF NR-ARFCN_y', '5G KPI Total Info Layer2 MAC DL Throughput [Mbps]_x', '5G KPI Total Info Layer2 MAC DL Throughput [Mbps]_y', '5G KPI PCell Layer2 MAC DL Throughput [Mbps]_x', '5G KPI PCell Layer2 MAC DL Throughput [Mbps]_y', '5G KPI PCell RF Serving SS-RSRP [dBm]_x', '5G KPI PCell RF Serving SS-RSRP [dBm]_y', '5G KPI PCell RF Pathloss [dB]_x', '5G KPI PCell RF Pathloss [dB]_y', '5G KPI PCell RF BandWidth_x', '5G KPI PCell RF BandWidth_y']].dropna()
        nsa_arfcn_list = list(temp['5G KPI PCell RF NR-ARFCN_x'])
        sa_arfcn_list =  list(temp['5G KPI PCell RF NR-ARFCN_y'])
        nsa_pci_list = list(temp['5G KPI PCell RF Serving PCI_x'])
        sa_pci_list =  list(temp['5G KPI PCell RF Serving PCI_y'])
        nsa_xput_list = list(temp['5G KPI PCell Layer2 MAC DL Throughput [Mbps]_x'])
        sa_xput_list = list(temp['5G KPI PCell Layer2 MAC DL Throughput [Mbps]_y'])
        nsa_pl_list = list(temp['5G KPI PCell RF Pathloss [dB]_x'])
        sa_pl_list = list(temp['5G KPI PCell RF Pathloss [dB]_y'])
        nsa_power_list = list(temp['5G KPI PCell RF Serving SS-RSRP [dBm]_x'])
        sa_power_list = list(temp['5G KPI PCell RF Serving SS-RSRP [dBm]_y'])
        nsa_bw_list = list(temp['5G KPI PCell RF BandWidth_x'])
        sa_bw_list = list(temp['5G KPI PCell RF BandWidth_y'])

        for nsa_pci, sa_pci, nsa_arfcn, sa_arfcn, nsa_xput, sa_xput, nsa_pl, sa_pl, nsa_power, sa_power, nsa_bw, sa_bw in zip(nsa_pci_list, sa_pci_list, nsa_arfcn_list, sa_arfcn_list, nsa_xput_list, sa_xput_list, nsa_pl_list, sa_pl_list, nsa_power_list, sa_power_list, nsa_bw_list, sa_bw_list):
            if nsa_pci == sa_pci:
                if nsa_arfcn == sa_arfcn:
                    same_pci_same_arfcn+=1
                    nsa_same_pci_same_arfcn_dl_xput.append(nsa_xput)
                    sa_same_pci_same_arfcn_dl_xput.append(sa_xput)

                    nsa_same_pci_same_arfcn_dl_pl.append(nsa_pl)
                    sa_same_pci_same_arfcn_dl_pl.append(sa_pl)

                    nsa_same_pci_same_arfcn_dl_power.append(nsa_power)
                    sa_same_pci_same_arfcn_dl_power.append(sa_power)

                    nsa_same_pci_same_arfcn_dl_bw.append(nsa_bw)
                    sa_same_pci_same_arfcn_dl_bw.append(sa_bw)
                else:
                    same_pci_diff_arfcn+=1
                    nsa_same_pci_diff_arfcn_dl_xput.append(nsa_xput)
                    sa_same_pci_diff_arfcn_dl_xput.append(sa_xput)

                    nsa_same_pci_diff_arfcn_dl_pl.append(nsa_pl)
                    sa_same_pci_diff_arfcn_dl_pl.append(sa_pl)

                    nsa_same_pci_diff_arfcn_dl_power.append(nsa_power)
                    sa_same_pci_diff_arfcn_dl_power.append(sa_power)

                    nsa_same_pci_diff_arfcn_dl_bw.append(nsa_bw)
                    sa_same_pci_diff_arfcn_dl_bw.append(sa_bw)
            else:
                if nsa_arfcn == sa_arfcn:
                    diff_pci_same_arfcn+=1
                    nsa_diff_pci_same_arfcn_dl_xput.append(nsa_xput)
                    sa_diff_pci_same_arfcn_dl_xput.append(sa_xput)
                else:
                    diff_pci_diff_arfcn+=1
                    nsa_diff_pci_diff_arfcn_dl_xput.append(nsa_xput)
                    sa_diff_pci_diff_arfcn_dl_xput.append(sa_xput)


# uplink 
if 1:
    if 1:
        for df in ul_df_list_nsa:
            if len(df) == 0:
                continue
            ts =  list(df['Timestamp'])
            lat = list(df['Lat'])
            lon = list(df['Lon'])

            prev_lat = lat[0]
            prev_lon = lon[0]
            distance = []
            for lat_curr, lon_curr in zip(lat, lon):
                distance.append(geopy.distance.geodesic((prev_lat, prev_lon), (lat_curr, lon_curr)).miles)
                prev_lat = lat_curr
                prev_lon = lon_curr
            
            df['distance'] = distance 
            df['Event Technology'] = df['Event Technology'].fillna(method='ffill')
            sub_df = df[['Event Technology', 'distance']].dropna()
            distance_dict = sub_df.groupby('Event Technology')['distance'].sum().to_dict()
            for k, v in distance_dict.items():
                if k not in nsa_event_distance_dict.keys():
                    nsa_event_distance_dict[k] = 0
                nsa_event_distance_dict[k]+=v

        for df in ul_df_list_sa:
            if len(df['Lat'].dropna()) == 0:
                continue
            ts =  list(df['Timestamp'])
            lat = list(df['Lat'])
            lon = list(df['Lon'])

            prev_lat = lat[0]
            prev_lon = lon[0]
            distance = []
            for lat_curr, lon_curr in zip(lat, lon):
                distance.append(geopy.distance.geodesic((prev_lat, prev_lon), (lat_curr, lon_curr)).miles)
                prev_lat = lat_curr
                prev_lon = lon_curr
            
            df['distance'] = distance 
            df['Event Technology'] = df['Event Technology'].fillna(method='ffill')
            sub_df = df[['Event Technology', 'distance']].dropna()
            distance_dict = sub_df.groupby('Event Technology')['distance'].sum().to_dict()
            for k, v in distance_dict.items():
                if k not in sa_event_distance_dict.keys():
                    sa_event_distance_dict[k] = 0
                sa_event_distance_dict[k]+=v


    if 1:
        merged_nsa_ul_df = pd.concat(ul_df_list_nsa)
        merged_nsa_ul_df = merged_nsa_ul_df.sort_values("Timestamp").reset_index(drop=True)
        merged_nsa_ul_df = merged_nsa_ul_df[['Timestamp', 'Lat', 'Lon', 'Event Technology', '5G KPI PCell RF Serving PCI', '5G KPI PCell RF NR-ARFCN', '5G KPI PCell RF Band', '5G KPI PCell RF BandWidth', '5G KPI Total Info Layer2 MAC UL Throughput [Mbps]', '5G KPI PCell Layer2 MAC UL Throughput [Mbps]', '5G KPI PCell RF PUSCH Power [dBm]', '5G KPI PCell RF Pathloss [dB]']]
        merged_sa_ul_df = pd.concat(ul_df_list_sa)
        merged_sa_ul_df = merged_sa_ul_df.sort_values("Timestamp").reset_index(drop=True)
        merged_sa_ul_df = merged_sa_ul_df[['Timestamp', 'Lat', 'Lon', 'Event Technology', '5G KPI PCell RF Serving PCI', '5G KPI PCell RF NR-ARFCN', '5G KPI PCell RF Band', '5G KPI PCell RF BandWidth', '5G KPI Total Info Layer2 MAC UL Throughput [Mbps]', '5G KPI PCell Layer2 MAC UL Throughput [Mbps]', '5G KPI PCell RF PUSCH Power [dBm]', '5G KPI PCell RF Pathloss [dB]']]

        # Perform asof merge with 1-second tolerance
        merged_df = pd.merge_asof(
            merged_nsa_ul_df,
            merged_sa_ul_df,
            on='Timestamp',
            direction='nearest',
            tolerance=0.2  # 1 second tolerance
        )

        # Drop duplicate rows if any
        merged_df = merged_df.drop_duplicates()
        merged_df = merged_df.dropna(
            subset=[
                '5G KPI Total Info Layer2 MAC UL Throughput [Mbps]_x',
                '5G KPI Total Info Layer2 MAC UL Throughput [Mbps]_y'
            ]
        )
        
        nsa_ul_xput = list(merged_df['5G KPI Total Info Layer2 MAC UL Throughput [Mbps]_x'].dropna())
        sa_ul_xput =  list(merged_df['5G KPI Total Info Layer2 MAC UL Throughput [Mbps]_y'].dropna())

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(np.sort(nsa_ul_xput), np.linspace(0, 1, len(nsa_ul_xput)), label='NSA', color='salmon')
        ax.plot(np.sort(sa_ul_xput), np.linspace(0, 1, len(sa_ul_xput)), label='SA', color='slategrey')

        ax.set_xlabel("Throughput (Mbps)", fontsize=22)
        ax.set_ylabel("CDF", fontsize=22)

        ax.legend(loc='best', fontsize=16)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 150)
        ax.grid(True)

        plt.tight_layout()
        plt.savefig('../plots/yearwise/tmobile_nsa_sa_ul_xput.pdf')
        plt.close()

        nsa_arfcn_list.extend(list(merged_df['5G KPI PCell RF NR-ARFCN_x'].dropna()))
        sa_arfcn_list.extend(list(merged_df['5G KPI PCell RF NR-ARFCN_y'].dropna()))

        temp = merged_df[['5G KPI PCell RF NR-ARFCN_x', '5G KPI PCell RF NR-ARFCN_y']].dropna()
        nsa_arfcn_list = list(temp['5G KPI PCell RF NR-ARFCN_x'])
        sa_arfcn_list =  list(temp['5G KPI PCell RF NR-ARFCN_y'])

        for nsa, sa in zip(nsa_arfcn_list, sa_arfcn_list):
            nsa_sa_same_row_arfcn.append("%s-%s" %(str(int(nsa)), str(int(sa))))
            if nsa == sa:
                nsa_sa_same_diff_arfcn.append("Same ARFCN")
            else:
                nsa_sa_same_diff_arfcn.append("Different ARFCN")

        temp = merged_df[['5G KPI PCell RF Serving PCI_x', '5G KPI PCell RF Serving PCI_y']].dropna()
        nsa_pci_list = list(temp['5G KPI PCell RF Serving PCI_x'])
        sa_pci_list =  list(temp['5G KPI PCell RF Serving PCI_y'])

        for nsa, sa in zip(nsa_pci_list, sa_pci_list):
            if nsa == sa:
                nsa_sa_same_diff_pci.append("Same PCI")
            else:
                nsa_sa_same_diff_pci.append("Different PCI")
        
        temp = merged_df[['5G KPI PCell RF Serving PCI_x', '5G KPI PCell RF Serving PCI_y', '5G KPI PCell RF NR-ARFCN_x', '5G KPI PCell RF NR-ARFCN_y', '5G KPI Total Info Layer2 MAC UL Throughput [Mbps]_x', '5G KPI Total Info Layer2 MAC UL Throughput [Mbps]_y', '5G KPI PCell Layer2 MAC UL Throughput [Mbps]_x', '5G KPI PCell Layer2 MAC UL Throughput [Mbps]_y', '5G KPI PCell RF PUSCH Power [dBm]_x', '5G KPI PCell RF Pathloss [dB]_x', '5G KPI PCell RF PUSCH Power [dBm]_y', '5G KPI PCell RF Pathloss [dB]_y', '5G KPI PCell RF BandWidth_x', '5G KPI PCell RF BandWidth_y']].dropna()
        nsa_arfcn_list = list(temp['5G KPI PCell RF NR-ARFCN_x'])
        sa_arfcn_list =  list(temp['5G KPI PCell RF NR-ARFCN_y'])
        nsa_pci_list = list(temp['5G KPI PCell RF Serving PCI_x'])
        sa_pci_list =  list(temp['5G KPI PCell RF Serving PCI_y'])
        nsa_xput_list = list(temp['5G KPI PCell Layer2 MAC UL Throughput [Mbps]_x'])
        sa_xput_list = list(temp['5G KPI PCell Layer2 MAC UL Throughput [Mbps]_y'])

        nsa_pl_list = list(temp['5G KPI PCell RF Pathloss [dB]_x'])
        sa_pl_list = list(temp['5G KPI PCell RF Pathloss [dB]_y'])
        nsa_power_list = list(temp['5G KPI PCell RF PUSCH Power [dBm]_x'])
        sa_power_list = list(temp['5G KPI PCell RF PUSCH Power [dBm]_y'])


        for nsa_pci, sa_pci, nsa_arfcn, sa_arfcn, nsa_xput, sa_xput, nsa_pl, sa_pl, nsa_power, sa_power, nsa_bw, sa_bw in zip(nsa_pci_list, sa_pci_list, nsa_arfcn_list, sa_arfcn_list, nsa_xput_list, sa_xput_list, nsa_pl_list, sa_pl_list, nsa_power_list, sa_power_list, nsa_bw_list, sa_bw_list):
            if nsa_pci == sa_pci:
                if nsa_arfcn == sa_arfcn:
                    same_pci_same_arfcn+=1
                    nsa_same_pci_same_arfcn_ul_xput.append(nsa_xput)
                    sa_same_pci_same_arfcn_ul_xput.append(sa_xput)

                    nsa_same_pci_same_arfcn_ul_pl.append(nsa_pl)
                    sa_same_pci_same_arfcn_ul_pl.append(sa_pl)

                    nsa_same_pci_same_arfcn_ul_power.append(nsa_power)
                    sa_same_pci_same_arfcn_ul_power.append(sa_power)

                    nsa_same_pci_same_arfcn_ul_bw.append(nsa_bw)
                    sa_same_pci_same_arfcn_ul_bw.append(sa_bw)
                else:
                    same_pci_diff_arfcn+=1
                    nsa_same_pci_diff_arfcn_ul_xput.append(nsa_xput)
                    sa_same_pci_diff_arfcn_ul_xput.append(sa_xput)

                    nsa_same_pci_diff_arfcn_ul_pl.append(nsa_pl)
                    sa_same_pci_diff_arfcn_ul_pl.append(sa_pl)

                    nsa_same_pci_diff_arfcn_ul_power.append(nsa_power)
                    sa_same_pci_diff_arfcn_ul_power.append(sa_power)

                    nsa_same_pci_diff_arfcn_ul_bw.append(nsa_bw)
                    sa_same_pci_diff_arfcn_ul_bw.append(sa_bw)
            else:
                if nsa_arfcn == sa_arfcn:
                    diff_pci_same_arfcn+=1
                    nsa_diff_pci_same_arfcn_ul_xput.append(nsa_xput)
                    sa_diff_pci_same_arfcn_ul_xput.append(sa_xput)
                else:
                    diff_pci_diff_arfcn+=1
                    nsa_diff_pci_diff_arfcn_ul_xput.append(nsa_xput)
                    sa_diff_pci_diff_arfcn_ul_xput.append(sa_xput)


if 1:
    bw_diff_flattened = []
    bw_dict = {'nsa' : [], 'sa' : []}
    bw_diff = {'sa >= nsa' : [], 'sa < nsa' : []}
    for nsa_xput, sa_xput, nsa_bw, sa_bw in zip(nsa_same_pci_same_arfcn_dl_xput, sa_same_pci_same_arfcn_dl_xput, nsa_same_pci_same_arfcn_dl_bw, sa_same_pci_same_arfcn_dl_bw):
        if sa_xput >= nsa_xput:
            bw_diff['sa >= nsa'].append(sa_bw - nsa_bw)
        elif sa_xput < nsa_xput:
            bw_diff['sa < nsa'].append(sa_bw - nsa_bw)
        bw_diff_flattened.append(sa_bw - nsa_bw)
        bw_dict['nsa'].append(nsa_bw)
        bw_dict['sa'].append(sa_bw)

    same_pci_same_arfcn_dl_bw_diff = bw_diff.copy()
    same_pci_same_arfcn_dl_bw_diff_flattened = bw_diff_flattened.copy()
    same_pci_same_arfcn_dl_bw = bw_dict.copy()

    bw_dict = {'nsa' : [], 'sa' : []}
    bw_diff_flattened = []
    bw_diff = {'sa >= nsa' : [], 'sa < nsa' : []}
    for nsa_xput, sa_xput, nsa_bw, sa_bw in zip(nsa_same_pci_diff_arfcn_dl_xput, sa_same_pci_diff_arfcn_dl_xput, nsa_same_pci_diff_arfcn_dl_bw, sa_same_pci_diff_arfcn_dl_bw):
        if sa_xput >= nsa_xput:
            bw_diff['sa >= nsa'].append(sa_bw - nsa_bw)
        elif sa_xput < nsa_xput:
            bw_diff['sa < nsa'].append(sa_bw - nsa_bw)
        bw_diff_flattened.append(sa_bw - nsa_bw)
        bw_dict['nsa'].append(nsa_bw)
        bw_dict['sa'].append(sa_bw)

    same_pci_diff_arfcn_dl_bw_diff = bw_diff.copy()
    same_pci_diff_arfcn_dl_bw_diff_flattened = bw_diff_flattened.copy()
    same_pci_diff_arfcn_dl_bw = bw_dict.copy()

    bw_diff_flattened = []
    bw_dict = {'nsa' : [], 'sa' : []}
    bw_diff = {'sa >= nsa' : [], 'sa < nsa' : []}
    for nsa_xput, sa_xput, nsa_bw, sa_bw in zip(nsa_same_pci_same_arfcn_ul_xput, sa_same_pci_same_arfcn_ul_xput, nsa_same_pci_same_arfcn_ul_bw, sa_same_pci_same_arfcn_ul_bw):
        if sa_xput >= nsa_xput:
            bw_diff['sa >= nsa'].append(sa_bw - nsa_bw)
        elif sa_xput < nsa_xput:
            bw_diff['sa < nsa'].append(sa_bw - nsa_bw)
        bw_diff_flattened.append(sa_bw - nsa_bw)
        bw_dict['nsa'].append(nsa_bw)
        bw_dict['sa'].append(sa_bw)

    same_pci_same_arfcn_ul_bw_diff = bw_diff.copy()
    same_pci_same_arfcn_ul_bw_diff_flattened = bw_diff_flattened.copy()
    same_pci_same_arfcn_ul_bw = bw_dict.copy()


    bw_diff_flattened = []
    bw_dict = {'nsa' : [], 'sa' : []}
    bw_diff = {'sa >= nsa' : [], 'sa < nsa' : []}
    for nsa_xput, sa_xput, nsa_bw, sa_bw in zip(nsa_same_pci_diff_arfcn_ul_xput, sa_same_pci_diff_arfcn_ul_xput, nsa_same_pci_diff_arfcn_ul_bw, sa_same_pci_diff_arfcn_ul_bw):
        if sa_xput >= nsa_xput:
            bw_diff['sa >= nsa'].append(sa_bw - nsa_bw)
        elif sa_xput < nsa_xput:
            bw_diff['sa < nsa'].append(sa_bw - nsa_bw)
        bw_diff_flattened.append(sa_bw - nsa_bw)
        bw_dict['nsa'].append(nsa_bw)
        bw_dict['sa'].append(sa_bw)

    same_pci_diff_arfcn_ul_bw_diff = bw_diff.copy()
    same_pci_diff_arfcn_ul_bw_diff_flattened = bw_diff_flattened.copy()
    same_pci_diff_arfcn_ul_bw = bw_dict.copy()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    ax[0].plot(np.sort(same_pci_same_arfcn_dl_bw_diff_flattened), np.linspace(0, 1, len(same_pci_same_arfcn_dl_bw_diff_flattened)), color='salmon', label='Same PCI Same ARFCN', lw=3)
    ax[0].plot(np.sort(same_pci_diff_arfcn_dl_bw_diff_flattened), np.linspace(0, 1, len(same_pci_diff_arfcn_dl_bw_diff_flattened)), color='slategray', label='Same PCI Diff ARFCN', lw=3)

    ax[1].plot(np.sort(same_pci_same_arfcn_ul_bw_diff_flattened), np.linspace(0, 1, len(same_pci_same_arfcn_ul_bw_diff_flattened)), color='salmon', label='Same PCI Same ARFCN', lw=3)
    ax[1].plot(np.sort(same_pci_diff_arfcn_ul_bw_diff_flattened), np.linspace(0, 1, len(same_pci_diff_arfcn_ul_bw_diff_flattened)), color='slategray', label='Same PCI Diff ARFCN', lw=3)

    ax[0].set_title("Downlink", fontweight='bold' , fontsize=18)
    ax[1].set_title("Uplink"   , fontweight='bold', fontsize=18)
    ax[0].set_ylim(0, 1)
    ax[0].set_xlim(-30, 30)
    ax[1].set_xlim(-30, 30)
    ax[0].legend(loc='best', fontsize=14)
    ax[0].grid(True)
    ax[1].grid(True)
    ax[0].set_ylabel('CDF', fontsize=20)
    fig.text(0.5, -0.04, "$BW_{SA} - BW_{NSA}$ (MHz)", ha='center', fontsize=20)
    plt.tight_layout()
    plt.savefig('../plots/yearwise/tmobile_nsa_sa_bw_PCI_ARFCN_flattened.pdf')
    plt.close()