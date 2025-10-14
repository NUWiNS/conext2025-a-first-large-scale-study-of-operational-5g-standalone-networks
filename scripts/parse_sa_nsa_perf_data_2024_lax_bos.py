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
import geopy.distance
import datetime
from geopy.distance import distance
from geopy.distance import geodesic
from shapely.geometry import Point
from collections import Counter
from pprint import pprint 
import matplotlib.patches as mpatches
from timezonefinder import TimezoneFinder
obj = TimezoneFinder()

def calculate_averages(lst):
    """
    This function calculates averages for every 4 and 8 samples from the input list.
    Args:
        lst (list): The input list of throughput samples.
    Returns:
        tuple: Two lists containing averages of every 4 and 8 samples.
    """
    # List to store averages for every 4 samples
    averages_4 = [
        sum(lst[i:i+4]) / 4 for i in range(0, len(lst) - (len(lst) % 4), 4)
    ]

    # List to store averages for every 8 samples
    averages_8 = [
        sum(lst[i:i+8]) / 8 for i in range(0, len(lst) - (len(lst) % 8), 8)
    ]

    return averages_4, averages_8


def get_tz_info(first_lat_lon):
    try:
        temp_first_tz = obj.timezone_at(lng=first_lat_lon[-1], lat=first_lat_lon[0])
        if "Indiana" in temp_first_tz:
            temp_first_tz = 'America/New_York'
        elif "Phoenix" in temp_first_tz:
            temp_first_tz = 'America/Denver'
        return temp_first_tz
    except Exception as ex:
        print("TZ cannot be fetched! Why?????")
        print(str(ex))
        return None
    

def return_nsa_ho_type(before_pci_count_max_occurrence, after_pci_count_max_occurrence, before_arfcn_count_max_occurrence, after_arfcn_count_max_occurrence):
    # ['intra_gnb', 'intra_freq'], ['intra_gnb', 'inter_freq'], ['inter_gnb', 'intra_freq'], ['inter_gnb', 'inter_freq']
    dict_key = ""
    if None in [before_pci_count_max_occurrence, after_pci_count_max_occurrence, before_arfcn_count_max_occurrence, after_arfcn_count_max_occurrence]:
        return dict_key
    
    if before_pci_count_max_occurrence == after_pci_count_max_occurrence:
        dict_key+='intra_gnb:'
    else:
        dict_key+='inter_gnb:'
    
    if before_arfcn_count_max_occurrence == after_arfcn_count_max_occurrence:
        dict_key+='intra_freq'
    else:
        dict_key+='inter_freq'

    return dict_key 


def calculate_percentage_of_occurrence(lst):
    total_elements = len(lst)
    
    # Use Counter to count the occurrence of each element
    element_counts = Counter(lst)

    # Calculate the percentage for each unique element
    percentage_dict = {element: round((count / total_elements) * 100) for element, count in element_counts.items()}

    return percentage_dict

def datetime_to_timestamp(datetime_str):
    int(datetime_str.astimezone(datetime.timezone.utc).timestamp())
    return datetime_str.astimezone(datetime.timezone.utc).timestamp()

def get_start_end_indices(df, start_time, end_time):
    # Get the index where the timestamp is greater than or equal to the start_time
    if len(df[df['Timestamp'] >= start_time]) == 0 or len(df[df['Timestamp'] <= end_time]) == 0:
        return pd.DataFrame()
    start_index = df[df['Timestamp'] >= start_time].index[0]

    # Get the index where the timestamp is less than or equal to the end_time
    end_index = df[df['Timestamp'] <= end_time].index[-1]

    return df[start_index:end_index]

def get_nuttcp_data_length(data):
    count = 0
    for d in data:
        if '0.50 sec = ' in d:
            count+=1 
    return count

def get_icmp_ping_data_length(data):
    count = 0
    for d in data:
        if 'icmp_seq=' in d and 'ttl=' in d:
            count+=1 
    return count

def get_tcp_ping_data_length(data):
    count = 0
    for d in data:
        if 'client' in d and 'INFO' in d and "bytes from" in d:
            count+=1 
    return count

def get_ho_type_for_timestamp(df, timestamp):
    """
    Returns the ho_type for a given timestamp.
    If the ho_type is missing, returns None.
    """
    latency = df.loc[df['Timestamp'] == timestamp, 'latency']
    return latency.iloc[0] if not latency.empty and not pd.isna(latency.iloc[0]) else None

# global vars 
possible_ho = ['NR To EUTRA Redirection Success', 'NR Interfreq Handover Success', 'ulInformationTransferMRDC', 'MCG DRB Success', 'NR SCG Addition Success', 'Mobility from NR to EUTRA Success', 'NR Intrafreq Handover Success', 'NR SCG Modification Success', 'scgFailureInformationNR', 'Handover Success', 'EUTRA To NR Redirection Success']

main_xput_tech_dl_dict = {'Skip' : 0}
main_xput_tech_ul_dict = {'Skip' : 0}
main_xput_tech_ping_dict = {'Skip' : 0}
main_xput_tech_ping_dict_day_wise = {'Skip' : 0}
main_tech_ping_band_dict = {'NSA' : {}, 'SA' : {}}
main_tech_ping_band_scs_dict = {'NSA' : {}, 'SA' : {}}
n25_ping_location = []
main_tech_ping_city_info_dict = {'NSA' : {}, 'SA' : {}}
main_tech_ping_rsrp_dict = {'NSA' : [], 'SA' : []}
main_tech_ping_pathloss_dict = {'NSA' : [], 'SA' : []}
main_ping_tx_power_dict = {'NSA' : [], 'SA' : []}
main_ping_tx_power_control_dict = {'NSA' : [], 'SA' : []}

fiveg_xput_tech_dl_dict = {'Skip' : 0}
lte_xput_tech_dl_dict = {'Skip' : 0}
fiveg_xput_tech_ul_dict = {'Skip' : 0}
lte_xput_tech_ul_dict = {'Skip' : 0}
main_sa_nsa_lte_tech_dl_mimo_dict = {}
main_sa_nsa_lte_tech_dl_mimo_layer_dict = {}

main_sa_nsa_lte_tech_ca_band_xput_dict = {'SA' : {}, 'NSA' : {}}
main_sa_nsa_lte_tech_ca_ul_band_xput_dict = {'SA' : {}, 'NSA' : {}}
main_sa_nsa_lte_tech_ca_band_combo_dict = {'5G (SA)' : {}, '5G (NSA)' : {}}
main_sa_nsa_lte_tech_ca_ul_band_combo_dict = {'5G (SA)' : {}, '5G (NSA)' : {}}
main_sa_nsa_tx_power_dict = {'5G (NSA)' : [], '5G (SA)' : []}
main_sa_nsa_tx_power_control_dict = {'5G (NSA)' : [], '5G (SA)' : []}
main_sa_nsa_rx_power_dict = {'5G (NSA)' : [], '5G (SA)' : []}
main_sa_nsa_pathloss_dict = {'5G (NSA)' : [], '5G (SA)' : []}
main_lat_lon_tech_df = pd.DataFrame()
main_sa_nsa_lte_tech_time_dict = {}
main_sa_nsa_lte_tech_dist_dict = {}
main_sa_nsa_lte_tech_dist_dict_tz = {'5G (SA)' : {}, '5G (NSA)' : {}}
main_sa_nsa_lte_tech_dist_dict_city = {'5G (SA)' : {}, '5G (NSA)' : {}}
main_sa_nsa_lte_tech_dist_dict_chic_bos = {}
main_sa_nsa_lte_tech_band_dict = {}
main_sa_nsa_lte_tech_ca_dict = {}
main_sa_nsa_lte_tech_ca_ul_dict = {}
main_sa_nsa_lte_tech_city_ca_dict = {}
main_sa_nsa_lte_tech_city_ca_ul_dict = {}

main_tech_xput_city_info_dict = {'NSA' : {}, 'SA' : {}}
main_tech_xput_tx_power_dict = {'NSA' : [], 'SA' : []}
main_tech_xput_tx_power_control_dict = {'NSA' : [], 'SA' : []}
main_tech_xput_dl_tx_power_dict = {'NSA' : [], 'SA' : []}
main_tech_xput_dl_tx_power_control_dict = {'NSA' : [], 'SA' : []}
main_tech_xput_ul_tx_power_dict = {'NSA' : [], 'SA' : []}
main_tech_xput_ul_tx_power_control_dict = {'NSA' : [], 'SA' : []}
main_tech_xput_rx_power_dict = {'NSA' : [], 'SA' : []}
main_tech_xput_dl_rx_power_dict = {'NSA' : [], 'SA' : []}
main_tech_xput_ul_rx_power_dict = {'NSA' : [], 'SA' : []}
main_tech_xput_pathloss_dict = {'NSA' : [], 'SA' : []}
main_tech_xput_dl_pathloss_dict = {'NSA' : [], 'SA' : []}
main_tech_xput_ul_pathloss_dict = {'NSA' : [], 'SA' : []}
main_tech_xput_dl_bandwidth_dict = {'NSA' : {}, 'SA' : {}}
main_tech_xput_ul_bandwidth_dict = {'NSA' : {}, 'SA' : {}}
main_tech_xput_dl_ca_bandwidth_dict = {'NSA' : {}, 'SA' : {}}
main_tech_xput_ul_ca_bandwidth_dict = {'NSA' : {}, 'SA' : {}}

main_tech_xput_dl_mean_dict = {'5G (NSA)' : [], '5G (SA)' : []}
main_tech_xput_dl_std_dict = {'5G (NSA)' : [], '5G (SA)' : []}
main_tech_xput_ul_mean_dict = {'5G (NSA)' : [], '5G (SA)' : []}
main_tech_xput_ul_std_dict = {'5G (NSA)' : [], '5G (SA)' : []}

main_tech_xput_dl_diff_dict = {'5G (NSA)' : {0.5 : [], 2 : [], 4: []}, '5G (SA)' : {0.5 : [], 2 : [], 4: []}}
main_tech_xput_ul_diff_dict = {'5G (NSA)' : {0.5 : [], 2 : [], 4: []}, '5G (SA)' : {0.5 : [], 2 : [], 4: []}}

lat_lon_count = -1

if os.path.exists('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/pkls/driving_trip_lax_bos_2024/city_tech_info.pkl'):
    fh = open("/home/moinakgh/csv_ho/nsa_sa_analysis_perf/pkls/driving_trip_lax_bos_2024/city_tech_info.pkl", "rb")
    city_tech_df = pkl.load(fh)
    fh.close()
    city_tech_df['Lat-Lon'] = city_tech_df.apply(lambda row: (row['Lat'], row['Lon']), axis=1)

city_tech_df = city_tech_df.drop_duplicates(subset='Lat-Lon')
lat_lon_city_dict = city_tech_df.set_index('Lat-Lon')['city_info'].to_dict()


def extract_sa_nsa_coverage():

    def return_city_info(lat_lon):
        global lat_lon_city_dict
        if lat_lon not in lat_lon_city_dict.keys():
            return None
        else:
            return lat_lon_city_dict[lat_lon]
        
    def extract_drive_trip_data_lax_bos():
        global main_lat_lon_tech_df
        global main_sa_nsa_lte_tech_time_dict
        global main_sa_nsa_lte_tech_dist_dict
        global main_sa_nsa_lte_tech_dist_dict_tz
        global main_sa_nsa_lte_tech_dist_dict_city
        global main_sa_nsa_lte_tech_dist_dict_chic_bos
        global main_sa_nsa_lte_tech_band_dict
        global main_sa_nsa_lte_tech_ca_dict
        global main_sa_nsa_lte_tech_ca_ul_dict
        global main_sa_nsa_lte_tech_ca_band_combo_dict
        global main_sa_nsa_lte_tech_ca_ul_band_combo_dict
        global main_sa_nsa_tx_power_dict
        global main_sa_nsa_tx_power_control_dict
        global main_sa_nsa_rx_power_dict
        global main_sa_nsa_pathloss_dict
        global main_sa_nsa_lte_tech_dl_mimo_dict
        global main_sa_nsa_lte_tech_dl_mimo_layer_dict
        global main_sa_nsa_lte_tech_city_ca_dict
        global main_sa_nsa_lte_tech_city_ca_ul_dict

        drive_trip_data_path = "../raw_data/data_2024/xcal_kpi_data"    

        for day in ['day_5', 'day_6', 'day_7', 'day_8']:
            tmobile_template_data = glob.glob(drive_trip_data_path + "/*tmobile*%s*xlsx" %day)
            for template_csv in tmobile_template_data:
                df_template = pd.read_excel(template_csv)

                df_template.drop(df_template.tail(8).index,inplace=True)
                df_template['TIME_STAMP'] = df_template['TIME_STAMP'].apply(datetime_to_timestamp)
                df_template = df_template.rename(columns={'TIME_STAMP' : 'Timestamp'})
                df_template = df_template.sort_values("Timestamp").reset_index(drop=True)

                df_template['Timestamp_PD'] = pd.to_datetime(df_template['Timestamp'], unit='s', errors='coerce')

                if day == 'day_5':
                    # Get the timestamp of the last row
                    last_timestamp = df_template['Timestamp'].iloc[-1]

                    # Create a subset of the dataframe where TIME_STAMP > last_timestamp - 3600
                    df_template = df_template[df_template['Timestamp'] > (last_timestamp - 3600)]

                # List of Unix timestamps to iterate over
                # Assuming unix_timestamp_list is already in Unix format
                # get app times 
                app_base = "../raw_data/data_2024/app_data/2024110%s" %day.split("_")[-1]
                app_folders = glob.glob(app_base + "/*")

                start_end_times = []
                for app_folder in app_folders:
                    if '.log' in app_folder:
                        continue 
                    
                    dl_temp_times = []
                    traceroute_temp_times = []
                    app_data_logs = sorted(glob.glob(app_folder + "/*.out"))
                    for app_data in app_data_logs:
                        if 'downlink' in app_data:
                            fh = open(app_data, "r")
                            data = fh.readlines()
                            if day in ['day_1', 'day_2']:
                                start = (int(data[0].split(":")[-1].strip()) - 3600000  ) / 1000          
                                try:
                                    end =   (int(data[-1].split(":")[-1].strip())  - 3600000) / 1000   
                                except:
                                    if len(dl_temp_times) == 0:
                                        end = start + 121
                                    else:
                                        end = start + 6
                            else: 
                                start = int(data[0].split(":")[-1].strip()) / 1000     
                                try:                            
                                    end = int(data[-1].split(":")[-1].strip()) / 1000   
                                except:
                                    if len(dl_temp_times) == 0:
                                        end = start + 121
                                    else:
                                        end = start + 6
                            fh.close()
                            dl_temp_times.extend([start, end])
                        elif 'traceroute' in app_data:
                            fh = open(app_data, "r")
                            data = fh.readlines()
                            if day in ['day_1', 'day_2']:
                                start = (int(data[0].split(":")[-1].strip()) - 3600000) / 1000    
                                try:                             
                                    end = (int(data[-1].split()[-1]) - 3600000) / 1000      
                                except:
                                    end = start + 30
                            else:      
                                start = int(data[0].split(":")[-1].strip()) / 1000     
                                try:                            
                                    end = int(data[-1].split()[-1]) / 1000                     
                                except:
                                    end = start + 30   
                            fh.close()
                            traceroute_temp_times.extend([start, end])
                    temp_times = dl_temp_times.copy()
                    temp_times.extend(traceroute_temp_times)
                    temp_times = sorted(temp_times)
                    # start_end_times.append(temp_times[0])
                    start_end_times.append((temp_times[0], temp_times[-1]))


                for start_end in start_end_times:
                    start, end = start_end
                    subframe = df_template[(df_template['Timestamp'] >= start) & (df_template['Timestamp'] <= end)]
                    # Slice the dataframe from start_idx to end_idx (not including end_idx)
                    # subframe = df_template.iloc[start_idx:end_idx - 1]
                    
                    technologies = list(set(subframe['Event Technology'].dropna()))
                    for technology in technologies:
                        if 'LTE' in technology:
                            subframe['Event Technology'] = subframe['Event Technology'].replace(technology, 'LTE')
                        elif 'NSA' in technology:
                            subframe['Event Technology'] = subframe['Event Technology'].replace(technology, '5G (NSA)')
                        elif '5G' in technology and 'NSA' not in technology:
                            subframe['Event Technology'] = subframe['Event Technology'].replace(technology, '5G (SA)')
                        else:
                            subframe['Event Technology'] = subframe['Event Technology'].replace(technology, 'Others')

                    subframe = subframe[['Timestamp', 'Lat', 'Lon', 'Event Technology', '5G KPI PCell RF Serving SS-RSRP [dBm]', '5G KPI PCell RF PUSCH Power [dBm]', '5G KPI PCell RF PUCCH Power [dBm]', '5G KPI PCell Layer1 DL MIMO', '5G KPI PCell Layer1 DL Layer Num (Mode)', '5G KPI PCell RF Pathloss [dB]']]
                    
                    if len(subframe) == 0:
                        continue
                    tech_subframe = subframe[['Event Technology']].dropna()
                    # Create a column to detect when the technology changes
                    tech_subframe['tech_change'] = tech_subframe['Event Technology'].ne(tech_subframe['Event Technology'].shift()).cumsum()

                    # # Split the dataframe into subframes based on consecutive technology
                    subframe_list = [group for _, group in tech_subframe.groupby('tech_change')]

                    # # Remove the 'tech_change' column from each subframe
                    subframe_list = [subf.drop(columns=['tech_change']) for subf in subframe_list]

                    stop_indices = [sub_df.index[0] for sub_df in subframe_list[1:]]
                    stop_indices.append(subframe.index[-1])
                    for temp_sf, stop_idx in zip(subframe_list, stop_indices):
                        sf = subframe.loc[temp_sf.index[0]:stop_idx]
                        
                        if sf['Event Technology'].iloc[0] not in main_sa_nsa_lte_tech_time_dict.keys():
                            main_sa_nsa_lte_tech_time_dict[sf['Event Technology'].iloc[0]] = []
                            main_sa_nsa_lte_tech_dist_dict[sf['Event Technology'].iloc[0]] = []
                            main_sa_nsa_lte_tech_dist_dict_chic_bos[sf['Event Technology'].iloc[0]] = []
                            main_sa_nsa_lte_tech_band_dict[sf['Event Technology'].iloc[0]] = []
                            main_sa_nsa_lte_tech_ca_dict[sf['Event Technology'].iloc[0]] = []
                            main_sa_nsa_lte_tech_ca_ul_dict[sf['Event Technology'].iloc[0]] = []
                            main_sa_nsa_lte_tech_dl_mimo_dict[sf['Event Technology'].iloc[0]] = []
                            main_sa_nsa_lte_tech_dl_mimo_layer_dict[sf['Event Technology'].iloc[0]] = []
                            main_sa_nsa_lte_tech_city_ca_dict[sf['Event Technology'].iloc[0]] = {}
                            main_sa_nsa_lte_tech_city_ca_ul_dict[sf['Event Technology'].iloc[0]] = {}
                        
                        if '5G' in sf['Event Technology'].iloc[0]:
                            main_sa_nsa_lte_tech_band_dict[sf['Event Technology'].iloc[0]].extend(list(df_template['5G KPI PCell RF Band'].loc[sf.index[0]:sf.index[-1]].dropna()))
                            main_sa_nsa_lte_tech_ca_dict[sf['Event Technology'].iloc[0]].extend(list(df_template['5G KPI Total Info DL CA Type'].loc[sf.index[0]:sf.index[-1]].dropna()))
                            main_sa_nsa_lte_tech_ca_ul_dict[sf['Event Technology'].iloc[0]].extend(list(df_template['5G KPI Total Info UL CA Type'].loc[sf.index[0]:sf.index[-1]].dropna()))

                            '''
                            ca_lat_lon_df = df_template[['Lat', 'Lon', '5G KPI Total Info DL CA Type']].loc[sf.index[0]:sf.index[-1]].dropna()
                            ca_lat_lon_df['Lat-Lon'] = ca_lat_lon_df.apply(lambda row: (row['Lat'], row['Lon']), axis=1)
                            ca_lat_lon_df['city_data'] = ca_lat_lon_df['Lat-Lon'].apply(return_city_data)
                            vegas_filtered_df = ca_lat_lon_df[ca_lat_lon_df['city_data'] == 'vegas']
                            if len(vegas_filtered_df) != 0:
                                if 'vegas' not in main_sa_nsa_lte_tech_city_ca_dict[sf['Event Technology'].iloc[0]].keys():
                                    main_sa_nsa_lte_tech_city_ca_dict[sf['Event Technology'].iloc[0]]['vegas'] = []
                                main_sa_nsa_lte_tech_city_ca_dict[sf['Event Technology'].iloc[0]]['vegas'].extend(list(vegas_filtered_df['5G KPI Total Info DL CA Type']))

                            main_sa_nsa_lte_tech_dl_mimo_dict[sf['Event Technology'].iloc[0]].extend(list(df_template['5G KPI PCell Layer1 DL MIMO'].loc[sf.index[0]:sf.index[-1]].dropna()))
                            main_sa_nsa_lte_tech_dl_mimo_layer_dict[sf['Event Technology'].iloc[0]].extend(list(df_template['5G KPI PCell Layer1 DL Layer Num (Mode)'].loc[sf.index[0]:sf.index[-1]].dropna()))

                            '''

                            ca_band_df = df_template[['5G KPI PCell RF Band', '5G KPI SCell[1] RF Band', '5G KPI SCell[2] RF Band', '5G KPI SCell[3] RF Band', '5G KPI Total Info DL CA Type']].loc[sf.index[0]:sf.index[-1]]
                            ca_band_df = ca_band_df.dropna(subset=['5G KPI Total Info DL CA Type', '5G KPI PCell RF Band'])
                            ca_band_df = ca_band_df.fillna(0)

                            # Group by 'Category'
                            grouped_df = ca_band_df.groupby('5G KPI Total Info DL CA Type')

                            # Iterate over the grouped DataFrames
                            for group_name, group_df in grouped_df:
                                if group_name not in main_sa_nsa_lte_tech_ca_band_combo_dict[sf['Event Technology'].iloc[0]].keys():
                                    main_sa_nsa_lte_tech_ca_band_combo_dict[sf['Event Technology'].iloc[0]][group_name] = []

                                if 'NonCA' in group_name:
                                    main_sa_nsa_lte_tech_ca_band_combo_dict[sf['Event Technology'].iloc[0]][group_name].extend(list(group_df['5G KPI PCell RF Band']))
                                elif '2CA' in group_name:
                                    main_sa_nsa_lte_tech_ca_band_combo_dict[sf['Event Technology'].iloc[0]][group_name].extend(list(group_df['5G KPI PCell RF Band'].astype(str) + ":" + group_df['5G KPI SCell[1] RF Band'].astype(str)))
                                elif '3CA' in group_name:
                                    main_sa_nsa_lte_tech_ca_band_combo_dict[sf['Event Technology'].iloc[0]][group_name].extend(list(group_df['5G KPI PCell RF Band'].astype(str) + ":" + group_df['5G KPI SCell[1] RF Band'].astype(str) + ":" + group_df['5G KPI SCell[2] RF Band'].astype(str))) 
                                elif '4CA' in group_name:
                                    main_sa_nsa_lte_tech_ca_band_combo_dict[sf['Event Technology'].iloc[0]][group_name].extend(list(group_df['5G KPI PCell RF Band'].astype(str) + ":" + group_df['5G KPI SCell[1] RF Band'].astype(str) + ":" + group_df['5G KPI SCell[2] RF Band'].astype(str) + ":" + group_df['5G KPI SCell[3] RF Band'].astype(str)))

                            # UL CA
                            ca_ul_band_df = df_template[['5G KPI PCell RF Band', '5G KPI SCell[1] RF Band', '5G KPI Total Info UL CA Type']].loc[sf.index[0]:sf.index[-1]]
                            ca_ul_band_df = ca_ul_band_df.dropna(subset=['5G KPI Total Info UL CA Type', '5G KPI PCell RF Band'])
                            ca_ul_band_df = ca_ul_band_df.fillna(0)

                            # Group by 'Category'
                            grouped_df = ca_ul_band_df.groupby('5G KPI Total Info UL CA Type')

                            # Iterate over the grouped DataFrames
                            for group_name, group_df in grouped_df:
                                if group_name not in main_sa_nsa_lte_tech_ca_ul_band_combo_dict[sf['Event Technology'].iloc[0]].keys():
                                    main_sa_nsa_lte_tech_ca_ul_band_combo_dict[sf['Event Technology'].iloc[0]][group_name] = []

                                if 'NonCA' in group_name:
                                    main_sa_nsa_lte_tech_ca_ul_band_combo_dict[sf['Event Technology'].iloc[0]][group_name].extend(list(group_df['5G KPI PCell RF Band']))
                                elif '2CA' in group_name:
                                    main_sa_nsa_lte_tech_ca_ul_band_combo_dict[sf['Event Technology'].iloc[0]][group_name].extend(list(group_df['5G KPI PCell RF Band'].astype(str) + ":" + group_df['5G KPI SCell[1] RF Band'].astype(str)))

                            # add all the tx and rx power data
                            main_sa_nsa_tx_power_dict[sf['Event Technology'].iloc[0]].extend(list(sf['5G KPI PCell RF PUSCH Power [dBm]'].dropna()))
                            main_sa_nsa_tx_power_control_dict[sf['Event Technology'].iloc[0]].extend(list(sf['5G KPI PCell RF PUCCH Power [dBm]'].dropna()))
                            main_sa_nsa_rx_power_dict[sf['Event Technology'].iloc[0]].extend(list(sf['5G KPI PCell RF Serving SS-RSRP [dBm]'].dropna()))
                            main_sa_nsa_pathloss_dict[sf['Event Technology'].iloc[0]].extend(list(sf['5G KPI PCell RF Pathloss [dB]'].dropna()))
                            # add all the tx and rx power data
                            path_loss_df = sf[['Lat', 'Lon', '5G KPI PCell RF Pathloss [dB]', '5G KPI PCell RF PUSCH Power [dBm]', '5G KPI PCell RF PUCCH Power [dBm]', '5G KPI PCell RF Serving SS-RSRP [dBm]']]
                            path_loss_df['Lat-Lon'] = path_loss_df.apply(lambda row: (row['Lat'], row['Lon']), axis=1)
                            path_loss_df['city_info'] = path_loss_df['Lat-Lon'].apply(return_city_info)
                            path_loss_df['city_info'] = path_loss_df['city_info'].fillna(method='ffill').fillna(method='bfill')
                            # grouped = path_loss_df.groupby('city_info')
                            # try:
                            #     big_city_group = grouped.get_group('big-city')
                            #     main_sa_nsa_pathloss_dict[sf['Event Technology'].iloc[0]].extend(list(big_city_group['5G KPI PCell RF Pathloss [dB]'].dropna()))
                            #     main_sa_nsa_tx_power_dict[sf['Event Technology'].iloc[0]].extend(list(big_city_group['5G KPI PCell RF PUSCH Power [dBm]'].dropna()))
                            #     main_sa_nsa_tx_power_control_dict[sf['Event Technology'].iloc[0]].extend(list(big_city_group['5G KPI PCell RF PUCCH Power [dBm]'].dropna()))
                            #     main_sa_nsa_rx_power_dict[sf['Event Technology'].iloc[0]].extend(list(big_city_group['5G KPI PCell RF Serving SS-RSRP [dBm]'].dropna()))
                            # except:
                            #     pass 


                        main_sa_nsa_lte_tech_time_dict[sf['Event Technology'].iloc[0]].append(sf['Timestamp'].iloc[-1] - sf['Timestamp'].iloc[0])

                        diff_ts = []
                        prev_ts = sf['Timestamp'].iloc[0]
                        for ts in sf['Timestamp']:
                            diff_ts.append(ts - prev_ts)
                            prev_ts = ts 
                            
                        prev_lat_lon = [sf['Lat'].iloc[0], sf['Lon'].iloc[0]]
                        sum_dist = 0
                        for lat, lon, ts in zip(sf['Lat'].iloc[1:], sf['Lon'].iloc[1:], diff_ts[1:]):
                            if pd.isnull(lat) or pd.isnull(lon):
                                continue
                            dist_current = geopy.distance.geodesic([lat, lon], prev_lat_lon).miles
                            if ts > 3:
                                sum_dist+=0
                            else:
                                if dist_current > 1:
                                    continue
                                sum_dist+=dist_current
                            prev_lat_lon = [lat, lon]
                        
                        main_sa_nsa_lte_tech_dist_dict[sf['Event Technology'].iloc[0]].append(sum_dist)
                        if '5G' in sf['Event Technology'].iloc[0]:
                            tz_info = get_tz_info(list(sf[['Lat', 'Lon']].median()))
                            if tz_info not in main_sa_nsa_lte_tech_dist_dict_tz[sf['Event Technology'].iloc[0]].keys():
                                main_sa_nsa_lte_tech_dist_dict_tz[sf['Event Technology'].iloc[0]][tz_info] = []
                            main_sa_nsa_lte_tech_dist_dict_tz[sf['Event Technology'].iloc[0]][tz_info].append(sum_dist)

                            city = path_loss_df['city_info'].iloc[len(path_loss_df) // 2]
                            if city not in main_sa_nsa_lte_tech_dist_dict_city[sf['Event Technology'].iloc[0]].keys():
                                main_sa_nsa_lte_tech_dist_dict_city[sf['Event Technology'].iloc[0]][city] = []
                            main_sa_nsa_lte_tech_dist_dict_city[sf['Event Technology'].iloc[0]][city].append(sum_dist)

                        if day in ['day_6', 'day_7', 'day_8']:
                            main_sa_nsa_lte_tech_dist_dict_chic_bos[sf['Event Technology'].iloc[0]].append(sum_dist)
                            
                        

                lat_lon_tech_df = df_template[['Timestamp', 'Lat', 'Lon', 'Event Technology']].dropna()

                lat_lon_tech_df['Event Technology'] = lat_lon_tech_df['Event Technology'].replace('5G-NR', '5G-NR_SA')
                lat_lon_tech_df['Event Technology'] = lat_lon_tech_df['Event Technology'].replace('5G-NR(2CA)', '5G-NR_SA')
                lat_lon_tech_df['Event Technology'] = lat_lon_tech_df['Event Technology'].replace('5G-NR(3CA)', '5G-NR_SA')
                lat_lon_tech_df['Event Technology'] = lat_lon_tech_df['Event Technology'].replace('5G-NR(4CA)', '5G-NR_SA')

                technologies = list(set(lat_lon_tech_df['Event Technology']))
                for technology in technologies:
                    if 'LTE' in technology:
                        lat_lon_tech_df['Event Technology'] = lat_lon_tech_df['Event Technology'].replace(technology, 'LTE')
                    elif 'NSA' in technology:
                        lat_lon_tech_df['Event Technology'] = lat_lon_tech_df['Event Technology'].replace(technology, '5G (NSA)')
                    elif '5G' in technology and 'NSA' not in technology:
                        lat_lon_tech_df['Event Technology'] = lat_lon_tech_df['Event Technology'].replace(technology, '5G (SA)')
                    else:
                        lat_lon_tech_df['Event Technology'] = lat_lon_tech_df['Event Technology'].replace(technology, 'Others')
                
                main_lat_lon_tech_df = pd.concat([main_lat_lon_tech_df, lat_lon_tech_df])


    sub_cities_df = pd.read_csv('../raw_data/us_cities_of_interest.csv')
    def extrapolate_road_data_new(original_lat_lon):
        global lat_lon_count
        lat_lon_count+=1
        print("Parsing count = %s" %str(lat_lon_count))
        distance_from_city = []
        for lat_lng in list(sub_cities_df['lat-lon']):
            lat_lng = [float(i) for i in lat_lng.strip("()").split(",")]
            distance_from_city.append(geodesic(lat_lng, original_lat_lon).miles)
        if min(distance_from_city) <= 5:
            return 'big-city'
        elif min(distance_from_city) > 5 and min(distance_from_city) <= 10:
            return 'unclassified'
        else:
            return 'not-big-city'


    if 1:
        extract_drive_trip_data_lax_bos()

        # save all data to pkl
        global main_lat_lon_tech_df, main_sa_nsa_lte_tech_time_dict, main_sa_nsa_lte_tech_dist_dict, main_sa_nsa_lte_tech_dist_dict_tz, main_sa_nsa_lte_tech_dist_dict_city, main_sa_nsa_lte_tech_dist_dict_chic_bos, main_sa_nsa_lte_tech_band_dict, main_sa_nsa_lte_tech_ca_dict, main_sa_nsa_lte_tech_ca_ul_dict, main_sa_nsa_lte_tech_ca_band_combo_dict, main_sa_nsa_lte_tech_ca_ul_band_combo_dict, main_sa_nsa_tx_power_dict, main_sa_nsa_tx_power_control_dict, main_sa_nsa_rx_power_dict, main_sa_nsa_pathloss_dict, main_sa_nsa_lte_tech_dl_mimo_dict, main_sa_nsa_lte_tech_dl_mimo_layer_dict, main_sa_nsa_lte_tech_city_ca_dict
        fh = open("../pkls/driving_trip_lax_bos_2024/coverage_data.pkl", "wb")
        pkl.dump([main_lat_lon_tech_df, main_sa_nsa_lte_tech_time_dict, main_sa_nsa_lte_tech_dist_dict, main_sa_nsa_lte_tech_dist_dict_tz, main_sa_nsa_lte_tech_dist_dict_city, main_sa_nsa_lte_tech_dist_dict_chic_bos, main_sa_nsa_lte_tech_band_dict, main_sa_nsa_lte_tech_ca_dict, main_sa_nsa_lte_tech_ca_ul_dict, main_sa_nsa_lte_tech_ca_band_combo_dict, main_sa_nsa_lte_tech_ca_ul_band_combo_dict, main_sa_nsa_tx_power_dict, main_sa_nsa_tx_power_control_dict, main_sa_nsa_rx_power_dict, main_sa_nsa_pathloss_dict, main_sa_nsa_lte_tech_dl_mimo_dict, main_sa_nsa_lte_tech_dl_mimo_layer_dict, main_sa_nsa_lte_tech_city_ca_dict], fh)
        fh.close()

        main_lat_lon_tech_df['Lat-Lon'] = main_lat_lon_tech_df.apply(lambda row: (row['Lat'], row['Lon']), axis=1)
        # if not os.path.exists("/home/moinakgh/csv_ho/nsa_sa_analysis_perf/pkls/driving_trip_lax_bos_2024/city_tech_info.pkl"):
        if 1:
            main_lat_lon_tech_df['city_info'] = main_lat_lon_tech_df['Lat-Lon'].apply(extrapolate_road_data_new)
            city_tech_df = main_lat_lon_tech_df[['Lat', 'Lon', 'city_info', 'Event Technology']].dropna()
            fh = open("../pkls/driving_trip_lax_bos_2024/city_tech_info.pkl", "wb")
            city_tech_df = pkl.dump(city_tech_df, fh)
            fh.close()



def extract_overall_xput_dist():


    def extract_drive_trip_data_lax_bos():
        global main_xput_tech_dl_dict
        global main_xput_tech_ul_dict

        drive_trip_data_path = "../raw_data/data_2024/xcal_kpi_data"    


        for day in ['day_5', 'day_6', 'day_7', 'day_8']:
            tmobile_template_data = glob.glob(drive_trip_data_path + "/*tmobile*%s*xlsx" %day)
            for template_csv in tmobile_template_data:
                df_template = pd.read_excel(template_csv)

                df_template.drop(df_template.tail(8).index,inplace=True)
                df_template['TIME_STAMP'] = df_template['TIME_STAMP'].apply(datetime_to_timestamp)
                df_template = df_template.rename(columns={'TIME_STAMP' : 'Timestamp'})
                df_template = df_template.sort_values("Timestamp").reset_index(drop=True)

                if day == 'day_5':
                    # Get the timestamp of the last row
                    last_timestamp = df_template['Timestamp'].iloc[-1]

                    # Create a subset of the dataframe where TIME_STAMP > last_timestamp - 3600
                    df_template = df_template[df_template['Timestamp'] > (last_timestamp - 3600)]

                # get app times 
                app_base = "../raw_data/data_2024/app_data/2024110%s" %day.split("_")[-1]
                app_folders = sorted(glob.glob(app_base + "/*"))

                downlink_start_end_times = []
                uplink_start_end_times = []
                for app_folder in app_folders:
                    if '.log' in app_folder:
                        continue 
                    
                    dl_temp_times = []
                    ul_temp_times = []
                    app_data_logs = sorted(glob.glob(app_folder + "/*.out"))
                    for app_data in app_data_logs:
                        if 'downlink' in app_data:
                            fh = open(app_data, "r")
                            data = fh.readlines()
                            if day in ['day_1', 'day_2']:
                                start = (int(data[0].split(":")[-1].strip()) - 3600000  ) / 1000      
                                end = start + (get_nuttcp_data_length(data)) * 0.5    

                            else: 
                                start = int(data[0].split(":")[-1].strip()) / 1000     
                                end = start + (get_nuttcp_data_length(data)) * 0.5    

                            fh.close()
                            dl_temp_times.extend([start, end])
                        elif 'uplink' in app_data:
                            fh = open(app_data, "r")
                            data = fh.readlines()
                            if day in ['day_1', 'day_2']:
                                start = (int(data[0].split(":")[-1].strip()) - 3600000  ) / 1000          
                                end = start + (get_nuttcp_data_length(data)) * 0.5    

                            else: 
                                start = int(data[0].split(":")[-1].strip()) / 1000     
                                end = start + (get_nuttcp_data_length(data)) * 0.5    

                            fh.close()
                            ul_temp_times.extend([start, end])
                    dl_temp_times = sorted(dl_temp_times)
                    downlink_start_end_times.append((dl_temp_times[0], dl_temp_times[1]))
                    ul_temp_times = sorted(ul_temp_times)
                    try:
                        uplink_start_end_times.append((ul_temp_times[0], ul_temp_times[1]))
                    except:
                        # no uplink
                        pass 

                # work with downlink first
                print("Processing df_template: %s" %template_csv)
                for start_end_time in downlink_start_end_times:
                    start_time, end_time = start_end_time
                    sub_df = get_start_end_indices(df_template, start_time, end_time)
                    if len(sub_df) == 0 or len(sub_df['Event Technology'].dropna()) == 0:
                        continue
                    sub_df = sub_df.loc[sub_df['Event Technology'].dropna().index[0]:]
                    sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR', '5G-NR_SA')
                    sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR(2CA)', '5G-NR_SA')
                    sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR(3CA)', '5G-NR_SA')
                    sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR(4CA)', '5G-NR_SA')
                    

                    # Step 1: Identify rows where LTE Event or 5G Event match any event in possible_ho
                    try:
                        ho_rows = sub_df[(sub_df['Event LTE Events'].isin(possible_ho)) | (sub_df['Event 5G-NR Events'].isin(possible_ho))]
                    except:
                        ho_rows = sub_df[(sub_df['Event 5G-NR Events'].isin(possible_ho))]


                    # Step 2: For each ho_row, find the rows within a 600 ms window after the Timestamp
                    for ho_index, ho_row in ho_rows.iterrows():
                        ho_timestamp = ho_row['Timestamp']
                        # Identify rows within 600 ms after the ho_row's timestamp
                        window_rows = sub_df[(sub_df['Timestamp'] > ho_timestamp) & (sub_df['Timestamp'] <= ho_timestamp + 0.6)]
                        
                        # Step 3: Update the Throughput column of these rows to 'Skip'
                        # sub_df.loc[window_rows.index, 'Smart Phone Smart Throughput Mobile Network DL Throughput [Mbps]'] = 'Skip'
                        # First, identify the rows where the column is non-null
                        non_null_mask = sub_df.loc[window_rows.index, 'Smart Phone Smart Throughput Mobile Network DL Throughput [Mbps]'].notna()

                        # Replace only those non-null entries with 'Skip'
                        sub_df.loc[window_rows.index[non_null_mask], 'Smart Phone Smart Throughput Mobile Network DL Throughput [Mbps]'] = 'Skip'

                    # sub_df[['Timestamp', 'Event Technology', 'Event LTE Events', 'Event 5G-NR Events', 'Smart Phone Smart Throughput Mobile Network DL Throughput [Mbps]']].to_csv('test.csv')
                    sub_df['Event Technology'] = sub_df['Event Technology'].fillna(method='ffill')
                    xput_data = sub_df[['Event Technology', 'Smart Phone Smart Throughput Mobile Network DL Throughput [Mbps]']].dropna()
                    for xput_tech, xput in zip(xput_data['Event Technology'], xput_data['Smart Phone Smart Throughput Mobile Network DL Throughput [Mbps]']):
                        if xput == 'Skip':
                            main_xput_tech_dl_dict['Skip']+=1
                        else:
                            if 'NSA' in xput_tech:
                                xput_tech = 'NSA'
                            elif '_SA' in xput_tech:
                                xput_tech = 'SA'
                            elif 'LTE' in xput_tech:
                                xput_tech = 'LTE'
                            if xput_tech not in main_xput_tech_dl_dict.keys():
                                main_xput_tech_dl_dict[xput_tech] = []
                            main_xput_tech_dl_dict[xput_tech].append(xput)

                # work with uplink then 
                for start_end_time in uplink_start_end_times:
                    start_time, end_time = start_end_time
                    sub_df = get_start_end_indices(df_template, start_time, end_time)
                    if len(sub_df) == 0 or len(sub_df['Event Technology'].dropna()) == 0:
                        continue
                    sub_df = sub_df.loc[sub_df['Event Technology'].dropna().index[0]:]
                    sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR', '5G-NR_SA')
                    sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR(2CA)', '5G-NR_SA')
                    sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR(3CA)', '5G-NR_SA')
                    sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR(4CA)', '5G-NR_SA')
                    

                    # Step 1: Identify rows where LTE Event or 5G Event match any event in possible_ho
                    try:
                        ho_rows = sub_df[(sub_df['Event LTE Events'].isin(possible_ho)) | (sub_df['Event 5G-NR Events'].isin(possible_ho))]
                    except:
                        ho_rows = sub_df[(sub_df['Event 5G-NR Events'].isin(possible_ho))]


                    # Step 2: For each ho_row, find the rows within a 600 ms window after the Timestamp
                    for ho_index, ho_row in ho_rows.iterrows():
                        ho_timestamp = ho_row['Timestamp']
                        # Identify rows within 600 ms after the ho_row's timestamp
                        window_rows = sub_df[(sub_df['Timestamp'] > ho_timestamp) & (sub_df['Timestamp'] <= ho_timestamp + 0.6)]
                        
                        # Step 3: Update the Throughput column of these rows to 'Skip'
                        # sub_df.loc[window_rows.index, 'Smart Phone Smart Throughput Mobile Network UL Throughput [Mbps]'] = 'Skip'
                        # First, identify the rows where the column is non-null
                        non_null_mask = sub_df.loc[window_rows.index, 'Smart Phone Smart Throughput Mobile Network UL Throughput [Mbps]'].notna()

                        # Replace only those non-null entries with 'Skip'
                        sub_df.loc[window_rows.index[non_null_mask], 'Smart Phone Smart Throughput Mobile Network UL Throughput [Mbps]'] = 'Skip'

                    # sub_df[['Timestamp', 'Event Technology', 'Event LTE Events', 'Event 5G-NR Events', 'Smart Phone Smart Throughput Mobile Network UL Throughput [Mbps]']].to_csv('test.csv')
                    sub_df['Event Technology'] = sub_df['Event Technology'].fillna(method='ffill')
                    xput_data = sub_df[['Event Technology', 'Smart Phone Smart Throughput Mobile Network UL Throughput [Mbps]']].dropna()
                    for xput_tech, xput in zip(xput_data['Event Technology'], xput_data['Smart Phone Smart Throughput Mobile Network UL Throughput [Mbps]']):
                        if xput == 'Skip':
                            main_xput_tech_ul_dict['Skip']+=1
                        else:
                            if 'NSA' in xput_tech:
                                xput_tech = 'NSA'
                            elif '_SA' in xput_tech:
                                xput_tech = 'SA'
                            elif 'LTE' in xput_tech:
                                xput_tech = 'LTE'
                            if xput_tech not in main_xput_tech_ul_dict.keys():
                                main_xput_tech_ul_dict[xput_tech] = []
                            main_xput_tech_ul_dict[xput_tech].append(xput)
        

    extract_drive_trip_data_lax_bos()

    fh = open("../pkls/driving_trip_lax_bos_2024/overal_xput.pkl", "wb")
    pkl.dump([main_xput_tech_dl_dict, main_xput_tech_ul_dict], fh)
    fh.close()

def extract_5G_lte_break_xput_dist_conext():

    def return_city_info(lat_lon):
        global lat_lon_city_dict
        if lat_lon not in lat_lon_city_dict.keys():
            return None
        else:
            return lat_lon_city_dict[lat_lon]

    # last_timestamp
    # 1730853188.3

    def extract_drive_trip_data_lax_bos():
        global fiveg_xput_tech_dl_dict
        global lte_xput_tech_dl_dict
        global fiveg_xput_tech_ul_dict
        global lte_xput_tech_ul_dict
        global main_sa_nsa_lte_tech_ca_band_xput_dict
        global main_sa_nsa_lte_tech_ca_ul_band_xput_dict
        global main_tech_xput_city_info_dict 
        global main_tech_xput_tx_power_dict  
        global main_tech_xput_tx_power_control_dict  
        global main_tech_xput_dl_tx_power_dict  
        global main_tech_xput_dl_tx_power_control_dict  
        global main_tech_xput_ul_tx_power_dict  
        global main_tech_xput_ul_tx_power_control_dict  
        global main_tech_xput_rx_power_dict  
        global main_tech_xput_dl_rx_power_dict  
        global main_tech_xput_ul_rx_power_dict  
        global main_tech_xput_pathloss_dict  
        global main_tech_xput_dl_pathloss_dict  
        global main_tech_xput_ul_pathloss_dict 
        global main_tech_xput_dl_bandwidth_dict
        global main_tech_xput_ul_bandwidth_dict
        global main_tech_xput_dl_ca_bandwidth_dict
        global main_tech_xput_ul_ca_bandwidth_dict

        global main_tech_xput_dl_mean_dict 
        global main_tech_xput_dl_std_dict  
        global main_tech_xput_ul_mean_dict 
        global main_tech_xput_ul_std_dict  

        global main_tech_xput_dl_diff_dict 
        global main_tech_xput_ul_diff_dict 

        # get the tput  test times
        drive_trip_data_path = "../raw_data/data_2024/xcal_kpi_data"    

        # dl pickle 
        fh = open("../pkls/driving_trip_lax_bos_2024/performance/2024_op_df_list/with_server/tmobile_dl.pkl", "rb")
        dl_df_list = pkl.load(fh)
        fh.close()

        # work with downlink first
        for sub_df in dl_df_list:
            if len(sub_df) == 0 or len(sub_df['Event Technology'].dropna()) == 0:
                continue
            if sub_df['Timestamp'].iloc[-1] < 1730853188.3:
                continue
            sub_df = sub_df.loc[sub_df['Event Technology'].dropna().index[0]:]
            sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR', '5G-NR_SA')
            sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR(2CA)', '5G-NR_SA')
            sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR(3CA)', '5G-NR_SA')
            sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR(4CA)', '5G-NR_SA')
            

            # Step 1: Identify rows where LTE Event or 5G Event match any event in possible_ho
            try:
                ho_rows = sub_df[(sub_df['Event LTE Events'].isin(possible_ho)) | (sub_df['Event 5G-NR Events'].isin(possible_ho))]
            except:
                ho_rows = sub_df[(sub_df['Event 5G-NR Events'].isin(possible_ho))]

            # Step 2: For each ho_row, find the rows within a 600 ms window after the Timestamp
            for ho_index, ho_row in ho_rows.iterrows():
                ho_timestamp = ho_row['Timestamp']
                # Identify rows within 600 ms after the ho_row's timestamp
                window_rows = sub_df[(sub_df['Timestamp'] > ho_timestamp) & (sub_df['Timestamp'] <= ho_timestamp + 0.11)]
                
                # Step 3: Update the Throughput column of these rows to 'Skip'
                # First, identify the rows where the column is non-null
                non_null_mask = sub_df.loc[window_rows.index, '5G KPI Total Info Layer2 MAC DL Throughput [Mbps]'].notna()

                # Replace only those non-null entries with 'Skip'
                sub_df.loc[window_rows.index[non_null_mask], '5G KPI Total Info Layer2 MAC DL Throughput [Mbps]'] = 'Skip'

            sub_df['Event Technology'] = sub_df['Event Technology'].fillna(method='ffill')
            sub_df['Lat-Lon'] = sub_df.apply(lambda row: (row['Lat'], row['Lon']), axis=1)
            sub_df['city_info'] = sub_df['Lat-Lon'].apply(return_city_info)
            sub_df['city_info'] = sub_df['city_info'].fillna(method='ffill').fillna(method='bfill')
            # get 5G MAC xput

            # get 5G MAC xput
            mac_cell_xput_cols = [i for i in sub_df.columns if 'MAC DL Throughput' in i and '5G' in i and 'Total' not in i]
            bw_cols = [i for i in sub_df.columns if 'RF BandWidth' in i and '5G' in i]
            band_cols = [i for i in sub_df.columns if 'RF Band' in i and '5G' in i and 'RF BandWidth' not in i]

            all_cols = ['Lat', 'Lon', 'Timestamp',  'city_info', 'Event Technology', '5G KPI Total Info Layer2 MAC DL Throughput [Mbps]', '5G KPI Total Info DL CA Type', '5G KPI PCell RF Serving SS-RSRP [dBm]', '5G KPI PCell RF PUSCH Power [dBm]', '5G KPI PCell RF PUCCH Power [dBm]', '5G KPI PCell RF Pathloss [dB]', 'Smart Phone Smart Throughput Mobile Network DL Throughput [Mbps]']
            all_cols.extend(mac_cell_xput_cols)
            all_cols.extend(bw_cols)
            all_cols.extend(band_cols)
            xput_data = sub_df[all_cols]
            xput_data = xput_data.dropna(subset=['Event Technology', '5G KPI Total Info Layer2 MAC DL Throughput [Mbps]'])

            # xput_data = sub_df[['Lat', 'Lon', 'Timestamp', 'city_info', 'Event Technology', '5G KPI Total Info Layer2 MAC DL Throughput [Mbps]', '5G KPI PCell RF Band', '5G KPI SCell[1] RF Band', '5G KPI SCell[2] RF Band', '5G KPI SCell[3] RF Band', '5G KPI Total Info DL CA Type', '5G KPI PCell RF Serving SS-RSRP [dBm]', '5G KPI PCell RF PUSCH Power [dBm]', '5G KPI PCell RF PUCCH Power [dBm]', '5G KPI PCell RF BandWidth', '5G KPI SCell[1] RF BandWidth', '5G KPI SCell[2] RF BandWidth', '5G KPI SCell[3] RF BandWidth', '5G KPI PCell RF Pathloss [dB]', 'Smart Phone Smart Throughput Mobile Network DL Throughput [Mbps]']]
            # xput_data = xput_data.dropna(subset=['Event Technology', '5G KPI Total Info Layer2 MAC DL Throughput [Mbps]'])

            if 1:
                xput_data_mod = xput_data.copy()
                technologies = list(set(xput_data_mod['Event Technology'].dropna()))
                for technology in technologies:
                    if 'LTE' in technology:
                        xput_data_mod['Event Technology'] = xput_data_mod['Event Technology'].replace(technology, 'LTE')
                    elif 'NSA' in technology:
                        xput_data_mod['Event Technology'] = xput_data_mod['Event Technology'].replace(technology, '5G (NSA)')
                    elif '5G' in technology and 'NSA' not in technology:
                        xput_data_mod['Event Technology'] = xput_data_mod['Event Technology'].replace(technology, '5G (SA)')
                    else:
                        xput_data_mod['Event Technology'] = xput_data_mod['Event Technology'].replace(technology, 'Others')
                
                if len(xput_data_mod) != 0:
                    tech_subframe = xput_data_mod[['Event Technology', 'Timestamp',  'Smart Phone Smart Throughput Mobile Network DL Throughput [Mbps]']].dropna()
                    # Create a column to detect when the technology changes
                    tech_subframe['tech_change'] = tech_subframe['Event Technology'].ne(tech_subframe['Event Technology'].shift()).cumsum()

                    # # Split the dataframe into subframes based on consecutive technology
                    subframe_list = [group for _, group in tech_subframe.groupby('tech_change')]

                    # # Remove the 'tech_change' column from each subframe
                    subframe_list = [subf.drop(columns=['tech_change']) for subf in subframe_list]

                    if len(subframe_list) > 0:
                        for xput_sub_frame in subframe_list:
                            if '5G' not in xput_sub_frame['Event Technology'].iloc[0]:
                                continue
                            if len(xput_sub_frame['Smart Phone Smart Throughput Mobile Network DL Throughput [Mbps]'].dropna()) < 2:
                                continue
                            # 10 second break
                            if 0: 
                                if xput_sub_frame['Timestamp'].iloc[-1] - xput_sub_frame['Timestamp'].iloc[0] < 9.5:
                                    continue 
                                # Create chunks of 10-second intervals
                                start_time = xput_data_mod['Timestamp'].iloc[0]

                                for _, group in xput_data_mod.groupby((xput_data_mod['Timestamp'] - start_time) // 10):
                                    if group['Timestamp'].iloc[-1] - group['Timestamp'].iloc[0] < 9.5:
                                        continue
                                    main_tech_xput_dl_mean_dict[xput_sub_frame['Event Technology'].iloc[0]].append(np.mean([i for i in group['Smart Phone Smart Throughput Mobile Network DL Throughput [Mbps]'].dropna() if i != 'Skip']))
                                    main_tech_xput_dl_std_dict[xput_sub_frame['Event Technology'].iloc[0]].append(np.std([i for i in group['Smart Phone Smart Throughput Mobile Network DL Throughput [Mbps]'].dropna() if i != 'Skip']))
                            else:
                                xput_list = list(xput_sub_frame['Smart Phone Smart Throughput Mobile Network DL Throughput [Mbps]'].dropna())
                                xput_list_4, xput_list_8 = calculate_averages(xput_list)

                                try:
                                    differences = [xput_list[i+1] - xput_list[i] for i in range(len(xput_list) - 1)]
                                    main_tech_xput_dl_diff_dict[xput_sub_frame['Event Technology'].iloc[0]][0.5].extend(differences)

                                    differences_4 = [xput_list_4[i+1] - xput_list_4[i] for i in range(len(xput_list_4) - 1)]
                                    main_tech_xput_dl_diff_dict[xput_sub_frame['Event Technology'].iloc[0]][2].extend(differences_4)

                                    differences_8 = [xput_list_8[i+1] - xput_list_8[i] for i in range(len(xput_list_8) - 1)]
                                    main_tech_xput_dl_diff_dict[xput_sub_frame['Event Technology'].iloc[0]][4].extend(differences_8)
                                except:
                                    a = 1

            original_list = [xput_data['Timestamp'], xput_data['Lat'], xput_data['Lon'], xput_data['Event Technology'], xput_data['5G KPI Total Info Layer2 MAC DL Throughput [Mbps]'], xput_data['5G KPI Total Info DL CA Type'], xput_data['5G KPI PCell RF Serving SS-RSRP [dBm]'], xput_data['5G KPI PCell RF PUSCH Power [dBm]'], xput_data['5G KPI PCell RF PUCCH Power [dBm]'], xput_data['5G KPI PCell RF Pathloss [dB]'], xput_data['city_info']]
            mac_cell_xput_cols = [i for i in xput_data.columns if 'MAC DL Throughput' in i and '5G' in i and 'Total' not in i]
            cell_list = []
            for col in mac_cell_xput_cols:
                cell_list.append(col.split()[2])
            
            for i in range(3):
                if i == 0:
                    # xput 
                    for cell in cell_list:
                        original_list.append(xput_data['5G KPI %s Layer2 MAC DL Throughput [Mbps]' %cell])
                elif i == 1:
                    # band
                    for cell in cell_list:
                        original_list.append(xput_data['5G KPI %s RF Band' %cell])
                elif i == 2:
                    # bandwidth
                    for cell in cell_list:
                        original_list.append(xput_data['5G KPI %s RF BandWidth' %cell])


            # for lat, lon, xput_tech, xput, ca, band_1, band_2, rx_power, tx_power, tx_power_control, bandwidth_1, bandwidth_2, pathloss, city_info in zip(xput_data['Lat'], xput_data['Lon'], xput_data['Event Technology'], xput_data['5G KPI Total Info Layer2 MAC DL Throughput [Mbps]'], xput_data['5G KPI Total Info DL CA Type'], xput_data['5G KPI PCell RF Band'], xput_data['5G KPI SCell[1] RF Band'], xput_data['5G KPI PCell RF Serving SS-RSRP [dBm]'], xput_data['5G KPI PCell RF PUSCH Power [dBm]'], xput_data['5G KPI PCell RF PUCCH Power [dBm]'], xput_data['5G KPI PCell RF BandWidth'], xput_data['5G KPI SCell[1] RF BandWidth'], xput_data['5G KPI PCell RF Pathloss [dB]'], xput_data['city_info']):
            prev_ts = None
            prev_ca = None 
            prev_band_combo = None
            prev_sum_bandwidth = None
            elem_count = -1
            zipped_list = list(zip(*original_list))
            for elem in zip(*original_list):
                elem_count+=1
                original_data = list(elem)[:11]
                added_data = list(elem)[11:]
                ts, lat, lon, xput_tech, xput, ca, rx_power, tx_power, tx_power_control, pathloss, city_info = original_data

                if xput == 'Skip':
                    fiveg_xput_tech_dl_dict['Skip']+=1
                else:
                    if 'NSA' in xput_tech:
                        xput_tech = 'NSA'
                    elif '_SA' in xput_tech:
                        xput_tech = 'SA'
                    elif 'LTE' in xput_tech:
                        continue
                    if xput_tech not in fiveg_xput_tech_dl_dict.keys():
                        fiveg_xput_tech_dl_dict[xput_tech] = []
                    fiveg_xput_tech_dl_dict[xput_tech].append(xput)

                    dividend = len(added_data) // 3
                    xput_cell_list = []
                    band_cell_list = []
                    bandwidth_cell_list = []
                    for i in range(dividend):
                        xput_cell_list.append(added_data[i])
                        band_cell_list.append(added_data[i + dividend])
                        bandwidth_cell_list.append(added_data[i + 2 * dividend])

                    # override the CA 
                    ca = 0
                    band_combo = ''
                    sum_bandwidth = 0
                    for x, ba, bw in zip(xput_cell_list, band_cell_list, bandwidth_cell_list):
                        if pd.isnull(x):
                            continue
                        if x != 'Skip' and x > 0:
                            ca+=1
                            band_combo += str(ba) + ":"
                            sum_bandwidth += bw
                    # check if previous element is not null
                    if 'nan' in band_combo and prev_ts != None and (ts - prev_ts) <= 0.3 and prev_ca == ca:
                        band_combo = prev_band_combo
                        sum_bandwidth = prev_sum_bandwidth
                    elif 'nan' in band_combo and elem_count + 1 < len(zipped_list) and zipped_list[elem_count + 1][0] != None and (zipped_list[elem_count + 1][0] - ts) <= 0.3:
                        next_added_data = zipped_list[elem_count + 1][11:]
                        next_dividend = len(next_added_data) // 3
                        next_xput_cell_list = []
                        next_band_cell_list = []
                        next_bandwidth_cell_list = []
                        for i in range(next_dividend):
                            next_xput_cell_list.append(next_added_data[i])
                            next_band_cell_list.append(next_added_data[i + next_dividend])
                            next_bandwidth_cell_list.append(next_added_data[i + 2 * next_dividend])

                        # override the CA 
                        next_ca = 0
                        next_band_combo = ''
                        next_sum_bandwidth = 0
                        for x, ba, bw in zip(next_xput_cell_list, next_band_cell_list, next_bandwidth_cell_list):
                            if pd.isnull(x):
                                continue
                            if x != 'Skip' and x > 0:
                                next_ca+=1
                                next_band_combo += str(ba) + ":"
                                next_sum_bandwidth += bw

                        if 'nan' not in next_band_combo:
                            band_combo = next_band_combo
                            sum_bandwidth = next_sum_bandwidth
                    elif 'nan' in band_combo:
                        a = 1

                    prev_ca = ca 
                    prev_band_combo = band_combo
                    prev_sum_bandwidth = sum_bandwidth
                    prev_ts = ts

                    if not pd.isnull(ca):
                        if ca not in main_sa_nsa_lte_tech_ca_band_xput_dict[xput_tech].keys():
                            main_sa_nsa_lte_tech_ca_band_xput_dict[xput_tech][ca] = {}

                        if band_combo not in main_sa_nsa_lte_tech_ca_band_xput_dict[xput_tech][ca].keys():
                            main_sa_nsa_lte_tech_ca_band_xput_dict[xput_tech][ca][band_combo] = []
                        main_sa_nsa_lte_tech_ca_band_xput_dict[xput_tech][ca][band_combo].append(xput)

                        if sum_bandwidth not in main_tech_xput_dl_bandwidth_dict[xput_tech].keys():
                            main_tech_xput_dl_bandwidth_dict[xput_tech][sum_bandwidth] = []
                        main_tech_xput_dl_bandwidth_dict[xput_tech][sum_bandwidth].append(xput)

                        if (ca, band_combo, sum_bandwidth) not in main_tech_xput_dl_ca_bandwidth_dict[xput_tech].keys():
                            main_tech_xput_dl_ca_bandwidth_dict[xput_tech][(ca, band_combo, sum_bandwidth)] = []
                        main_tech_xput_dl_ca_bandwidth_dict[xput_tech][(ca, band_combo, sum_bandwidth)].append(xput)

                    if 1:
                        main_tech_xput_tx_power_dict[xput_tech].append(tx_power)
                        main_tech_xput_tx_power_control_dict[xput_tech].append(tx_power_control)
                        main_tech_xput_dl_tx_power_dict[xput_tech].append(tx_power)
                        main_tech_xput_dl_tx_power_control_dict[xput_tech].append(tx_power_control)
                        main_tech_xput_rx_power_dict[xput_tech].append(rx_power)
                        main_tech_xput_dl_rx_power_dict[xput_tech].append(rx_power)
                        main_tech_xput_pathloss_dict[xput_tech].append(pathloss)
                        main_tech_xput_dl_pathloss_dict[xput_tech].append(pathloss)

            # get LTE MAC xput
            try:
                xput_data = sub_df[['Event Technology', 'LTE KPI MAC DL Throughput [Mbps]']].dropna()
                for xput_tech, xput in zip(xput_data['Event Technology'], xput_data['LTE KPI MAC DL Throughput [Mbps]']):
                    if xput == 'Skip':
                        lte_xput_tech_dl_dict['Skip']+=1
                    else:
                        if 'NSA' in xput_tech:
                            xput_tech = 'NSA'
                        elif '_SA' in xput_tech:
                            xput_tech = 'SA'
                        elif 'LTE' in xput_tech:
                            xput_tech = 'LTE'
                        if xput_tech not in lte_xput_tech_dl_dict.keys():
                            lte_xput_tech_dl_dict[xput_tech] = []
                        lte_xput_tech_dl_dict[xput_tech].append(xput)
            except:
                if xput_tech == 'LTE' or 'NSA' in xput_tech:
                        a = 1
                print("LTE data not present!")

        # ul pickle 
        fh = open("../pkls/driving_trip_lax_bos_2024/performance/2024_op_df_list/with_server/tmobile_ul.pkl", "rb")
        ul_df_list = pkl.load(fh)
        fh.close()

        # work with uplink then 
        for sub_df in ul_df_list:
            if len(sub_df) == 0 or len(sub_df['Event Technology'].dropna()) == 0:
                continue
            if sub_df['Timestamp'].iloc[-1] < 1730853188.3:
                continue
            sub_df = sub_df.loc[sub_df['Event Technology'].dropna().index[0]:]
            sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR', '5G-NR_SA')
            sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR(2CA)', '5G-NR_SA')
            sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR(3CA)', '5G-NR_SA')
            sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR(4CA)', '5G-NR_SA')
            

            # Step 1: Identify rows where LTE Event or 5G Event match any event in possible_ho
            try:
                ho_rows = sub_df[(sub_df['Event LTE Events'].isin(possible_ho)) | (sub_df['Event 5G-NR Events'].isin(possible_ho))]
            except:
                ho_rows = sub_df[(sub_df['Event 5G-NR Events'].isin(possible_ho))]

            # Step 2: For each ho_row, find the rows within a 600 ms window after the Timestamp
            for ho_index, ho_row in ho_rows.iterrows():
                ho_timestamp = ho_row['Timestamp']
                # Identify rows within 600 ms after the ho_row's timestamp
                window_rows = sub_df[(sub_df['Timestamp'] > ho_timestamp) & (sub_df['Timestamp'] <= ho_timestamp + 0.11)]
                
                # Step 3: Update the Throughput column of these rows to 'Skip'
                # First, identify the rows where the column is non-null
                non_null_mask = sub_df.loc[window_rows.index, '5G KPI PCell Layer2 MAC UL Throughput [Mbps]'].notna()

                # Replace only those non-null entries with 'Skip'
                sub_df.loc[window_rows.index[non_null_mask], '5G KPI PCell Layer2 MAC UL Throughput [Mbps]'] = 'Skip'

            sub_df['Event Technology'] = sub_df['Event Technology'].fillna(method='ffill')
            sub_df['Lat-Lon'] = sub_df.apply(lambda row: (row['Lat'], row['Lon']), axis=1)
            sub_df['city_info'] = sub_df['Lat-Lon'].apply(return_city_info)
            sub_df['city_info'] = sub_df['city_info'].fillna(method='ffill').fillna(method='bfill')
            # get 5G MAC xput
            mac_cell_xput_cols = [i for i in sub_df.columns if 'MAC UL Throughput' in i and '5G' in i and 'Total' not in i]
            bw_cols = [i for i in sub_df.columns if 'RF BandWidth' in i and '5G' in i]
            band_cols = [i for i in sub_df.columns if 'RF Band' in i and '5G' in i and 'RF BandWidth' not in i]
            all_cols = ['Lat', 'Lon', 'Timestamp',  'city_info', 'Event Technology', '5G KPI Total Info Layer2 MAC UL Throughput [Mbps]', '5G KPI Total Info UL CA Type', '5G KPI PCell RF Serving SS-RSRP [dBm]', '5G KPI PCell RF PUSCH Power [dBm]', '5G KPI PCell RF PUCCH Power [dBm]', '5G KPI PCell RF Pathloss [dB]', 'Smart Phone Smart Throughput Mobile Network UL Throughput [Mbps]']
            all_cols.extend(mac_cell_xput_cols)
            all_cols.extend(bw_cols)
            all_cols.extend(band_cols)

            xput_data = sub_df[all_cols]
            xput_data = xput_data.dropna(subset=['Event Technology', '5G KPI Total Info Layer2 MAC UL Throughput [Mbps]'])

            # # xput_data = sub_df[['Event Technology', '5G KPI PCell Layer2 MAC UL Throughput [Mbps]']].dropna()
            # # for xput_tech, xput in zip(xput_data['Event Technology'], xput_data['5G KPI PCell Layer2 MAC UL Throughput [Mbps]']):
            # xput_data = sub_df[['Lat', 'Lon', 'Timestamp', 'city_info', 'Event Technology', '5G KPI Total Info Layer2 MAC UL Throughput [Mbps]', '5G KPI PCell RF Band', '5G KPI SCell[1] RF Band', '5G KPI Total Info UL CA Type', '5G KPI PCell RF Serving SS-RSRP [dBm]', '5G KPI PCell RF PUSCH Power [dBm]', '5G KPI PCell RF PUCCH Power [dBm]', '5G KPI PCell RF BandWidth', '5G KPI SCell[1] RF BandWidth', '5G KPI PCell RF Pathloss [dB]', 'Smart Phone Smart Throughput Mobile Network UL Throughput [Mbps]']]
            # xput_data = xput_data.dropna(subset=['Event Technology', '5G KPI Total Info Layer2 MAC UL Throughput [Mbps]'])

            if 1:
                xput_data_mod = xput_data.copy()
                technologies = list(set(xput_data_mod['Event Technology'].dropna()))
                for technology in technologies:
                    if 'LTE' in technology:
                        xput_data_mod['Event Technology'] = xput_data_mod['Event Technology'].replace(technology, 'LTE')
                    elif 'NSA' in technology:
                        xput_data_mod['Event Technology'] = xput_data_mod['Event Technology'].replace(technology, '5G (NSA)')
                    elif '5G' in technology and 'NSA' not in technology:
                        xput_data_mod['Event Technology'] = xput_data_mod['Event Technology'].replace(technology, '5G (SA)')
                    else:
                        xput_data_mod['Event Technology'] = xput_data_mod['Event Technology'].replace(technology, 'Others')
                
                if len(xput_data_mod) != 0:
                    tech_subframe = xput_data_mod[['Event Technology', 'Timestamp',  'Smart Phone Smart Throughput Mobile Network UL Throughput [Mbps]']].dropna()
                    # Create a column to detect when the technology changes
                    tech_subframe['tech_change'] = tech_subframe['Event Technology'].ne(tech_subframe['Event Technology'].shift()).cumsum()

                    # # Split the dataframe into subframes based on consecutive technology
                    subframe_list = [group for _, group in tech_subframe.groupby('tech_change')]

                    # # Remove the 'tech_change' column from each subframe
                    subframe_list = [subf.drop(columns=['tech_change']) for subf in subframe_list]

                    if len(subframe_list) > 0:
                        for xput_sub_frame in subframe_list:
                            if '5G' not in xput_sub_frame['Event Technology'].iloc[0]:
                                continue 
                            if len(xput_sub_frame['Smart Phone Smart Throughput Mobile Network UL Throughput [Mbps]'].dropna()) < 2:
                                continue
                            if 0:
                                if xput_sub_frame['Timestamp'].iloc[-1] - xput_sub_frame['Timestamp'].iloc[0] < 9.5:
                                    continue 
                                # Create chunks of 10-second intervals
                                start_time = xput_data_mod['Timestamp'].iloc[0]

                                for _, group in xput_data_mod.groupby((xput_data_mod['Timestamp'] - start_time) // 10):
                                    if group['Timestamp'].iloc[-1] - group['Timestamp'].iloc[0] < 9.5:
                                        continue
                                    main_tech_xput_ul_mean_dict[xput_sub_frame['Event Technology'].iloc[0]].append(np.mean([i for i in group['Smart Phone Smart Throughput Mobile Network UL Throughput [Mbps]'].dropna() if i != 'Skip']))
                                    main_tech_xput_ul_std_dict[xput_sub_frame['Event Technology'].iloc[0]].append(np.std([i for i in group['Smart Phone Smart Throughput Mobile Network UL Throughput [Mbps]'].dropna() if i != 'Skip']))
                            else:
                                xput_list = list(xput_sub_frame['Smart Phone Smart Throughput Mobile Network UL Throughput [Mbps]'].dropna())
                                xput_list_4, xput_list_8 = calculate_averages(xput_list)

                                try:
                                    differences = [xput_list[i+1] - xput_list[i] for i in range(len(xput_list) - 1)]
                                    main_tech_xput_ul_diff_dict[xput_sub_frame['Event Technology'].iloc[0]][0.5].extend(differences)

                                    differences_4 = [xput_list_4[i+1] - xput_list_4[i] for i in range(len(xput_list_4) - 1)]
                                    main_tech_xput_ul_diff_dict[xput_sub_frame['Event Technology'].iloc[0]][2].extend(differences_4)

                                    differences_8 = [xput_list_8[i+1] - xput_list_8[i] for i in range(len(xput_list_8) - 1)]
                                    main_tech_xput_ul_diff_dict[xput_sub_frame['Event Technology'].iloc[0]][4].extend(differences_8)
                                except:
                                    a = 1



            original_list = [xput_data['Timestamp'], xput_data['Lat'], xput_data['Lon'], xput_data['Event Technology'], xput_data['5G KPI Total Info Layer2 MAC UL Throughput [Mbps]'], xput_data['5G KPI Total Info UL CA Type'], xput_data['5G KPI PCell RF Serving SS-RSRP [dBm]'], xput_data['5G KPI PCell RF PUSCH Power [dBm]'], xput_data['5G KPI PCell RF PUCCH Power [dBm]'], xput_data['5G KPI PCell RF Pathloss [dB]'], xput_data['city_info']]
            mac_cell_xput_cols = [i for i in xput_data.columns if 'MAC UL Throughput' in i and '5G' in i and 'Total' not in i]
            cell_list = []
            for col in mac_cell_xput_cols:
                cell_list.append(col.split()[2])
            
            for i in range(3):
                if i == 0:
                    # xput 
                    for cell in cell_list:
                        original_list.append(xput_data['5G KPI %s Layer2 MAC UL Throughput [Mbps]' %cell])
                elif i == 1:
                    # band
                    for cell in cell_list:
                        original_list.append(xput_data['5G KPI %s RF Band' %cell])
                elif i == 2:
                    # bandwidth
                    for cell in cell_list:
                        original_list.append(xput_data['5G KPI %s RF BandWidth' %cell])
                        
            # for lat, lon, xput_tech, xput, ca, band_1, band_2, rx_power, tx_power, tx_power_control, bandwidth_1, bandwidth_2, pathloss, city_info in zip(xput_data['Lat'], xput_data['Lon'], xput_data['Event Technology'], xput_data['5G KPI Total Info Layer2 MAC UL Throughput [Mbps]'], xput_data['5G KPI Total Info UL CA Type'], xput_data['5G KPI PCell RF Band'], xput_data['5G KPI SCell[1] RF Band'], xput_data['5G KPI PCell RF Serving SS-RSRP [dBm]'], xput_data['5G KPI PCell RF PUSCH Power [dBm]'], xput_data['5G KPI PCell RF PUCCH Power [dBm]'], xput_data['5G KPI PCell RF BandWidth'], xput_data['5G KPI SCell[1] RF BandWidth'], xput_data['5G KPI PCell RF Pathloss [dB]'], xput_data['city_info']):
            # xput_data = sub_df[['Event Technology', '5G KPI PCell Layer2 MAC UL Throughput [Mbps]']].dropna()
            # for xput_tech, xput in zip(xput_data['Event Technology'], xput_data['5G KPI PCell Layer2 MAC UL Throughput [Mbps]']):
            prev_ts = None
            prev_ca = None 
            prev_band_combo = None
            prev_sum_bandwidth = None
            elem_count = -1
            zipped_list = list(zip(*original_list))
            for elem in zip(*original_list):
                elem_count+=1
                original_data = list(elem)[:11]
                added_data = list(elem)[11:]
                ts, lat, lon, xput_tech, xput, ca, rx_power, tx_power, tx_power_control, pathloss, city_info = original_data

                if xput == 'Skip':
                    fiveg_xput_tech_ul_dict['Skip']+=1
                else:
                    if 'NSA' in xput_tech:
                        xput_tech = 'NSA'
                    elif '_SA' in xput_tech:
                        xput_tech = 'SA'
                    elif 'LTE' in xput_tech:
                        continue
                    if xput_tech not in fiveg_xput_tech_ul_dict.keys():
                        fiveg_xput_tech_ul_dict[xput_tech] = []
                    fiveg_xput_tech_ul_dict[xput_tech].append(xput)

                    dividend = len(added_data) // 3
                    xput_cell_list = []
                    band_cell_list = []
                    bandwidth_cell_list = []
                    for i in range(dividend):
                        xput_cell_list.append(added_data[i])
                        band_cell_list.append(added_data[i + dividend])
                        bandwidth_cell_list.append(added_data[i + 2 * dividend])

                    # override the CA 
                    ca = 0
                    band_combo = ''
                    sum_bandwidth = 0
                    for x, ba, bw in zip(xput_cell_list, band_cell_list, bandwidth_cell_list):
                        if pd.isnull(x):
                            continue
                        if x != 'Skip' and x > 0:
                            ca+=1
                            band_combo += str(ba) + ":"
                            sum_bandwidth += bw
                    # check if previous element is not null
                    if 'nan' in band_combo and prev_ts != None and (ts - prev_ts) <= 0.3 and prev_ca == ca:
                        band_combo = prev_band_combo
                        sum_bandwidth = prev_sum_bandwidth
                    elif 'nan' in band_combo and elem_count + 1 < len(zipped_list) and zipped_list[elem_count + 1][0] != None and (zipped_list[elem_count + 1][0] - ts) <= 0.3:
                        next_added_data = zipped_list[elem_count + 1][11:]
                        next_dividend = len(next_added_data) // 3
                        next_xput_cell_list = []
                        next_band_cell_list = []
                        next_bandwidth_cell_list = []
                        for i in range(next_dividend):
                            next_xput_cell_list.append(next_added_data[i])
                            next_band_cell_list.append(next_added_data[i + next_dividend])
                            next_bandwidth_cell_list.append(next_added_data[i + 2 * next_dividend])

                        # override the CA 
                        next_ca = 0
                        next_band_combo = ''
                        next_sum_bandwidth = 0
                        for x, ba, bw in zip(next_xput_cell_list, next_band_cell_list, next_bandwidth_cell_list):
                            if pd.isnull(x):
                                continue
                            if x != 'Skip' and x > 0:
                                next_ca+=1
                                next_band_combo += str(ba) + ":"
                                next_sum_bandwidth += bw

                        if 'nan' not in next_band_combo:
                            band_combo = next_band_combo
                            sum_bandwidth = next_sum_bandwidth
                    elif 'nan' in band_combo:
                        a = 1

                    prev_ca = ca 
                    prev_band_combo = band_combo
                    prev_sum_bandwidth = sum_bandwidth
                    prev_ts = ts

                    if not pd.isnull(ca):
                        
                        if ca not in main_sa_nsa_lte_tech_ca_ul_band_xput_dict[xput_tech].keys():
                            main_sa_nsa_lte_tech_ca_ul_band_xput_dict[xput_tech][ca] = {}

                        if band_combo not in main_sa_nsa_lte_tech_ca_ul_band_xput_dict[xput_tech][ca].keys():
                            main_sa_nsa_lte_tech_ca_ul_band_xput_dict[xput_tech][ca][band_combo] = []
                        main_sa_nsa_lte_tech_ca_ul_band_xput_dict[xput_tech][ca][band_combo].append(xput)

                        if sum_bandwidth not in main_tech_xput_ul_bandwidth_dict[xput_tech].keys():
                            main_tech_xput_ul_bandwidth_dict[xput_tech][sum_bandwidth] = []
                        main_tech_xput_ul_bandwidth_dict[xput_tech][sum_bandwidth].append(xput)

                        if (ca, band_combo, sum_bandwidth) not in main_tech_xput_ul_ca_bandwidth_dict[xput_tech].keys():
                            main_tech_xput_ul_ca_bandwidth_dict[xput_tech][(ca, band_combo, sum_bandwidth)] = []
                        main_tech_xput_ul_ca_bandwidth_dict[xput_tech][(ca, band_combo, sum_bandwidth)].append(xput)

                    # get city info     
                    # filtered_df = city_tech_df[city_tech_df['Lat-Lon'].isin({(lat, lon)})]
                    # if len(filtered_df) == 0:
                    #     city_data = extrapolate_road_data_new((lat, lon))
                    # else:
                    #     city_data = filtered_df.iloc[0]['city_info']

                    # if city_data not in main_tech_xput_city_info_dict[xput_tech].keys():
                    #     main_tech_xput_city_info_dict[xput_tech][city_data] = []

                    # main_tech_xput_city_info_dict[xput_tech][city_data].append(xput)
                    if 1:
                        main_tech_xput_tx_power_dict[xput_tech].append(tx_power)
                        main_tech_xput_tx_power_control_dict[xput_tech].append(tx_power_control)
                        main_tech_xput_ul_tx_power_dict[xput_tech].append(tx_power)
                        main_tech_xput_ul_tx_power_control_dict[xput_tech].append(tx_power_control)
                        main_tech_xput_rx_power_dict[xput_tech].append(rx_power)
                        main_tech_xput_ul_rx_power_dict[xput_tech].append(rx_power)
                        main_tech_xput_pathloss_dict[xput_tech].append(pathloss)
                        main_tech_xput_ul_pathloss_dict[xput_tech].append(pathloss)
            # get LTE MAC xput
            try:
                xput_data = sub_df[['Event Technology', 'LTE KPI MAC UL Throughput [Mbps]']].dropna()
                for xput_tech, xput in zip(xput_data['Event Technology'], xput_data['LTE KPI MAC UL Throughput [Mbps]']):
                    if xput == 'Skip':
                        lte_xput_tech_ul_dict['Skip']+=1
                    else:
                        if 'NSA' in xput_tech:
                            xput_tech = 'NSA'
                        elif '_SA' in xput_tech:
                            xput_tech = 'SA'
                        elif 'LTE' in xput_tech:
                            xput_tech = 'LTE'
                        if xput_tech not in lte_xput_tech_ul_dict.keys():
                            lte_xput_tech_ul_dict[xput_tech] = []
                        lte_xput_tech_ul_dict[xput_tech].append(xput)
            except:
                if xput_tech == 'LTE' or 'NSA' in xput_tech:
                    a = 1
                print("LTE data not present!")

    if 1:
        extract_drive_trip_data_lax_bos()

        global fiveg_xput_tech_dl_dict, lte_xput_tech_dl_dict, fiveg_xput_tech_ul_dict, lte_xput_tech_ul_dict, main_sa_nsa_lte_tech_ca_band_xput_dict, main_sa_nsa_lte_tech_ca_ul_band_xput_dict

        fh = open("../pkls/driving_trip_lax_bos_2024/xput_break.pkl", "wb")
        pkl.dump([fiveg_xput_tech_dl_dict, lte_xput_tech_dl_dict, fiveg_xput_tech_ul_dict, lte_xput_tech_ul_dict, main_sa_nsa_lte_tech_ca_band_xput_dict, main_sa_nsa_lte_tech_ca_ul_band_xput_dict], fh)
        fh.close()

        global main_tech_xput_city_info_dict, main_tech_xput_tx_power_dict, main_tech_xput_tx_power_control_dict, main_tech_xput_dl_tx_power_dict, main_tech_xput_dl_tx_power_control_dict, main_tech_xput_ul_tx_power_dict, main_tech_xput_ul_tx_power_control_dict, main_tech_xput_rx_power_dict, main_tech_xput_dl_rx_power_dict, main_tech_xput_ul_rx_power_dict, main_tech_xput_pathloss_dict, main_tech_xput_dl_pathloss_dict, main_tech_xput_ul_pathloss_dict, main_tech_xput_dl_bandwidth_dict, main_tech_xput_ul_bandwidth_dict, main_tech_xput_dl_mean_dict, main_tech_xput_dl_std_dict, main_tech_xput_ul_mean_dict, main_tech_xput_ul_std_dict, main_tech_xput_dl_diff_dict, main_tech_xput_ul_diff_dict, main_tech_xput_dl_ca_bandwidth_dict, main_tech_xput_ul_ca_bandwidth_dict
        fh = open("../pkls/driving_trip_lax_bos_2024/xput_break_part_2.pkl", "wb")
        pkl.dump([main_tech_xput_city_info_dict, main_tech_xput_tx_power_dict, main_tech_xput_tx_power_control_dict, main_tech_xput_dl_tx_power_dict, main_tech_xput_dl_tx_power_control_dict, main_tech_xput_ul_tx_power_dict, main_tech_xput_ul_tx_power_control_dict, main_tech_xput_rx_power_dict, main_tech_xput_dl_rx_power_dict, main_tech_xput_ul_rx_power_dict, main_tech_xput_pathloss_dict, main_tech_xput_dl_pathloss_dict, main_tech_xput_ul_pathloss_dict, main_tech_xput_dl_bandwidth_dict, main_tech_xput_ul_bandwidth_dict, main_tech_xput_dl_mean_dict, main_tech_xput_dl_std_dict, main_tech_xput_ul_mean_dict, main_tech_xput_ul_std_dict, main_tech_xput_dl_diff_dict, main_tech_xput_ul_diff_dict, main_tech_xput_dl_ca_bandwidth_dict, main_tech_xput_ul_ca_bandwidth_dict], fh)
        fh.close()

def extract_overall_latency_dist():
    sub_cities_df = pd.read_csv('../raw_data/us_cities_of_interest.csv')
    def extrapolate_road_data_new(original_lat_lon):
        distance_from_city = []
        for lat_lng in list(sub_cities_df['lat-lon']):
            lat_lng = [float(i) for i in lat_lng.strip("()").split(",")]
            distance_from_city.append(geodesic(lat_lng, original_lat_lon).miles)
        if min(distance_from_city) <= 5:
            return 'big-city'
        elif min(distance_from_city) > 5 and min(distance_from_city) <= 10:
            return 'unclassified'
        else:
            return 'not-big-city'
        
    def return_city_info(lat_lon):
        global lat_lon_city_dict
        if lat_lon not in lat_lon_city_dict.keys():
            return None
        else:
            return lat_lon_city_dict[lat_lon]
        
    def extract_drive_trip_data_lax_bos():
        global main_tech_ping_band_dict
        global main_tech_ping_band_scs_dict
        global n25_ping_location
        global main_xput_tech_ping_dict
        global main_tech_ping_rsrp_dict
        global main_tech_ping_pathloss_dict
        global main_tech_ping_city_info_dict
        global main_ping_tx_power_dict
        global main_ping_tx_power_control_dict

        try:
            global city_tech_df
        except:
            city_tech_df = None 

        # get the tput  test times
        drive_trip_data_path = "../raw_data/data_2024/xcal_kpi_data"    


        for day in ['day_5', 'day_6', 'day_7', 'day_8']:
            tmobile_template_data = glob.glob(drive_trip_data_path + "/*tmobile*%s*xlsx" %day)
            for template_csv in tmobile_template_data:
                print("Parsing now: %s" %template_csv)
                df_template = pd.read_excel(template_csv)

                df_template.drop(df_template.tail(8).index,inplace=True)
                df_template['TIME_STAMP'] = df_template['TIME_STAMP'].apply(datetime_to_timestamp)
                df_template = df_template.rename(columns={'TIME_STAMP' : 'Timestamp'})
                df_template = df_template.sort_values("Timestamp").reset_index(drop=True)

                if day == 'day_5':
                    # Get the timestamp of the last row
                    last_timestamp = df_template['Timestamp'].iloc[-1]

                    # Create a subset of the dataframe where TIME_STAMP > last_timestamp - 3600
                    df_template = df_template[df_template['Timestamp'] > (last_timestamp - 3600)]
                    
                # get app times 
                app_base = "../raw_data/data_2024/app_data/2024110%s" %day.split("_")[-1]
                app_folders = sorted(glob.glob(app_base + "/*"))

                ping_start_end_times = []
                for app_folder in app_folders:
                    if '.log' in app_folder:
                        continue 
                    
                    ping_temp_times = []
                    app_data_logs = sorted(glob.glob(app_folder + "/*.out"))
                    for app_data in app_data_logs:
                        if 'icmp_ping' in app_data:
                            fh = open(app_data, "r")
                            data = fh.readlines()
                            if day in ['day_1', 'day_2']:
                                start = (int(data[0].split(":")[-1].strip()) - 3600000  ) / 1000      
                                end = start + (get_icmp_ping_data_length(data)) * 0.2  

                            else: 
                                start = int(data[0].split(":")[-1].strip()) / 1000     
                                end = start + (get_icmp_ping_data_length(data)) * 0.2    

                            fh.close()
                            ping_temp_times.extend([start, end])
                     
                    ping_temp_times = sorted(ping_temp_times)
                    try:
                        ping_start_end_times.append((ping_temp_times[0], ping_temp_times[1], data))
                    except:
                        pass 


                # work with ping first
                for start_end_time in ping_start_end_times:
                    start_time, end_time, data = start_end_time
                    sub_df = get_start_end_indices(df_template, start_time, end_time)
                    if len(sub_df) == 0 or len(sub_df['Event Technology'].dropna()) == 0:
                        continue
                    sub_df = sub_df.loc[sub_df['Event Technology'].dropna().index[0]:]
                    sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR', '5G-NR_SA')
                    sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR(2CA)', '5G-NR_SA')
                    sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR(3CA)', '5G-NR_SA')
                    sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR(4CA)', '5G-NR_SA')
                    
                    # get the ping data 
                    ping_data = []
                    ping_timestamp = []
                    ping_time = start_time - 0.2
                    for d in data:
                        if 'bytes' in d and 'icmp_seq' in d and 'ttl' in d and 'time' in d:
                            try:
                                d = float(d.strip().replace("\n", "").split()[-2].split("=")[-1])
                            except:
                                continue
                            ping_data.append(d)
                            ping_time+=0.2
                            ping_timestamp.append(ping_time)

                    ping_df = pd.DataFrame({'Timestamp': ping_timestamp, 'ping_data': ping_data})
                    sub_df['Timestamp'] = sub_df['Timestamp'].fillna(method='ffill')

                    sub_df = pd.concat([sub_df, ping_df])
                    sub_df = sub_df.sort_values(by=["Timestamp"])
                    sub_df.reset_index(inplace=True)

                    sub_df['Event Technology'] = sub_df['Event Technology'].fillna(method='bfill')
                    sub_df['5G KPI PCell RF Band'] = sub_df['5G KPI PCell RF Band'].fillna(method='bfill')
                    sub_df['Lat'] = sub_df['Lat'].fillna(method='bfill')
                    sub_df['Lon'] = sub_df['Lon'].fillna(method='bfill')
                    sub_df['5G KPI PCell RF Serving SS-RSRP [dBm]'] = sub_df['5G KPI PCell RF Serving SS-RSRP [dBm]'].fillna(method='bfill')
                    sub_df['5G KPI PCell RF PUSCH Power [dBm]'] = sub_df['5G KPI PCell RF PUSCH Power [dBm]'].fillna(method='bfill')
                    sub_df['5G KPI PCell RF PUCCH Power [dBm]'] = sub_df['5G KPI PCell RF PUCCH Power [dBm]'].fillna(method='bfill')
                    sub_df['5G KPI PCell RF Pathloss [dB]'] = sub_df['5G KPI PCell RF Pathloss [dB]'].fillna(method='bfill')
                    sub_df['5G KPI PCell RF Subcarrier Spacing'] = sub_df['5G KPI PCell RF Subcarrier Spacing'].fillna(method='bfill')

                    sub_df['Lat-Lon'] = sub_df.apply(lambda row: (row['Lat'], row['Lon']), axis=1)
                    sub_df['city_info'] = sub_df['Lat-Lon'].apply(return_city_info)
                    sub_df['city_info'] = sub_df['city_info'].fillna(method='ffill').fillna(method='bfill')

                    ping_df_data = sub_df[['Lat', 'Lon', 'city_info', 'Event Technology', 'ping_data', '5G KPI PCell RF Band', '5G KPI PCell RF Serving SS-RSRP [dBm]', '5G KPI PCell RF PUSCH Power [dBm]', '5G KPI PCell RF PUCCH Power [dBm]', '5G KPI PCell RF Pathloss [dB]', '5G KPI PCell RF Subcarrier Spacing']].dropna()
                    for ping_tech, ping, band, lat, lon, rsrp, tx_power, tx_power_control, pathloss, city_info, scs in zip(ping_df_data['Event Technology'], ping_df_data['ping_data'], ping_df_data['5G KPI PCell RF Band'], ping_df_data['Lat'], ping_df_data['Lon'], ping_df_data['5G KPI PCell RF Serving SS-RSRP [dBm]'], ping_df_data['5G KPI PCell RF PUSCH Power [dBm]'], ping_df_data['5G KPI PCell RF PUCCH Power [dBm]'], ping_df_data['5G KPI PCell RF Pathloss [dB]'], ping_df_data['city_info'], ping_df_data['5G KPI PCell RF Subcarrier Spacing']):
                        if ping == 'Skip':
                            main_xput_tech_ping_dict['Skip']+=1
                            main_xput_tech_ping_dict_day_wise['Skip']+=1
                        else:
                            if 'NSA' in ping_tech:
                                ping_tech = 'NSA'

                            elif '_SA' in ping_tech:
                                ping_tech = 'SA'

                            elif 'LTE' in ping_tech:
                                ping_tech = 'LTE'

                            # tech ping data 
                            if ping_tech not in main_xput_tech_ping_dict.keys():
                                main_xput_tech_ping_dict[ping_tech] = []
                                main_xput_tech_ping_dict_day_wise[ping_tech] = {}
                            
                            if day not in main_xput_tech_ping_dict_day_wise[ping_tech].keys():
                                main_xput_tech_ping_dict_day_wise[ping_tech][day] = []
                            main_xput_tech_ping_dict[ping_tech].append(ping)
                            main_xput_tech_ping_dict_day_wise[ping_tech][day].append(ping)
                            if ping_tech != 'LTE':
                                # get band info
                                if band not in main_tech_ping_band_dict[ping_tech].keys():
                                    main_tech_ping_band_dict[ping_tech][band] = []
                                    main_tech_ping_band_scs_dict[ping_tech][band] = []
                                if 'n25' in band:
                                    n25_ping_location.append((lat, lon))
                                
                                main_tech_ping_band_dict[ping_tech][band].append(ping)
                                main_tech_ping_band_scs_dict[ping_tech][band].append(scs)

                                # get rsrp data 
                                if 1:
                                    main_tech_ping_rsrp_dict[ping_tech].append(rsrp)
                                    main_tech_ping_pathloss_dict[ping_tech].append(pathloss)
                                    main_ping_tx_power_dict[ping_tech].append(tx_power)
                                    main_ping_tx_power_control_dict[ping_tech].append(tx_power_control)



    if 1:
        extract_drive_trip_data_lax_bos()

        fh = open("../pkls/driving_trip_lax_bos_2024/rtt_break.pkl", "wb")
        pkl.dump([main_xput_tech_ping_dict, main_xput_tech_ping_dict_day_wise, main_tech_ping_band_dict, main_tech_ping_city_info_dict, main_tech_ping_rsrp_dict, main_tech_ping_pathloss_dict, main_ping_tx_power_dict, main_ping_tx_power_control_dict, n25_ping_location, main_tech_ping_band_scs_dict], fh)
        fh.close()
   
def get_ho_duration():
    if 0:
        drive_trip_base = "../raw_data/data_2024/ho_duration_data/"

        ho_duration_files = glob.glob(drive_trip_base + "*.xlsx")

        sa_duration_list = []
        sa_duration_list_intra_freq = []
        sa_duration_list_inter_freq = []
        sa_duration_list_intra_gnb = []
        sa_duration_list_inter_gnb = []
        nsa_nr_nr_duration_list = []
        nsa_nr_nr_duration_intra_list = []
        nsa_nr_nr_duration_inter_list = []
        nsa_lte_nr_duration_list = []
        lte_lte_duration_list = []
        nsa_nr_nr_time_diff_after_lte_ho = []
        nsa_lte_nr_time_diff_after_lte_ho = []
        sa_inter_intra_dict = {}
        nsa_inter_intra_dict = {'intra_gnb:intra_freq' : [], 'intra_gnb:inter_freq' : [], 'inter_gnb:intra_freq' : [], 'inter_gnb:inter_freq' : []}

        nsa_inter_intra_ts = {}
        for ho_duration_file in ho_duration_files:
            if 'day_1' in ho_duration_file or 'day_2' in ho_duration_file or 'day_3' in ho_duration_file or 'day_4' in ho_duration_file:
                continue
            df_template = pd.read_excel(ho_duration_file)

            df_template.drop(df_template.tail(8).index,inplace=True)
            df_template['TIME_STAMP'] = df_template['TIME_STAMP'].apply(datetime_to_timestamp)
            
            df_template = df_template.rename(columns={'TIME_STAMP' : 'Timestamp'})
            df_template = df_template.sort_values("Timestamp").reset_index(drop=True)

            if 'day_5' in ho_duration_file:
                # Get the timestamp of the last row
                last_timestamp = df_template['Timestamp'].iloc[-1]

                # Create a subset of the dataframe where TIME_STAMP > last_timestamp - 3600
                df_template = df_template[df_template['Timestamp'] > (last_timestamp - 3600)]

            try:
                sa_duration_list.extend(list(df_template['5G-NR RRC NR MCG Mobility Statistics Intra-NR HandoverDuration [sec]'].dropna()))
            except:
                pass

            try:
                sa_duration_list_inter_freq.extend(list(df_template['5G-NR RRC NR MCG Mobility Statistics Interfreq Handover Duration [sec]'].dropna()))
            except:
                pass
            
            try:
                sa_duration_list_intra_freq.extend(list(df_template['5G-NR RRC NR MCG Mobility Statistics Intrafreq Handover Duration [sec]'].dropna()))
            except:
                pass 

            try:
                sa_duration_list_inter_gnb.extend(list(df_template['5G-NR RRC NR MCG Mobility Statistics Inter-gNB Handover Duration [sec]'].dropna()))
            except:
                pass
            
            try:
                sa_duration_list_intra_gnb.extend(list(df_template['5G-NR RRC NR MCG Mobility Statistics Intra-gNB Handover Duration [sec]'].dropna()))
            except:
                pass

            try:
                nsa_nr_nr_duration_list.extend(list(df_template['5G-NR RRC NR SCG Mobility Statistics NR SCG Modification[NR to NR] Duration [sec]'].dropna()))
            except:
                pass

            try:
                nsa_lte_nr_duration_list.extend(list(df_template['5G-NR RRC NR SCG Mobility Statistics NR SCG Addition[LTE to NR] Duration [sec]'].dropna()))
            except:
                pass

            try:
                lte_lte_duration_list.extend(list(df_template['HO Statistics Intra-LTE HO Duration [sec]'].dropna()))
            except:
                pass

            try:
                temp = df_template[['Timestamp', '5G KPI PCell RF Serving PCI', '5G KPI PCell RF NR-ARFCN', '5G-NR RRC NR SCG Mobility Statistics NR SCG Modification[NR to NR] Duration [sec]']]
                # Step 1: Filter rows where the specified column is not empty
                non_empty_indices = temp[temp['5G-NR RRC NR SCG Mobility Statistics NR SCG Modification[NR to NR] Duration [sec]'].notna()].index

                # Step 2 and 3: Get rows within 500 ms before and after the non-empty indices
                result_indices = []

                for idx in non_empty_indices:
                    # Get the current timestamp
                    current_timestamp = temp.loc[idx, 'Timestamp']
                    
                    # Get rows within ±500 ms of the current timestamp
                    before = temp[(temp['Timestamp'] >= current_timestamp - 0.600) &
                                        (temp['Timestamp'] <= current_timestamp)]
                    after = temp[(temp['Timestamp'] > current_timestamp) &
                                        (temp['Timestamp'] <= current_timestamp + 0.600)]
                    
                    # Add the indices to the result list
                    result_indices.append((before, after))

                for tple in result_indices:
                    try:
                        before = tple[0][['5G KPI PCell RF Serving PCI', '5G KPI PCell RF NR-ARFCN']].dropna()
                        after = tple[1][['5G KPI PCell RF Serving PCI', '5G KPI PCell RF NR-ARFCN']].dropna()
                        before_pci_count_max_occurrence, before_pci_count_max_frequency = Counter(before['5G KPI PCell RF Serving PCI'].dropna()).most_common(1)[0]
                        after_pci_count_max_occurrence, after_pci_count_max_frequency = Counter(after['5G KPI PCell RF Serving PCI'].dropna()).most_common(1)[0]

                        before_arfcn_count_max_occurrence, before_arfcn_count_max_frequency = Counter(before['5G KPI PCell RF NR-ARFCN'].dropna()).most_common(1)[0]
                        after_arfcn_count_max_occurrence, after_arfcn_count_max_frequency = Counter(after['5G KPI PCell RF NR-ARFCN'].dropna()).most_common(1)[0]

                        if before_arfcn_count_max_occurrence == after_arfcn_count_max_occurrence:
                            # no change freq
                            nsa_nr_nr_duration_intra_list.append(list(tple[0]['5G-NR RRC NR SCG Mobility Statistics NR SCG Modification[NR to NR] Duration [sec]'].dropna())[0])
                            intra_inter_ts = tple[0]['5G-NR RRC NR SCG Mobility Statistics NR SCG Modification[NR to NR] Duration [sec]'].dropna().index[-1]
                            intra_inter_ts = tple[0]['Timestamp'].loc[intra_inter_ts]
                            nsa_inter_intra_ts[intra_inter_ts] = 'intra'
                        else:
                            nsa_nr_nr_duration_inter_list.append(list(tple[0]['5G-NR RRC NR SCG Mobility Statistics NR SCG Modification[NR to NR] Duration [sec]'].dropna())[0])
                            intra_inter_ts = tple[0]['5G-NR RRC NR SCG Mobility Statistics NR SCG Modification[NR to NR] Duration [sec]'].dropna().index[-1]
                            intra_inter_ts = tple[0]['Timestamp'].loc[intra_inter_ts]
                            nsa_inter_intra_ts[intra_inter_ts] = 'inter'

                        try:
                            temp_inter_intra = return_nsa_ho_type(before_pci_count_max_occurrence, after_pci_count_max_occurrence, before_arfcn_count_max_occurrence, after_arfcn_count_max_occurrence)
                            if temp_inter_intra != "":
                                nsa_inter_intra_dict[temp_inter_intra].append(list(tple[0]['5G-NR RRC NR SCG Mobility Statistics NR SCG Modification[NR to NR] Duration [sec]'].dropna())[0])
                        except:
                            a = 1
                    except:
                        continue 

            except:
                pass

            # 
            for col in ['5G-NR RRC NR MCG Mobility Statistics Intra-gNB Handover Duration [sec]', '5G-NR RRC NR MCG Mobility Statistics Inter-gNB Handover Duration [sec]', '5G-NR RRC NR MCG Mobility Statistics Intrafreq Handover Duration [sec]', '5G-NR RRC NR MCG Mobility Statistics Interfreq Handover Duration [sec]']:
                if col not in df_template.columns:
                    df_template[col] = pd.NA
            
            sub_df = df_template[['5G-NR RRC NR MCG Mobility Statistics Intra-gNB Handover Duration [sec]', '5G-NR RRC NR MCG Mobility Statistics Inter-gNB Handover Duration [sec]', '5G-NR RRC NR MCG Mobility Statistics Intrafreq Handover Duration [sec]', '5G-NR RRC NR MCG Mobility Statistics Interfreq Handover Duration [sec]']]
            sub_df.rename(columns={'5G-NR RRC NR MCG Mobility Statistics Intra-gNB Handover Duration [sec]': 'intra_gnb'}, inplace=True)
            sub_df.rename(columns={'5G-NR RRC NR MCG Mobility Statistics Inter-gNB Handover Duration [sec]': 'inter_gnb'}, inplace=True)
            sub_df.rename(columns={'5G-NR RRC NR MCG Mobility Statistics Intrafreq Handover Duration [sec]': 'intra_freq'}, inplace=True)
            sub_df.rename(columns={'5G-NR RRC NR MCG Mobility Statistics Interfreq Handover Duration [sec]': 'inter_freq'}, inplace=True)
            sub_df = sub_df.dropna(how='all')
            
            for combo in [['intra_gnb', 'intra_freq'], ['intra_gnb', 'inter_freq'], ['inter_gnb', 'intra_freq'], ['inter_gnb', 'inter_freq']]:
                temp_df = sub_df[combo]
                temp_df = temp_df.dropna()
                dict_key = combo[0] + ":" + combo[1]
                if dict_key not in sa_inter_intra_dict.keys():
                    sa_inter_intra_dict[dict_key] = []

                sa_inter_intra_dict[dict_key].extend(list(temp_df[combo[0]]))

            # try to get the diff between a nr-nr/lte-nr and lte-lte ho
            if 'HO Statistics Intra-LTE HO Duration [sec]' not in df_template.columns:
                continue
            nsa_durations_df = df_template[['Timestamp', 'HO Statistics Intra-LTE HO Duration [sec]', '5G-NR RRC NR SCG Mobility Statistics NR SCG Modification[NR to NR] Duration [sec]', '5G-NR RRC NR SCG Mobility Statistics NR SCG Addition[LTE to NR] Duration [sec]']]
            non_empty_indices = nsa_durations_df[nsa_durations_df['HO Statistics Intra-LTE HO Duration [sec]'].notna()].index
            start_idx = list(non_empty_indices)[0]
            start_end_idx_tuple = []
            for idx in list(non_empty_indices)[1:]:
                start_end_idx_tuple.append((start_idx, idx))
                start_idx = idx 
            start_end_idx_tuple.append((list(non_empty_indices)[-1], nsa_durations_df.index[-1]))
            for start_end_idx in start_end_idx_tuple:
                current_timestamp = nsa_durations_df.loc[start_end_idx[0], 'Timestamp']
                after = nsa_durations_df[(nsa_durations_df['Timestamp'] > current_timestamp)]
                after = after[:start_end_idx[1]]
                try:
                    nr_nr_ts = after['Timestamp'][after['5G-NR RRC NR SCG Mobility Statistics NR SCG Modification[NR to NR] Duration [sec]'].dropna().index[0]]
                    nsa_nr_nr_time_diff_after_lte_ho.append(nr_nr_ts - current_timestamp)
                except:
                    pass 
                
                try:
                    lte_nr_ts = after['Timestamp'][after['5G-NR RRC NR SCG Mobility Statistics NR SCG Addition[LTE to NR] Duration [sec]'].dropna().index[0]]
                    nsa_lte_nr_time_diff_after_lte_ho.append(lte_nr_ts - current_timestamp)
                except:
                    pass 

        fh = open("../pkls/driving_trip_lax_bos_2024/ho_duration.pkl", "wb")
        pkl.dump([sa_duration_list,\
        sa_duration_list_intra_freq,\
        sa_duration_list_inter_freq,\
        sa_duration_list_intra_gnb,\
        sa_duration_list_inter_gnb,\
        nsa_nr_nr_duration_list,\
        nsa_lte_nr_duration_list,\
        lte_lte_duration_list,\
        nsa_nr_nr_time_diff_after_lte_ho,\
        nsa_lte_nr_time_diff_after_lte_ho,\
        sa_inter_intra_dict, \
        nsa_nr_nr_duration_intra_list,\
        nsa_nr_nr_duration_inter_list,\
        nsa_inter_intra_dict], fh)
        fh.close()
    
    if 1:
        drive_trip_base = "../raw_data/data_2024/security_ho_data/"
        ho_security_files = glob.glob(drive_trip_base + "*.xlsx")

        nsa_latency_sec_list = []
        sa_latency_sec_list = []
        sa_latency_sec_dict = {}
        for security_file in ho_security_files:
            if 'day_1' in security_file or 'day_2' in security_file or 'day_3' in security_file or 'day_4' in security_file:
                continue
            print("Currently parsing file: " + security_file)
            df_template = pd.read_excel(security_file)

            df_template.drop(df_template.tail(8).index,inplace=True)
            df_template['TIME_STAMP'] = df_template['TIME_STAMP'].apply(datetime_to_timestamp)
            
            df_template = df_template.rename(columns={'TIME_STAMP' : 'Timestamp'})
            df_template = df_template.sort_values("Timestamp").reset_index(drop=True)


            if 'day_5' in security_file:
                # Get the timestamp of the last row
                last_timestamp = df_template['Timestamp'].iloc[-1]

                # Create a subset of the dataframe where TIME_STAMP > last_timestamp - 3600
                df_template = df_template[df_template['Timestamp'] > (last_timestamp - 3600)]


            ho_duration_df = pd.read_excel(security_file.replace('security_ho', 'ho_duration'))
            ho_duration_df.drop(ho_duration_df.tail(8).index,inplace=True)
            ho_duration_df['TIME_STAMP'] = ho_duration_df['TIME_STAMP'].apply(datetime_to_timestamp)
            
            ho_duration_df = ho_duration_df.rename(columns={'TIME_STAMP' : 'Timestamp'})
            ho_duration_df = ho_duration_df.sort_values("Timestamp").reset_index(drop=True)

            ho_duration_df = ho_duration_df[['Timestamp', '5G-NR RRC NR MCG Mobility Statistics Intrafreq Handover Duration [sec]', '5G-NR RRC NR MCG Mobility Statistics Interfreq Handover Duration [sec]']]

            sa_latency_sec_dict[security_file] = []

            for column_name in ['Timestamp', '5G-NR RRC NR SCG Mobility Statistics NR SCG Modification[NR to NR] Duration [sec]', 'radioBearerConfig securityConfig securityAlgorithmConfig cipheringAlgorithm', 'radioBearerConfig securityConfig securityAlgorithmConfig integrityProtAlgorithm', 'LTE Signaling Message RRC Handover Security Config HO intraLTE cipheringAlgorithm', 'LTE Signaling Message RRC Handover Security Config HO intraLTE integrityProtAlgorithm']:
                if column_name not in df_template.columns:
                    df_template[column_name] = np.nan

            if 1:
                nsa_df = df_template[['Timestamp', '5G-NR RRC NR SCG Mobility Statistics NR SCG Modification[NR to NR] Duration [sec]', 'radioBearerConfig securityConfig securityAlgorithmConfig cipheringAlgorithm', 'radioBearerConfig securityConfig securityAlgorithmConfig integrityProtAlgorithm', 'LTE Signaling Message RRC Handover Security Config HO intraLTE cipheringAlgorithm', 'LTE Signaling Message RRC Handover Security Config HO intraLTE integrityProtAlgorithm']]
                
                # Filter rows where `x` is not null
                non_null_x = nsa_df[nsa_df['5G-NR RRC NR SCG Mobility Statistics NR SCG Modification[NR to NR] Duration [sec]'].notna()]

                # Create a dictionary to store sub-DataFrames
                sub_dfs = []

                # Iterate through non-null `x` rows
                for index, row in non_null_x.iterrows():
                    timestamp = row['Timestamp']
                    x_value = row['5G-NR RRC NR SCG Mobility Statistics NR SCG Modification[NR to NR] Duration [sec]']
                    
                    # Get the sub-DataFrame where Timestamp is in the range [Timestamp - x, Timestamp]
                    sub_df = nsa_df[(nsa_df['Timestamp'] >= timestamp - (x_value + 0.1)) & (nsa_df['Timestamp'] <= timestamp)]
                    
                    # Store the sub-DataFrame using the current `x` value or index as the key
                    sub_dfs.append([index, sub_df])
                
                for sub_df_combo in sub_dfs:
                    index, sub_df = sub_df_combo
                    ho_latency = sub_df.loc[index]['5G-NR RRC NR SCG Mobility Statistics NR SCG Modification[NR to NR] Duration [sec]']
                    # nsa_inter_intra_ts
                    try:
                        intra_inter_str = [nsa_inter_intra_ts[i] for i in nsa_inter_intra_ts.keys() if i == sub_df.loc[index]['Timestamp']][0]
                    except:
                        intra_inter_str = None 

                    fiveg_cipher, fiveg_integrity, lte_cipher, lte_integrity = [np.nan] * 4
                    if len(sub_df.loc[:index - 1]) > 0:
                        temp = sub_df.loc[:index - 1]
                        columns_to_check = ['radioBearerConfig securityConfig securityAlgorithmConfig cipheringAlgorithm',
                            'radioBearerConfig securityConfig securityAlgorithmConfig integrityProtAlgorithm',
                            'LTE Signaling Message RRC Handover Security Config HO intraLTE cipheringAlgorithm',
                            'LTE Signaling Message RRC Handover Security Config HO intraLTE integrityProtAlgorithm'
                        ]

                        # Drop rows where all specified columns are null
                        temp = temp.dropna(subset=columns_to_check, how='all')
                        if len(temp) > 0:
                            for index, row in temp.iterrows():
                                fiveg_cipher = row['radioBearerConfig securityConfig securityAlgorithmConfig cipheringAlgorithm']
                                fiveg_integrity = row['radioBearerConfig securityConfig securityAlgorithmConfig integrityProtAlgorithm']
                                lte_cipher = row['LTE Signaling Message RRC Handover Security Config HO intraLTE cipheringAlgorithm']
                                lte_integrity = row['LTE Signaling Message RRC Handover Security Config HO intraLTE integrityProtAlgorithm']
                                break
                    nsa_latency_sec_list.append([ho_latency, fiveg_cipher, fiveg_integrity, lte_cipher, lte_integrity, intra_inter_str])

            if 1:
                sa_df = df_template[['Timestamp', '5G-NR RRC NR MCG Mobility Statistics Intra-NR HandoverDuration [sec]', 'radioBearerConfig securityConfig securityAlgorithmConfig cipheringAlgorithm', 'radioBearerConfig securityConfig securityAlgorithmConfig integrityProtAlgorithm']]

                # Filter rows where `x` is not null
                non_null_x = sa_df[sa_df['5G-NR RRC NR MCG Mobility Statistics Intra-NR HandoverDuration [sec]'].notna()]

                # Create a dictionary to store sub-DataFrames
                sub_dfs = []

                # Iterate through non-null `x` rows
                for index, row in non_null_x.iterrows():
                    timestamp = row['Timestamp']
                    x_value = row['5G-NR RRC NR MCG Mobility Statistics Intra-NR HandoverDuration [sec]']
                    
                    # Get the sub-DataFrame where Timestamp is in the range [Timestamp - x, Timestamp]
                    sub_df = sa_df[(sa_df['Timestamp'] >= timestamp - (x_value + 0.1)) & (sa_df['Timestamp'] <= timestamp)]
                    
                    # Store the sub-DataFrame using the current `x` value or index as the key
                    sub_dfs.append([index, sub_df])
                
                for sub_df_combo in sub_dfs:
                    index, sub_df = sub_df_combo
                    ho_latency = sub_df.loc[index]['5G-NR RRC NR MCG Mobility Statistics Intra-NR HandoverDuration [sec]']
                    intra_inter_str = None
                    intra_freq_duration = inter_freq_duration = None
                    try:
                        intra_freq_duration = ho_duration_df.loc[ho_duration_df['Timestamp'] == sub_df.loc[index]['Timestamp'], '5G-NR RRC NR MCG Mobility Statistics Intrafreq Handover Duration [sec]']
                        intra_freq_duration =  intra_freq_duration.iloc[0] if not intra_freq_duration.empty and not pd.isna(intra_freq_duration.iloc[0]) else None
                    except:
                        a = 1
                    try:
                        inter_freq_duration = ho_duration_df.loc[ho_duration_df['Timestamp'] == sub_df.loc[index]['Timestamp'], '5G-NR RRC NR MCG Mobility Statistics Interfreq Handover Duration [sec]']
                        inter_freq_duration =  inter_freq_duration.iloc[0] if not inter_freq_duration.empty and not pd.isna(inter_freq_duration.iloc[0]) else None
                    except:
                        a = 1

                    if inter_freq_duration == None and intra_freq_duration != None:
                        intra_inter_str = 'intra'
                    elif inter_freq_duration != None and intra_freq_duration == None:
                        intra_inter_str = 'inter'
                    elif inter_freq_duration != None and intra_freq_duration != None:
                        a = 1
                    elif inter_freq_duration == None and intra_freq_duration == None:
                        a = 1
                    fiveg_cipher, fiveg_integrity = [np.nan] * 2
                    if len(sub_df.loc[:index - 1]) > 0:
                        temp = sub_df.loc[:index - 1]
                        columns_to_check = ['radioBearerConfig securityConfig securityAlgorithmConfig cipheringAlgorithm',
                            'radioBearerConfig securityConfig securityAlgorithmConfig integrityProtAlgorithm',
                        ]

                        # Drop rows where all specified columns are null
                        temp = temp.dropna(subset=columns_to_check, how='all')
                        if len(temp) > 0:
                            for index, row in temp.iterrows():
                                fiveg_cipher = row['radioBearerConfig securityConfig securityAlgorithmConfig cipheringAlgorithm']
                                fiveg_integrity = row['radioBearerConfig securityConfig securityAlgorithmConfig integrityProtAlgorithm']
                                break
                    sa_latency_sec_list.append([ho_latency, fiveg_cipher, fiveg_integrity, intra_inter_str])
                    sa_latency_sec_dict[security_file].append([ho_latency, fiveg_cipher, fiveg_integrity, intra_inter_str])

       
        color_dict = {'5G-cipher-only' : 'salmon', '5G-cipher+lte-(cipher+integrity)' : 'black', 'no-5G-cipher-integrity' : 'slategrey', 'no-5G-or-lte-cipher-integrity' : 'slategrey', '5G-cipher+5G-integrity' : 'black'}
        label_dict = {'5G-cipher-only' : '5G cipher only', '5G-cipher+lte-(cipher+integrity)' : '5G cipher & LTE\n(cipher + integrity)', 'no-5G-cipher-integrity' : 'No cipher and integrity', 'no-5G-or-lte-cipher-integrity' : 'No cipher and integrity', '5G-cipher+5G-integrity' : '5G cipher & 5G integrity'}
        sa_sec_dict = {'5G-cipher-only' : [], '5G-integrity-only' : [], '5G-cipher+5G-integrity' : [], 'no-5G-cipher-integrity' : [], 'unknown' : []}
        for sa_latency_sec_combo in sa_latency_sec_list:
            ho_latency, cipher, integrity, inter_intra_str = sa_latency_sec_combo
            if not pd.isnull(cipher) and not pd.isnull(integrity):
                sa_sec_dict['5G-cipher+5G-integrity'].append(ho_latency)
            elif not pd.isnull(cipher) and pd.isnull(integrity):
                sa_sec_dict['5G-cipher-only'].append(ho_latency)
            elif pd.isnull(cipher) and not pd.isnull(integrity):
                sa_sec_dict['5G-integrity-only'].append(ho_latency)
            elif pd.isnull(cipher) and pd.isnull(integrity):
                sa_sec_dict['no-5G-cipher-integrity'].append(ho_latency)
            else:
                sa_sec_dict['unknown'].append(sa_latency_sec_combo)


        nsa_sec_dict = {'5G-cipher-only' : [],  '5G-cipher+lte-(cipher+integrity)' : [], 'no-5G-or-lte-cipher-integrity' : [], 'unknown' : []}
        for nsa_latency_sec_combo in nsa_latency_sec_list:
            ho_latency, fiveg_cipher, fiveg_integrity, lte_cipher, lte_integrity, intra_inter_str = nsa_latency_sec_combo
            if not pd.isnull(fiveg_cipher) and pd.isnull(lte_cipher):
                nsa_sec_dict['5G-cipher-only'].append(ho_latency)
            elif not pd.isnull(fiveg_cipher) and not pd.isnull(lte_cipher):
                nsa_sec_dict['5G-cipher+lte-(cipher+integrity)'].append(ho_latency)
            elif pd.isnull(fiveg_cipher) and  pd.isnull(lte_cipher):
                nsa_sec_dict['no-5G-or-lte-cipher-integrity'].append(ho_latency)
            else:
                nsa_sec_dict['unknown'].append(ho_latency)

        fig, ax = plt.subplots(figsize=(8, 4))
        for sec_type in nsa_sec_dict.keys():
            if len(nsa_sec_dict[sec_type]) == 0:
                continue 
            sorted_data = np.sort(nsa_sec_dict[sec_type])
            ax.plot(sorted_data, np.linspace(0, 1, sorted_data.size), label="NSA (" + label_dict[sec_type] + ")", color=color_dict[sec_type])

        for sec_type in sa_sec_dict.keys():
            if len(sa_sec_dict[sec_type]) == 0:
                continue 
            sorted_data = np.sort(sa_sec_dict[sec_type])
            ax.plot(sorted_data, np.linspace(0, 1, sorted_data.size), label="SA (" + label_dict[sec_type]+ ")" , color=color_dict[sec_type], ls='--')

        ax.set_ylim(0, 1)
        ax.set_xlim(0, 0.12)
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)
        # ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=11, frameon=True)
        ax.legend(loc='lower right', fontsize=14)
        ax.grid(True)
        ax.set_ylabel("CDF")
        ax.set_xlabel("Handover duration (s)")
        plt.tight_layout()
        plt.savefig('../plots/yearwise/ho_duration_security_break.pdf')
        plt.close()

# function calls 
if 1:
    get_ho_duration()

else:
    extract_sa_nsa_coverage()
    extract_overall_xput_dist()
    extract_5G_lte_break_xput_dist_conext()
    extract_overall_latency_dist()
