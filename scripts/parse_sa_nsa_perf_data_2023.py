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
import xyzservices as xyz
import geopy.distance
from pprint import pprint
from geopy.distance import distance
from geopy.distance import geodesic
from shapely.geometry import Point
from collections import Counter
from collections import defaultdict
from earfcn.convert import earfcn2freq, earfcn2band, freq2earfcn


new_york_city_lat_lon = [(40.757640, -73.985470), (40.696207, -73.984628), (40.721333, -73.998177), (40.723829, -73.996294), (40.858873, -73.872615), (40.570446, -74.109868)]
boston_city_lat_lon = [(42.356747, -71.057755)]

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

def datetime_to_timestamp_new(datetime_str):
    import datetime
    int(datetime_str.astimezone(datetime.timezone.utc).timestamp())
    return datetime_str.astimezone(datetime.timezone.utc).timestamp()

def downtown_measurements_mod(start_tuple, end_tuple):
    lat_lon_dt_dict = {'LA' : (34.05872013582416, -118.23766913901929), 'LV' : (36.11290509947277, -115.1731529445295), 'SLC' : (40.725262, -111.854019), 'DE' : (39.744331, -105.009438), 'CHIC' : (41.89307, -87.623787), 'INDY' : (39.768028, -86.15094), 'CLEV' : (41.5005, -81.674026) }
    for key in lat_lon_dt_dict:
        distance_from_start = geopy.distance.geodesic(lat_lon_dt_dict[key], start_tuple).miles
        distance_from_end = geopy.distance.geodesic(lat_lon_dt_dict[key], end_tuple).miles
        
        if distance_from_start < 2 or distance_from_end < 2:
            #downtown measurement
            return True
    return False

def calculate_percentage_of_occurrence(lst):
    total_elements = len(lst)
    
    # Use Counter to count the occurrence of each element
    element_counts = Counter(lst)

    # Calculate the percentage for each unique element
    percentage_dict = {element: round((count / total_elements) * 100) for element, count in element_counts.items()}

    return percentage_dict

def datetime_to_timestamp(datetime_str):
    from datetime import datetime
    if pd.isnull(datetime_str):
        return datetime_str
    date, time_all = datetime_str.split()
    temp_year = date.split("-")[0]
    temp_month = date.split("-")[1]
    temp_date = date.split("-")[2]
    datetime_string = temp_date + "." + temp_month + "." + temp_year + " " + time_all
    dt_obj = datetime.strptime(datetime_string, '%d.%m.%Y %H:%M:%S.%f')
    sec = dt_obj.timestamp() 
    return sec

def get_start_end_indices(df, start_time, end_time):
    # Get the index where the timestamp is greater than or equal to the start_time
    if len(df[df['GPS Time'] >= start_time]) == 0 or len(df[df['GPS Time'] <= end_time]) == 0:
        return pd.DataFrame()
    start_index = df[df['GPS Time'] >= start_time].index[0]

    # Get the index where the timestamp is less than or equal to the end_time
    end_index = df[df['GPS Time'] <= end_time].index[-1]

    return df[start_index:end_index]



# global vars 

possible_ho = ['NR To EUTRA Redirection Success', 'NR Interfreq Handover Success', 'ulInformationTransferMRDC', 'MCG DRB Success', 'NR SCG Addition Success', 'Mobility from NR to EUTRA Success', 'NR Intrafreq Handover Success', 'NR SCG Modification Success', 'scgFailureInformationNR', 'Handover Success', 'EUTRA To NR Redirection Success']

main_xput_tech_dl_dict = {'Skip' : 0}
main_xput_tech_ul_dict = {'Skip' : 0}
main_xput_tech_ping_dict = {'Skip' : 0}
main_tech_ping_band_dict = {'NSA' : {}, 'SA' : {}}
main_tech_ping_band_scs_dict = {'NSA' : {}, 'SA' : {}}
main_tech_ping_city_info_dict = {'NSA' : {}, 'SA' : {}}
main_tech_ping_rsrp_dict = {'NSA' : [], 'SA' : []}
main_tech_ping_pathloss_dict = {'NSA' : [], 'SA' : []}
main_ping_tx_power_dict = {'NSA' : [], 'SA' : []}
main_ping_tx_power_control_dict = {'NSA' : [], 'SA' : []}

fiveg_xput_tech_dl_dict = {'Skip' : 0}
lte_xput_tech_dl_dict = {'Skip' : 0}
fiveg_xput_tech_ul_dict = {'Skip' : 0}
lte_xput_tech_ul_dict = {'Skip' : 0}

main_lat_lon_tech_df = pd.DataFrame()

main_sa_nsa_lte_tech_time_dict = {}
main_sa_nsa_lte_tech_dist_dict = {}
main_sa_nsa_lte_tech_dist_dict_city = {'5G (SA)' : {}, '5G (NSA)' : {}}
main_sa_nsa_lte_tech_band_dict = {}
main_sa_nsa_lte_tech_ca_dict = {}
main_sa_nsa_lte_tech_ca_ul_dict = {}
main_sa_nsa_lte_tech_city_ca_dict = {}
main_sa_nsa_lte_tech_city_ca_ul_dict = {}
main_sa_nsa_lte_tech_dl_mimo_dict = {}
main_sa_nsa_lte_tech_dl_mimo_layer_dict = {}

main_sa_nsa_lte_tech_ca_band_combo_dict = {'5G (SA)' : {}, '5G (NSA)' : {}}
main_sa_nsa_lte_tech_ca_ul_band_combo_dict = {'5G (SA)' : {}, '5G (NSA)' : {}}
main_sa_nsa_lte_tech_ca_band_xput_dict = {'SA' : {}, 'NSA' : {}}
main_sa_nsa_lte_tech_ca_ul_band_xput_dict = {'SA' : {}, 'NSA' : {}}
main_sa_nsa_tx_power_dict = {'5G (NSA)' : [], '5G (SA)' : []}
main_sa_nsa_tx_power_control_dict = {'5G (NSA)' : [], '5G (SA)' : []}
main_sa_nsa_rx_power_dict = {'5G (NSA)' : [], '5G (SA)' : []}
main_sa_nsa_pathloss_dict = {'5G (NSA)' : [], '5G (SA)' : []}

main_tech_xput_city_info_dict = {'NSA' : {}, 'SA' : {}}
main_tech_xput_tx_power_dict = {'NSA' : [], 'SA' : []}
main_tech_xput_tx_power_control_dict = {'NSA' : [], 'SA' : []}
main_tech_xput_dl_tx_power_dict = {'NSA' : [], 'SA' : []}
main_tech_xput_dl_tx_power_control_dict = {'NSA' : [], 'SA' : []}
main_tech_xput_ul_tx_power_dict = {'NSA' : [], 'SA' : []}
main_tech_xput_ul_tx_power_control_dict = {'NSA' : [], 'SA' : []}
main_tech_xput_dl_bandwidth_dict = {'NSA' : {}, 'SA' : {}}
main_tech_xput_ul_bandwidth_dict = {'NSA' : {}, 'SA' : {}}
main_tech_xput_dl_ca_bandwidth_dict = {'NSA' : {}, 'SA' : {}}
main_tech_xput_ul_ca_bandwidth_dict = {'NSA' : {}, 'SA' : {}}
main_tech_xput_rx_power_dict = {'NSA' : [], 'SA' : []}
main_tech_xput_dl_rx_power_dict = {'NSA' : [], 'SA' : []}
main_tech_xput_ul_rx_power_dict = {'NSA' : [], 'SA' : []}
main_tech_xput_pathloss_dict = {'NSA' : [], 'SA' : []}
main_tech_xput_dl_pathloss_dict = {'NSA' : [], 'SA' : []}
main_tech_xput_ul_pathloss_dict = {'NSA' : [], 'SA' : []}

main_tech_xput_dl_mean_dict = {'5G (NSA)' : [], '5G (SA)' : []}
main_tech_xput_dl_std_dict = {'5G (NSA)' : [], '5G (SA)' : []}
main_tech_xput_ul_mean_dict = {'5G (NSA)' : [], '5G (SA)' : []}
main_tech_xput_ul_std_dict = {'5G (NSA)' : [], '5G (SA)' : []}

main_tech_xput_dl_diff_dict = {'5G (NSA)' : {0.5 : [], 2 : [], 4: []}, '5G (SA)' : {0.5 : [], 2 : [], 4: []}}
main_tech_xput_ul_diff_dict = {'5G (NSA)' : {0.5 : [], 2 : [], 4: []}, '5G (SA)' : {0.5 : [], 2 : [], 4: []}}

lat_lon_count = -1

if os.path.exists('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/pkls/data_2023/city_tech_info.pkl'):
    fh = open("/home/moinakgh/csv_ho/nsa_sa_analysis_perf/pkls/data_2023/city_tech_info.pkl", "rb")
    city_tech_df = pkl.load(fh)
    fh.close()
    city_tech_df['Lat-Lon'] = city_tech_df.apply(lambda row: (row['Lat'], row['Lon']), axis=1)

city_tech_df = city_tech_df.drop_duplicates(subset='Lat-Lon')
lat_lon_city_dict = city_tech_df.set_index('Lat-Lon')['city_info'].to_dict()

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

def extract_sa_nsa_coverage():

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
    
    def return_city_data(lat_lon):
        city_list = []
        
        new_york_city_lat_lon = [(40.757640, -73.985470), (40.696207, -73.984628), (40.721333, -73.998177), (40.723829, -73.996294), (40.858873, -73.872615), (40.570446, -74.109868)]

        distance_from_city = []
        for lat_lng in new_york_city_lat_lon:
            distance_from_city.append(geodesic(lat_lng, lat_lon).miles)
        if min(distance_from_city) <= 5:
            city_list.append('new_york')
        
        boston_city_lat_lon = [(42.356747, -71.057755)]
        distance_from_city = []
        for lat_lng in boston_city_lat_lon:
            distance_from_city.append(geodesic(lat_lng, lat_lon).miles)
        if min(distance_from_city) <= 5:
            city_list.append('boston')

        city_list = list(set(city_list))
        if len(city_list) == 0:
            return None 
        elif len(city_list) == 1:
            return city_list[0]
        else:
            print("WTF!")
            sys.exit(1)


    def extract_drive_trip_2_data():
        global main_lat_lon_tech_df
        global main_sa_nsa_lte_tech_time_dict
        global main_sa_nsa_lte_tech_dist_dict
        global main_sa_nsa_lte_tech_dist_dict_city
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

        fh = open("../pkls/start_end_times_dl_ul_east_coast_trip_2.pkl", "rb")
        downlink_start_end_times, uplink_start_end_times, ping_start_end_times = pkl.load(fh)
        fh.close()

        # trip 2 cst data 
        fh = open("../pkls/start_end_times_dl_ul_cst_trip_2.pkl", "rb")
        downlink_start_end_times_2, uplink_start_end_times_2, ping_start_end_times_2 = pkl.load(fh)
        for op in downlink_start_end_times.keys():
            downlink_start_end_times[op].extend(downlink_start_end_times_2[op])
            uplink_start_end_times[op].extend(uplink_start_end_times_2[op])
        fh.close()

        drive_trip_2_data_path = "../raw_data/data_2023/xcal_kpi_data"
        day_to_date_map = {'day_1' : "2023-05-13 ", 'day_2' : "2023-05-14 ", 'day_3' : "2023-05-15 ", 'day_4' : "2023-05-16 ", 'day_4_cst' : "2023-05-16 "}


        for day in ['day_1', 'day_2', 'day_3', 'day_4', 'day_4_cst']:
            tmobile_template_data = glob.glob(drive_trip_2_data_path + "/*tmobile*%s*csv" %day)
            for template_csv in tmobile_template_data:
                df_template = pd.read_csv(template_csv)

                df_template.drop(df_template.tail(8).index,inplace=True)
                df_template['TIME_STAMP'] = df_template['TIME_STAMP'].apply(datetime_to_timestamp)
                df_template = df_template.rename(columns={'TIME_STAMP' : 'Timestamp'})
                df_template['GPS Time'] = [day_to_date_map[day] + i + ".000000" if not pd.isnull(i) else i for i in df_template['GPS Time']]
                df_template['GPS Time'] = df_template['GPS Time'].apply(datetime_to_timestamp)
                df_template = df_template.sort_values("Timestamp").reset_index(drop=True)

                # Assuming df_template is your dataframe and unix_timestamp_list is your list of timestamps
                # Convert 'GPS Time' in df_template to datetime if not already
                df_template['GPS Time'] = pd.to_datetime(df_template['GPS Time'], unit='s', errors='coerce')

                # List of Unix timestamps to iterate over
                # Assuming unix_timestamp_list is already in Unix format
                unix_timestamp_list = [i[0] for i in downlink_start_end_times['tmobile']]  # Replace with your list

                # Convert the list of Unix timestamps to datetime
                timestamp_list = pd.to_datetime(unix_timestamp_list, unit='s')

                # Initialize an empty list to store the indices of the closest rows
                closest_indices = []

                # Iterate over each timestamp
                for ts in timestamp_list:
                    if ts.date() != df_template['GPS Time'].dropna().iloc[0].date():
                        continue
                    # Find the row index with the closest 'GPS Time'
                    closest_idx = (df_template['GPS Time'] - ts).abs().idxmin()
                    closest_indices.append(closest_idx)


                # Loop through the index list and slice the dataframe
                closest_indices = sorted(list(set(closest_indices)))
                for i in range(len(closest_indices) - 1):
                    start_idx = closest_indices[i]
                    end_idx = closest_indices[i + 1]
                    
                    # Slice the dataframe from start_idx to end_idx (not including end_idx)
                    subframe = df_template.iloc[start_idx:end_idx - 1]
                    
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
                            
                            ca_lat_lon_df = df_template[['Lat', 'Lon', '5G KPI Total Info DL CA Type']].loc[sf.index[0]:sf.index[-1]].dropna()
                            ca_lat_lon_df['Lat-Lon'] = ca_lat_lon_df.apply(lambda row: (row['Lat'], row['Lon']), axis=1)
                            ca_lat_lon_df['city_data'] = ca_lat_lon_df['Lat-Lon'].apply(return_city_data)
                            boston_filtered_df = ca_lat_lon_df[ca_lat_lon_df['city_data'] == 'boston']
                            if len(boston_filtered_df) != 0:
                                if 'boston' not in main_sa_nsa_lte_tech_city_ca_dict[sf['Event Technology'].iloc[0]].keys():
                                    main_sa_nsa_lte_tech_city_ca_dict[sf['Event Technology'].iloc[0]]['boston'] = []
                                main_sa_nsa_lte_tech_city_ca_dict[sf['Event Technology'].iloc[0]]['boston'].extend(list(boston_filtered_df['5G KPI Total Info DL CA Type']))
                              
                            new_york_filtered_df = ca_lat_lon_df[ca_lat_lon_df['city_data'] == 'new_york']
                            if len(new_york_filtered_df) != 0:
                                if 'new_york' not in main_sa_nsa_lte_tech_city_ca_dict[sf['Event Technology'].iloc[0]].keys():
                                    main_sa_nsa_lte_tech_city_ca_dict[sf['Event Technology'].iloc[0]]['new_york'] = []
                                main_sa_nsa_lte_tech_city_ca_dict[sf['Event Technology'].iloc[0]]['new_york'].extend(list(new_york_filtered_df['5G KPI Total Info DL CA Type']))

                                

                            main_sa_nsa_lte_tech_dl_mimo_dict[sf['Event Technology'].iloc[0]].extend(list(df_template['5G KPI PCell Layer1 DL MIMO'].loc[sf.index[0]:sf.index[-1]].dropna()))
                            main_sa_nsa_lte_tech_dl_mimo_layer_dict[sf['Event Technology'].iloc[0]].extend(list(df_template['5G KPI PCell Layer1 DL Layer Num (Mode)'].loc[sf.index[0]:sf.index[-1]].dropna()))

                            ca_band_df = df_template[['5G KPI PCell RF Band', '5G KPI SCell[1] RF Band', '5G KPI Total Info DL CA Type']].loc[sf.index[0]:sf.index[-1]]
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
                            # path_loss_df['city_info'] = path_loss_df['Lat-Lon'].apply(extrapolate_road_data_new)
                            path_loss_df['city_info'] = path_loss_df['Lat-Lon'].apply(return_city_info)
                            path_loss_df['city_info'] = path_loss_df['city_info'].fillna(method='ffill').fillna(method='bfill')


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
                            # sum_dist+=geopy.distance.geodesic([lat, lon], prev_lat_lon).miles
                            # prev_lat_lon = [lat, lon]
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
                            city = path_loss_df['city_info'].iloc[len(path_loss_df) // 2]
                            if city not in main_sa_nsa_lte_tech_dist_dict_city[sf['Event Technology'].iloc[0]].keys():
                                main_sa_nsa_lte_tech_dist_dict_city[sf['Event Technology'].iloc[0]][city] = []
                            main_sa_nsa_lte_tech_dist_dict_city[sf['Event Technology'].iloc[0]][city].append(sum_dist)

                # To handle the last slice from the last index to the end of the dataframe
                subframe = df_template.iloc[closest_indices[-1]:]

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
                if len(subframe) != 0:
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
                            main_sa_nsa_lte_tech_band_dict[sf['Event Technology'].iloc[0]] = []
                            main_sa_nsa_lte_tech_ca_dict[sf['Event Technology'].iloc[0]] = []
                            main_sa_nsa_lte_tech_ca_ul_dict[sf['Event Technology'].iloc[0]] = []
                            main_sa_nsa_lte_tech_dl_mimo_dict[sf['Event Technology'].iloc[0]] = []
                            main_sa_nsa_lte_tech_dl_mimo_layer_dict[sf['Event Technology'].iloc[0]] = []
                            main_sa_nsa_lte_tech_city_ca_dict[sf['Event Technology'].iloc[0]] = {}
                        
                        if '5G' in sf['Event Technology'].iloc[0]:
                            main_sa_nsa_lte_tech_band_dict[sf['Event Technology'].iloc[0]].extend(list(df_template['5G KPI PCell RF Band'].loc[sf.index[0]:sf.index[-1]].dropna()))
                            main_sa_nsa_lte_tech_ca_dict[sf['Event Technology'].iloc[0]].extend(list(df_template['5G KPI Total Info DL CA Type'].loc[sf.index[0]:sf.index[-1]].dropna()))
                            main_sa_nsa_lte_tech_ca_ul_dict[sf['Event Technology'].iloc[0]].extend(list(df_template['5G KPI Total Info UL CA Type'].loc[sf.index[0]:sf.index[-1]].dropna()))

                            ca_lat_lon_df = df_template[['Lat', 'Lon', '5G KPI Total Info DL CA Type']].loc[sf.index[0]:sf.index[-1]].dropna()
                            ca_lat_lon_df['Lat-Lon'] = ca_lat_lon_df.apply(lambda row: (row['Lat'], row['Lon']), axis=1)
                            ca_lat_lon_df['city_data'] = ca_lat_lon_df['Lat-Lon'].apply(return_city_data)
                            boston_filtered_df = ca_lat_lon_df[ca_lat_lon_df['city_data'] == 'boston']
                            if len(boston_filtered_df) != 0:
                                if 'boston' not in main_sa_nsa_lte_tech_city_ca_dict[sf['Event Technology'].iloc[0]].keys():
                                    main_sa_nsa_lte_tech_city_ca_dict[sf['Event Technology'].iloc[0]]['boston'] = []
                                main_sa_nsa_lte_tech_city_ca_dict[sf['Event Technology'].iloc[0]]['boston'].extend(list(boston_filtered_df['5G KPI Total Info DL CA Type']))
                            new_york_filtered_df = ca_lat_lon_df[ca_lat_lon_df['city_data'] == 'new_york']
                            if len(new_york_filtered_df) != 0:
                                if 'new_york' not in main_sa_nsa_lte_tech_city_ca_dict[sf['Event Technology'].iloc[0]].keys():
                                    main_sa_nsa_lte_tech_city_ca_dict[sf['Event Technology'].iloc[0]]['new_york'] = []
                                main_sa_nsa_lte_tech_city_ca_dict[sf['Event Technology'].iloc[0]]['new_york'].extend(list(new_york_filtered_df['5G KPI Total Info DL CA Type']))

                            main_sa_nsa_lte_tech_dl_mimo_dict[sf['Event Technology'].iloc[0]].extend(list(df_template['5G KPI PCell Layer1 DL MIMO'].loc[sf.index[0]:sf.index[-1]].dropna()))
                            main_sa_nsa_lte_tech_dl_mimo_layer_dict[sf['Event Technology'].iloc[0]].extend(list(df_template['5G KPI PCell Layer1 DL Layer Num (Mode)'].loc[sf.index[0]:sf.index[-1]].dropna()))

                            ca_band_df = df_template[['5G KPI PCell RF Band', '5G KPI SCell[1] RF Band', '5G KPI Total Info DL CA Type']].loc[sf.index[0]:sf.index[-1]]
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

                            # add city the tx and rx power data
                            path_loss_df = sf[['Lat', 'Lon', '5G KPI PCell RF Pathloss [dB]', '5G KPI PCell RF PUSCH Power [dBm]', '5G KPI PCell RF PUCCH Power [dBm]', '5G KPI PCell RF Serving SS-RSRP [dBm]']]
                            path_loss_df['Lat-Lon'] = path_loss_df.apply(lambda row: (row['Lat'], row['Lon']), axis=1)
                            # path_loss_df['city_info'] = path_loss_df['Lat-Lon'].apply(extrapolate_road_data_new)
                            path_loss_df['city_info'] = path_loss_df['Lat-Lon'].apply(return_city_info)
                            path_loss_df['city_info'] = path_loss_df['city_info'].fillna(method='ffill').fillna(method='bfill')
                            # grouped = path_loss_df.groupby('city_info')
 
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
                            # sum_dist+=geopy.distance.geodesic([lat, lon], prev_lat_lon).miles
                            # prev_lat_lon = [lat, lon]

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
                            city = path_loss_df['city_info'].iloc[len(path_loss_df) // 2]
                            if city not in main_sa_nsa_lte_tech_dist_dict_city[sf['Event Technology'].iloc[0]].keys():
                                main_sa_nsa_lte_tech_dist_dict_city[sf['Event Technology'].iloc[0]][city] = []
                            main_sa_nsa_lte_tech_dist_dict_city[sf['Event Technology'].iloc[0]][city].append(sum_dist)


                lat_lon_tech_df = df_template[['Timestamp', 'Lat', 'Lon', 'Event Technology']].dropna()

                lat_lon_tech_df['Event Technology'] = lat_lon_tech_df['Event Technology'].replace('5G-NR', '5G-NR_SA')
                lat_lon_tech_df['Event Technology'] = lat_lon_tech_df['Event Technology'].replace('5G-NR(2CA)', '5G-NR_SA')

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
        extract_drive_trip_2_data()
        global main_lat_lon_tech_df, main_sa_nsa_lte_tech_time_dict, main_sa_nsa_lte_tech_dist_dict, main_sa_nsa_lte_tech_dist_dict_city, main_sa_nsa_lte_tech_band_dict, main_sa_nsa_lte_tech_ca_dict, main_sa_nsa_lte_tech_ca_ul_dict, main_sa_nsa_lte_tech_ca_band_combo_dict, main_sa_nsa_lte_tech_ca_ul_band_combo_dict, main_sa_nsa_tx_power_dict, main_sa_nsa_tx_power_control_dict, main_sa_nsa_rx_power_dict, main_sa_nsa_pathloss_dict, main_sa_nsa_lte_tech_dl_mimo_dict, main_sa_nsa_lte_tech_dl_mimo_layer_dict, main_sa_nsa_lte_tech_city_ca_dict
        fh = open("../pkls/data_2023/coverage_data.pkl", "wb")
        pkl.dump([main_lat_lon_tech_df, main_sa_nsa_lte_tech_time_dict, main_sa_nsa_lte_tech_dist_dict, main_sa_nsa_lte_tech_dist_dict_city, main_sa_nsa_lte_tech_band_dict, main_sa_nsa_lte_tech_ca_dict, main_sa_nsa_lte_tech_ca_ul_dict, main_sa_nsa_lte_tech_ca_band_combo_dict, main_sa_nsa_lte_tech_ca_ul_band_combo_dict, main_sa_nsa_tx_power_dict, main_sa_nsa_tx_power_control_dict, main_sa_nsa_rx_power_dict, main_sa_nsa_pathloss_dict, main_sa_nsa_lte_tech_dl_mimo_dict, main_sa_nsa_lte_tech_dl_mimo_layer_dict, main_sa_nsa_lte_tech_city_ca_dict], fh)
        fh.close()

    # print sa nsa ca band combo 
    tech_ca_band_combo_dist_dict = {}
    for tech in main_sa_nsa_lte_tech_ca_band_combo_dict.keys():
        tech_ca_band_combo_dist_dict[tech] = {}
        for ca_type in main_sa_nsa_lte_tech_ca_band_combo_dict[tech].keys():
            tech_ca_band_combo_dist_dict[tech][ca_type] = calculate_percentage_of_occurrence(main_sa_nsa_lte_tech_ca_band_combo_dict[tech][ca_type])

    # get big city or small city information 
    main_lat_lon_tech_df['Lat-Lon'] = main_lat_lon_tech_df.apply(lambda row: (row['Lat'], row['Lon']), axis=1)
    # if not os.path.exists("/home/moinakgh/csv_ho/nsa_sa_analysis_perf/pkls/data_2023/city_tech_info.pkl"):
    if 1:
        main_lat_lon_tech_df['city_info'] = main_lat_lon_tech_df['Lat-Lon'].apply(extrapolate_road_data_new)
        city_tech_df = main_lat_lon_tech_df[['Lat', 'Lon', 'city_info', 'Event Technology']].dropna()
        fh = open("../pkls/data_2023/city_tech_info.pkl", "wb")
        city_tech_df = pkl.dump(city_tech_df, fh)
        fh.close()


def extract_overall_xput_dist():


    def extract_drive_trip_2_data():
        global main_xput_tech_dl_dict
        global main_xput_tech_ul_dict
        # get the tput  test times
        fh = open("../pkls/start_end_times_dl_ul_east_coast_trip_2.pkl", "rb")
        downlink_start_end_times, uplink_start_end_times, ping_start_end_times = pkl.load(fh)
        fh.close()

        # trip 2 cst data 
        fh = open("../pkls/start_end_times_dl_ul_cst_trip_2.pkl", "rb")
        downlink_start_end_times_2, uplink_start_end_times_2, ping_start_end_times_2 = pkl.load(fh)
        for op in downlink_start_end_times.keys():
            downlink_start_end_times[op].extend(downlink_start_end_times_2[op])
            uplink_start_end_times[op].extend(uplink_start_end_times_2[op])
        fh.close()

        drive_trip_2_data_path = "../raw_data/data_2023/xcal_kpi_data"


        day_to_date_map = {'day_1' : "2023-05-13 ", 'day_2' : "2023-05-14 ", 'day_3' : "2023-05-15 ", 'day_4' : "2023-05-16 ", 'day_4_cst' : "2023-05-16 "}


        for day in ['day_1', 'day_2', 'day_3', 'day_4', 'day_4_cst']:
            tmobile_template_data = glob.glob(drive_trip_2_data_path + "/*tmobile*%s*csv" %day)
            for template_csv in tmobile_template_data:
                df_template = pd.read_csv(template_csv)

                df_template.drop(df_template.tail(8).index,inplace=True)
                df_template['TIME_STAMP'] = df_template['TIME_STAMP'].apply(datetime_to_timestamp)
                
                # prev_timestamp = list(df_template['TIME_STAMP'])[0]
                # df_new['Diff_Timestamp'] = df_new['TIME_STAMP'].apply(get_prev_timestamp_diff)
                # df_new = df_new.rename(columns={'TIME_STAMP' : 'Timestamp'})
                # df_new["GAP_INFO"] = ["gap" if item > 2 else "" for i, item in enumerate(list(df_new['Diff_Timestamp']))]

                df_template = df_template.rename(columns={'TIME_STAMP' : 'Timestamp'})
                df_template['GPS Time'] = [day_to_date_map[day] + i + ".000000" if not pd.isnull(i) else i for i in df_template['GPS Time']]
                df_template['GPS Time'] = df_template['GPS Time'].apply(datetime_to_timestamp)

                df_template = df_template.sort_values("Timestamp").reset_index(drop=True)
                # work with downlink first
                for start_end_time in downlink_start_end_times['tmobile']:
                    start_time, end_time = start_end_time
                    sub_df = get_start_end_indices(df_template, start_time, end_time)
                    if len(sub_df) == 0:
                        continue
                    sub_df = sub_df.loc[sub_df['Event Technology'].dropna().index[0]:]
                    sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR', '5G-NR_SA')
                    sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR(2CA)', '5G-NR_SA')
                    

                    # Step 1: Identify rows where LTE Event or 5G Event match any event in possible_ho
                    ho_rows = sub_df[(sub_df['Event LTE Events'].isin(possible_ho)) | (sub_df['Event 5G-NR Events'].isin(possible_ho))]

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
                for start_end_time in uplink_start_end_times['tmobile']:
                    start_time, end_time = start_end_time
                    sub_df = get_start_end_indices(df_template, start_time, end_time)
                    if len(sub_df) == 0:
                        continue
                    sub_df = sub_df.loc[sub_df['Event Technology'].dropna().index[0]:]
                    sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR', '5G-NR_SA')
                    sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR(2CA)', '5G-NR_SA')
                    

                    # Step 1: Identify rows where LTE Event or 5G Event match any event in possible_ho
                    ho_rows = sub_df[(sub_df['Event LTE Events'].isin(possible_ho)) | (sub_df['Event 5G-NR Events'].isin(possible_ho))]

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
        
    extract_drive_trip_2_data()

    fh = open("../pkls/data_2023/overal_xput.pkl", "wb")
    pkl.dump([main_xput_tech_dl_dict, main_xput_tech_ul_dict], fh)
    fh.close()

def extract_5G_lte_break_xput_dist():
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
    

    def extract_drive_trip_2_data():
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
        try:
            global city_tech_df
        except:
            city_tech_df = None 
        # get the tput  test times
        fh = open("../pkls/start_end_times_dl_ul_east_coast_trip_2.pkl", "rb")
        downlink_start_end_times, uplink_start_end_times, ping_start_end_times = pkl.load(fh)
        fh.close()

        # trip 2 cst data 
        fh = open("../pkls/start_end_times_dl_ul_cst_trip_2.pkl", "rb")
        downlink_start_end_times_2, uplink_start_end_times_2, ping_start_end_times_2 = pkl.load(fh)
        for op in downlink_start_end_times.keys():
            downlink_start_end_times[op].extend(downlink_start_end_times_2[op])
            uplink_start_end_times[op].extend(uplink_start_end_times_2[op])
        fh.close()

        drive_trip_2_data_path = "../raw_data/data_2023/xcal_kpi_data"


        day_to_date_map = {'day_1' : "2023-05-13 ", 'day_2' : "2023-05-14 ", 'day_3' : "2023-05-15 ", 'day_4' : "2023-05-16 ", 'day_4_cst' : "2023-05-16 "}


        for day in ['day_1', 'day_2', 'day_3', 'day_4', 'day_4_cst']:
            tmobile_template_data = glob.glob(drive_trip_2_data_path + "/*tmobile*%s*csv" %day)
            for template_csv in tmobile_template_data:
                df_template = pd.read_csv(template_csv)

                df_template.drop(df_template.tail(8).index,inplace=True)
                df_template['TIME_STAMP'] = df_template['TIME_STAMP'].apply(datetime_to_timestamp)
                df_template = df_template.rename(columns={'TIME_STAMP' : 'Timestamp'})
                df_template['GPS Time'] = [day_to_date_map[day] + i + ".000000" if not pd.isnull(i) else i for i in df_template['GPS Time']]
                df_template['GPS Time'] = df_template['GPS Time'].apply(datetime_to_timestamp)
                df_template = df_template.sort_values("Timestamp").reset_index(drop=True)
                if 0:
                    lte_dl_cols = []
                    lte_ul_cols = []
                    fiveg_dl_cols = []
                    fiveg_ul_cols = []
                    cols = df_template.columns
                    for col in cols:
                        if 'LTE' in col and 'Throughput' in col and 'MAC' in col and 'DL':
                            lte_dl_cols.append(col)
                        if 'LTE' in col and 'Throughput' in col and 'MAC' in col and 'UL':
                            lte_ul_cols.append(col)
                        if '5G' in col and 'Throughput' in col and 'MAC' in col and 'DL':
                            fiveg_dl_cols.append(col)
                        if '5G' in col and 'Throughput' in col and 'MAC' in col and 'UL':
                            fiveg_ul_cols.append(col)
                # work with downlink first
                for start_end_time in downlink_start_end_times['tmobile']:
                    start_time, end_time = start_end_time
                    sub_df = get_start_end_indices(df_template, start_time, end_time)
                    if len(sub_df) == 0:
                        continue
                    sub_df = sub_df.loc[sub_df['Event Technology'].dropna().index[0]:]
                    sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR', '5G-NR_SA')
                    sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR(2CA)', '5G-NR_SA')
                    

                    # Step 1: Identify rows where LTE Event or 5G Event match any event in possible_ho
                    ho_rows = sub_df[(sub_df['Event LTE Events'].isin(possible_ho)) | (sub_df['Event 5G-NR Events'].isin(possible_ho))]

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
                    mac_cell_xput_cols = [i for i in sub_df.columns if 'MAC DL Throughput' in i and '5G' in i and 'Total' not in i]
                    bw_cols = [i for i in sub_df.columns if 'RF BandWidth' in i and '5G' in i]
                    band_cols = [i for i in sub_df.columns if 'RF Band' in i and '5G' in i and 'RF BandWidth' not in i]

                    all_cols = ['Lat', 'Lon', 'Timestamp',  'city_info', 'Event Technology', '5G KPI Total Info Layer2 MAC DL Throughput [Mbps]', '5G KPI Total Info DL CA Type', '5G KPI PCell RF Serving SS-RSRP [dBm]', '5G KPI PCell RF PUSCH Power [dBm]', '5G KPI PCell RF PUCCH Power [dBm]', '5G KPI PCell RF Pathloss [dB]', 'Smart Phone Smart Throughput Mobile Network DL Throughput [Mbps]']
                    all_cols.extend(mac_cell_xput_cols)
                    all_cols.extend(bw_cols)
                    all_cols.extend(band_cols)
                    xput_data = sub_df[all_cols]
                    xput_data = xput_data.dropna(subset=['Event Technology', '5G KPI Total Info Layer2 MAC DL Throughput [Mbps]'])

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

                                            a = 1
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
                                if x == 'Skip' or x > 0:
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
                                    if x == 'Skip' or x > 0:
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
                    xput_data = sub_df[['Event Technology', 'LTE KPI PCell MAC DL Throughput[Mbps]']].dropna()
                    for xput_tech, xput in zip(xput_data['Event Technology'], xput_data['LTE KPI PCell MAC DL Throughput[Mbps]']):
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

                      
        # work with uplink then 
                for start_end_time in uplink_start_end_times['tmobile']:
                    start_time, end_time = start_end_time
                    sub_df = get_start_end_indices(df_template, start_time, end_time)
                    if len(sub_df) == 0:
                        continue
                    sub_df = sub_df.loc[sub_df['Event Technology'].dropna().index[0]:]
                    sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR', '5G-NR_SA')
                    sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR(2CA)', '5G-NR_SA')
                    

                    # Step 1: Identify rows where LTE Event or 5G Event match any event in possible_ho
                    ho_rows = sub_df[(sub_df['Event LTE Events'].isin(possible_ho)) | (sub_df['Event 5G-NR Events'].isin(possible_ho))]

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



                    original_list = [xput_data['Timestamp'],xput_data['Lat'], xput_data['Lon'], xput_data['Event Technology'], xput_data['5G KPI Total Info Layer2 MAC UL Throughput [Mbps]'], xput_data['5G KPI Total Info UL CA Type'], xput_data['5G KPI PCell RF Serving SS-RSRP [dBm]'], xput_data['5G KPI PCell RF PUSCH Power [dBm]'], xput_data['5G KPI PCell RF PUCCH Power [dBm]'], xput_data['5G KPI PCell RF Pathloss [dB]'], xput_data['city_info']]
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
                                if x == 'Skip' or x > 0:
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
                                    if x == 'Skip' or x > 0:
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
                    xput_data = sub_df[['Event Technology', 'LTE KPI PCell MAC UL Throughput[Mbps]']].dropna()
                    for xput_tech, xput in zip(xput_data['Event Technology'], xput_data['LTE KPI PCell MAC UL Throughput[Mbps]']):
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



    extract_drive_trip_2_data()

    fh = open("../pkls/data_2023/xput_break.pkl", "wb")
    pkl.dump([fiveg_xput_tech_dl_dict, lte_xput_tech_dl_dict, fiveg_xput_tech_ul_dict, lte_xput_tech_ul_dict, main_sa_nsa_lte_tech_ca_band_xput_dict, main_sa_nsa_lte_tech_ca_ul_band_xput_dict], fh)
    fh.close()


    fh = open("../pkls/data_2023/xput_break_part_2.pkl", "wb")
    pkl.dump([main_tech_xput_city_info_dict, main_tech_xput_tx_power_dict, main_tech_xput_tx_power_control_dict, main_tech_xput_dl_tx_power_dict, main_tech_xput_dl_tx_power_control_dict, main_tech_xput_ul_tx_power_dict, main_tech_xput_ul_tx_power_control_dict, main_tech_xput_rx_power_dict, main_tech_xput_dl_rx_power_dict, main_tech_xput_ul_rx_power_dict, main_tech_xput_pathloss_dict, main_tech_xput_dl_pathloss_dict, main_tech_xput_ul_pathloss_dict, main_tech_xput_dl_bandwidth_dict, main_tech_xput_ul_bandwidth_dict, main_tech_xput_dl_mean_dict, main_tech_xput_dl_std_dict, main_tech_xput_ul_mean_dict, main_tech_xput_ul_std_dict, main_tech_xput_dl_diff_dict, main_tech_xput_ul_diff_dict, main_tech_xput_dl_ca_bandwidth_dict, main_tech_xput_ul_ca_bandwidth_dict], fh)
    fh.close()


def extract_overall_latency_dist():

    def return_city_info(lat_lon):
        global lat_lon_city_dict
        if lat_lon not in lat_lon_city_dict.keys():
            return None
        else:
            return lat_lon_city_dict[lat_lon]


    sub_cities_df = pd.read_csv('../raw_data/us_cities_of_interest.csv')
    def extrapolate_road_data_new(original_lat_lon):
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
        
    def extract_drive_trip_2_data():
        global main_xput_tech_ping_dict
        global main_tech_ping_band_dict
        global main_tech_ping_city_info_dict
        global main_tech_ping_rsrp_dict
        global main_tech_ping_pathloss_dict
        global main_ping_tx_power_dict
        global main_ping_tx_power_control_dict
        global main_tech_ping_band_scs_dict
        try:
            global city_tech_df
        except:
            city_tech_df = None 
        # get the ping test times
        fh = open("../pkls/ping_data_drive_trip_2.pkl", "rb")
        ping_data_times = pkl.load(fh)
        fh.close()

        # trip 2 cst data 
        fh = open("../pkls/ping_data_drive_trip_2_cst.pkl", "rb")
        ping_data_times_2 = pkl.load(fh)
        for op in ping_data_times.keys():
            ping_data_times[op].extend(ping_data_times_2[op])
        fh.close()

        drive_trip_2_data_path = "../raw_data/data_2023/xcal_kpi_data"


        day_to_date_map = {'day_1' : "2023-05-13 ", 'day_2' : "2023-05-14 ", 'day_3' : "2023-05-15 ", 'day_4' : "2023-05-16 ", 'day_4_cst' : "2023-05-16 "}


        for day in ['day_1', 'day_2', 'day_3', 'day_4', 'day_4_cst']:
            tmobile_template_data = glob.glob(drive_trip_2_data_path + "/*tmobile*%s*csv" %day)
            for template_csv in tmobile_template_data:
                df_template = pd.read_csv(template_csv)

                df_template.drop(df_template.tail(8).index,inplace=True)
                df_template['TIME_STAMP'] = df_template['TIME_STAMP'].apply(datetime_to_timestamp)
                
                df_template = df_template.rename(columns={'TIME_STAMP' : 'Timestamp'})
                df_template['GPS Time'] = [day_to_date_map[day] + i + ".000000" if not pd.isnull(i) else i for i in df_template['GPS Time']]
                df_template['GPS Time'] = df_template['GPS Time'].apply(datetime_to_timestamp)

                df_template = df_template.sort_values("Timestamp").reset_index(drop=True)
                # work with ping first
                for start_end_time in ping_data_times['tmobile']:
                    start_time, end_time, data = start_end_time
                    sub_df = get_start_end_indices(df_template, start_time, end_time)
                    if len(sub_df) == 0:
                        continue
                    sub_df = sub_df.loc[sub_df['Event Technology'].dropna().index[0]:]
                    sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR', '5G-NR_SA')
                    sub_df['Event Technology'] = sub_df['Event Technology'].replace('5G-NR(2CA)', '5G-NR_SA')
                    
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

                    ping_df = pd.DataFrame({'GPS Time': ping_timestamp, 'ping_data': ping_data})
                    sub_df['GPS Time'] = sub_df['GPS Time'].fillna(method='ffill')

                    sub_df = pd.concat([sub_df, ping_df])
                    sub_df = sub_df.sort_values(by=["GPS Time", "Timestamp"])
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

                    # 
                    ping_df_data = sub_df[['Lat', 'Lon', 'city_info', 'Event Technology', 'ping_data', '5G KPI PCell RF Band', '5G KPI PCell RF Serving SS-RSRP [dBm]', '5G KPI PCell RF PUSCH Power [dBm]', '5G KPI PCell RF PUCCH Power [dBm]', '5G KPI PCell RF Pathloss [dB]', '5G KPI PCell RF Subcarrier Spacing']].dropna()
                    for ping_tech, ping, band, lat, lon, rsrp, tx_power, tx_power_control, pathloss, city_info, scs in zip(ping_df_data['Event Technology'], ping_df_data['ping_data'], ping_df_data['5G KPI PCell RF Band'], ping_df_data['Lat'], ping_df_data['Lon'], ping_df_data['5G KPI PCell RF Serving SS-RSRP [dBm]'], ping_df_data['5G KPI PCell RF PUSCH Power [dBm]'], ping_df_data['5G KPI PCell RF PUCCH Power [dBm]'], ping_df_data['5G KPI PCell RF Pathloss [dB]'], ping_df_data['city_info'], ping_df_data['5G KPI PCell RF Subcarrier Spacing']):
                        if ping == 'Skip':
                            main_xput_tech_ping_dict['Skip']+=1
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
                            main_xput_tech_ping_dict[ping_tech].append(ping)

                            if ping_tech != 'LTE':
                                # get band info
                                if band not in main_tech_ping_band_dict[ping_tech].keys():
                                    main_tech_ping_band_dict[ping_tech][band] = []
                                    main_tech_ping_band_scs_dict[ping_tech][band] = []

                                main_tech_ping_band_dict[ping_tech][band].append(ping)
                                main_tech_ping_band_scs_dict[ping_tech][band].append(scs)

                                # get rsrp data 
                                if 1:
                                    main_tech_ping_rsrp_dict[ping_tech].append(rsrp)
                                    main_tech_ping_pathloss_dict[ping_tech].append(pathloss)
                                    main_ping_tx_power_dict[ping_tech].append(tx_power)
                                    main_ping_tx_power_control_dict[ping_tech].append(tx_power_control)


    if 1:
        extract_drive_trip_2_data()

        fh = open("../pkls/data_2023/rtt_break.pkl", "wb")
        pkl.dump([main_xput_tech_ping_dict, main_tech_ping_band_dict, main_tech_ping_city_info_dict, main_tech_ping_rsrp_dict, main_tech_ping_pathloss_dict, main_ping_tx_power_dict, main_ping_tx_power_control_dict, main_tech_ping_band_scs_dict], fh)
        fh.close()


def get_ho_duration():
    drive_trip_2_base = "../raw_data/data_2023/ho_duration_data/"

    ho_duration_files_trip_2 = glob.glob(drive_trip_2_base + "*.xlsx")

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

    for ho_duration_files in [ho_duration_files_trip_2]:
        for ho_duration_file in ho_duration_files:
            df_template = pd.read_excel(ho_duration_file)

            df_template.drop(df_template.tail(8).index,inplace=True)
            df_template['TIME_STAMP'] = df_template['TIME_STAMP'].apply(datetime_to_timestamp_new)
            
            df_template = df_template.rename(columns={'TIME_STAMP' : 'Timestamp'})
            df_template = df_template.sort_values("Timestamp").reset_index(drop=True)


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
                        else:
                            nsa_nr_nr_duration_inter_list.append(list(tple[0]['5G-NR RRC NR SCG Mobility Statistics NR SCG Modification[NR to NR] Duration [sec]'].dropna())[0])

                        try:
                            temp_inter_intra = return_nsa_ho_type(before_pci_count_max_occurrence, after_pci_count_max_occurrence, before_arfcn_count_max_occurrence, after_arfcn_count_max_occurrence)
                            if temp_inter_intra != "":
                                nsa_inter_intra_dict[temp_inter_intra].append(list(tple[0]['5G-NR RRC NR SCG Mobility Statistics NR SCG Modification[NR to NR] Duration [sec]'].dropna())[0])
                        except:
                            a = 1
                    except:
                        continue 

            except:
                import traceback
                traceback.print_exc()
            
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
                a = 1


    fh = open("../pkls/data_2023/ho_duration.pkl", "wb")
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

# function calls 
if 1:
    get_ho_duration()
    extract_sa_nsa_coverage()
    extract_overall_xput_dist()
    extract_5G_lte_break_xput_dist()
    extract_overall_latency_dist()

