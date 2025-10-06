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
import random
import matplotlib.patches as mpatches
from pprint import pprint
from geopy.distance import distance
from geopy.distance import geodesic
from shapely.geometry import Point
from collections import Counter
from collections import defaultdict
from earfcn.convert import earfcn2freq, earfcn2band, freq2earfcn
from timezonefinder import TimezoneFinder
obj = TimezoneFinder()
import matplotlib.patches as patches
from pprint import pprint
from collections import defaultdict
import math
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

original_rc = mpl.rcParams.copy()

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
    
# Function to calculate the 95th percentile
def get_percentile(data, percentile):
    return np.percentile(data, percentile)

def calculate_percentage_of_occurrence(lst):
    total_elements = len(lst)
    
    # Use Counter to count the occurrence of each element
    element_counts = Counter(lst)

    # Calculate the percentage for each unique element
    percentage_dict = {element: round((count / total_elements) * 100, 2) for element, count in element_counts.items()}

    return percentage_dict

def calculate_percentage_of_occurrence_abs(lst):
    total_elements = len(lst)
    
    # Use Counter to count the occurrence of each element
    element_counts = Counter(lst)

    # Calculate the percentage for each unique element
    percentage_dict = {element: count for element, count in element_counts.items()}

    return percentage_dict

main_band_colors = None
main_band_hatches = None
fh = open("../pkls/data_2023/city_tech_info.pkl", "rb")
city_tech_df_1 = pkl.load(fh)
city_tech_df_1['Lat-Lon'] = city_tech_df_1.apply(lambda row: (row['Lat'], row['Lon']), axis=1)
city_tech_df_1['Timezone'] = city_tech_df_1['Lat-Lon'].apply(get_tz_info)
fh.close()

fh = open("/home/moinakgh/csv_ho/nsa_sa_analysis_perf/pkls/driving_trip_lax_bos_2024/city_tech_info.pkl", "rb")
city_tech_df_3 = pkl.load(fh)
city_tech_df_3['Lat-Lon'] = city_tech_df_3.apply(lambda row: (row['Lat'], row['Lon']), axis=1)
city_tech_df_3['Timezone'] = city_tech_df_3['Lat-Lon'].apply(get_tz_info)
fh.close()


fh = open("../pkls/data_2023/coverage_data.pkl", "rb")
main_lat_lon_tech_df_1, main_sa_nsa_lte_tech_time_dict_1, main_sa_nsa_lte_tech_dist_dict_1, main_sa_nsa_lte_tech_dist_dict_city_1, main_sa_nsa_lte_tech_band_dict_1, main_sa_nsa_lte_tech_ca_dict_1, main_sa_nsa_lte_tech_ca_ul_dict_1, main_sa_nsa_lte_tech_ca_band_combo_dict_1, main_sa_nsa_lte_tech_ca_ul_band_combo_dict_1, main_sa_nsa_tx_power_dict_1, main_sa_nsa_tx_power_control_dict_1, main_sa_nsa_rx_power_dict_1, main_sa_nsa_pathloss_dict_1, main_sa_nsa_lte_tech_dl_mimo_dict_1, main_sa_nsa_lte_tech_dl_mimo_layer_dict_1, main_sa_nsa_lte_tech_city_ca_dict_1 = pkl.load(fh)
fh.close()


fh = open("/home/moinakgh/csv_ho/nsa_sa_analysis_perf/pkls/driving_trip_lax_bos_2024/coverage_data.pkl", "rb")
main_lat_lon_tech_df_3, main_sa_nsa_lte_tech_time_dict_3, main_sa_nsa_lte_tech_dist_dict_3, main_sa_nsa_lte_tech_dist_dict_tz_3, main_sa_nsa_lte_tech_dist_dict_city_3, main_sa_nsa_lte_tech_dist_dict_chic_bos_3, main_sa_nsa_lte_tech_band_dict_3, main_sa_nsa_lte_tech_ca_dict_3, main_sa_nsa_lte_tech_ca_ul_dict_3, main_sa_nsa_lte_tech_ca_band_combo_dict_3, main_sa_nsa_lte_tech_ca_ul_band_combo_dict_3, main_sa_nsa_tx_power_dict_3, main_sa_nsa_tx_power_control_dict_3, main_sa_nsa_rx_power_dict_3, main_sa_nsa_pathloss_dict_3, main_sa_nsa_lte_tech_dl_mimo_dict_3, main_sa_nsa_lte_tech_dl_mimo_layer_dict_3, main_sa_nsa_lte_tech_city_ca_dict_3 = pkl.load(fh)
fh.close()


fh = open("../pkls/data_2023/overal_xput.pkl", "rb")
main_xput_tech_dl_dict_1, main_xput_tech_ul_dict_1 = pkl.load(fh)
fh.close()

fh = open("/home/moinakgh/csv_ho/nsa_sa_analysis_perf/pkls/driving_trip_lax_bos_2024/overal_xput.pkl", "rb")
main_xput_tech_dl_dict_3, main_xput_tech_ul_dict_3 = pkl.load(fh)
fh.close()

fh = open("../pkls/data_2023/xput_break.pkl", "rb")
fiveg_xput_tech_dl_dict_1, lte_xput_tech_dl_dict_1, fiveg_xput_tech_ul_dict_1, lte_xput_tech_ul_dict_1, main_sa_nsa_lte_tech_ca_band_xput_dict_1, main_sa_nsa_lte_tech_ca_ul_band_xput_dict_1 = pkl.load(fh)
fh.close()

fh = open("/home/moinakgh/csv_ho/nsa_sa_analysis_perf/pkls/driving_trip_lax_bos_2024/xput_break.pkl", "rb")
fiveg_xput_tech_dl_dict_3, lte_xput_tech_dl_dict_3, fiveg_xput_tech_ul_dict_3, lte_xput_tech_ul_dict_3, main_sa_nsa_lte_tech_ca_band_xput_dict_3, main_sa_nsa_lte_tech_ca_ul_band_xput_dict_3 = pkl.load(fh)
fh.close()


fh = open("../pkls/data_2023/xput_break_part_2.pkl", "rb")
main_tech_xput_city_info_dict_1, main_tech_xput_tx_power_dict_1, main_tech_xput_tx_power_control_dict_1, main_tech_xput_dl_tx_power_dict_1, main_tech_xput_dl_tx_power_control_dict_1, main_tech_xput_ul_tx_power_dict_1, main_tech_xput_ul_tx_power_control_dict_1, main_tech_xput_rx_power_dict_1, main_tech_xput_dl_rx_power_dict_1, main_tech_xput_ul_rx_power_dict_1, main_tech_xput_pathloss_dict_1, main_tech_xput_dl_pathloss_dict_1, main_tech_xput_ul_pathloss_dict_1, main_tech_xput_dl_bandwidth_dict_1, main_tech_xput_ul_bandwidth_dict_1, main_tech_xput_dl_mean_dict_1, main_tech_xput_dl_std_dict_1, main_tech_xput_ul_mean_dict_1, main_tech_xput_ul_std_dict_1, main_tech_xput_dl_diff_dict_1, main_tech_xput_ul_diff_dict_1, main_tech_xput_dl_ca_bandwidth_dict_1, main_tech_xput_ul_ca_bandwidth_dict_1  = pkl.load(fh)
fh.close()

fh = open("/home/moinakgh/csv_ho/nsa_sa_analysis_perf/pkls/driving_trip_lax_bos_2024/xput_break_part_2.pkl", "rb")
main_tech_xput_city_info_dict_3, main_tech_xput_tx_power_dict_3, main_tech_xput_tx_power_control_dict_3, main_tech_xput_dl_tx_power_dict_3, main_tech_xput_dl_tx_power_control_dict_3, main_tech_xput_ul_tx_power_dict_3, main_tech_xput_ul_tx_power_control_dict_3, main_tech_xput_rx_power_dict_3, main_tech_xput_dl_rx_power_dict_3, main_tech_xput_ul_rx_power_dict_3, main_tech_xput_pathloss_dict_3, main_tech_xput_dl_pathloss_dict_3, main_tech_xput_ul_pathloss_dict_3, main_tech_xput_dl_bandwidth_dict_3, main_tech_xput_ul_bandwidth_dict_3, main_tech_xput_dl_mean_dict_3, main_tech_xput_dl_std_dict_3, main_tech_xput_ul_mean_dict_3, main_tech_xput_ul_std_dict_3, main_tech_xput_dl_diff_dict_3, main_tech_xput_ul_diff_dict_3, main_tech_xput_dl_ca_bandwidth_dict_3, main_tech_xput_ul_ca_bandwidth_dict_3 = pkl.load(fh)
fh.close()

fh = open("../pkls/data_2023/rtt_break.pkl", "rb")
main_xput_tech_ping_dict_1, main_tech_ping_band_dict_1, main_tech_ping_city_info_dict_1, main_tech_ping_rsrp_dict_1, main_tech_ping_pathloss_dict_1, main_ping_tx_power_dict_1, main_ping_tx_power_control_dict_1, main_tech_ping_band_scs_dict_1 = pkl.load(fh)
fh.close()


fh = open("/home/moinakgh/csv_ho/nsa_sa_analysis_perf/pkls/driving_trip_lax_bos_2024/rtt_break.pkl", "rb")
main_xput_tech_ping_dict_3, main_xput_tech_ping_dict_day_wise_3, main_tech_ping_band_dict_3, main_tech_ping_city_info_dict_3, main_tech_ping_rsrp_dict_3, main_tech_ping_pathloss_dict_3, main_ping_tx_power_dict_3, main_ping_tx_power_control_dict_3, n25_ping_location_3, main_tech_ping_band_scs_dict_3 = pkl.load(fh)
fh.close()


fh = open("../pkls/data_2023/ho_duration.pkl", "rb")
sa_duration_list,\
sa_duration_list_intra_freq,\
sa_duration_list_inter_freq,\
sa_duration_list_intra_gnb,\
sa_duration_list_inter_gnb,\
nsa_nr_nr_duration_list,\
nsa_lte_nr_duration_list,\
lte_lte_duration_list,\
nsa_nr_nr_time_diff_after_lte_ho,\
nsa_lte_nr_time_diff_after_lte_ho,\
sa_inter_intra_dict,\
nsa_nr_nr_duration_intra_list,\
nsa_nr_nr_duration_inter_list,\
nsa_inter_intra_dict = pkl.load(fh)
fh.close()

fh = open("/home/moinakgh/csv_ho/nsa_sa_analysis_perf/pkls/driving_trip_lax_bos_2024/ho_duration.pkl", "rb")
sa_duration_list_3,\
sa_duration_list_intra_freq_3,\
sa_duration_list_inter_freq_3,\
sa_duration_list_intra_gnb_3,\
sa_duration_list_inter_gnb_3,\
nsa_nr_nr_duration_list_3,\
nsa_lte_nr_duration_list_3,\
lte_lte_duration_list_3,\
nsa_nr_nr_time_diff_after_lte_ho_3,\
nsa_lte_nr_time_diff_after_lte_ho_3,\
sa_inter_intra_dict_3,\
nsa_nr_nr_duration_intra_list_3,\
nsa_nr_nr_duration_inter_list_3, \
nsa_inter_intra_dict_3 = pkl.load(fh)
fh.close()

import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

def plot_on_map_2():
    global city_tech_df_1, city_tech_df_3
    # Extract unique lat-lon points for each year and convert them to lists
    lat_lon_2023 = city_tech_df_1.drop_duplicates(subset='Lat-Lon')['Lat-Lon'].dropna().tolist()
    lat_lon_2024 = city_tech_df_3.drop_duplicates(subset='Lat-Lon')['Lat-Lon'].dropna().tolist()
    
    # Load U.S. country shapefile and state boundaries, convert both to EPSG:3857 projection
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    us_map = world[(world.name == "United States") & (world.geometry.is_valid)].to_crs("EPSG:3857")
    us_states = gpd.read_file('https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_state_20m.zip')
    us_states = us_states.to_crs("EPSG:3857")
    
    # Convert lat-lon data to GeoDataFrames with Mercator projection
    def convert_to_mercator(lat_lon):
        gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([lon for lat, lon in lat_lon], [lat for lat, lon in lat_lon]))
        gdf.set_crs("EPSG:4326", inplace=True)
        return gdf.to_crs("EPSG:3857")
    
    lat_lon_2023_gdf = convert_to_mercator(lat_lon_2023)
    lat_lon_2024_gdf = convert_to_mercator(lat_lon_2024)
    
    # Define major cities in the region (lat, lon, name)
    major_cities = [
        (41.8781, -87.6298, "Chicago"),      # Chicago, IL
        (42.3601, -71.0589, "Boston"),       # Boston, MA
        (41.4993, -81.6944, "Cleveland"),    # Cleveland, OH
        (39.7684, -86.1581, "Indianapolis"), # Indianapolis, IN
        (43.1566, -77.6088, "Rochester"),    # Rochester, NY
        (39.9612, -82.9988, "Columbus"),     # Columbus, OH
        (42.8864, -78.8784, "Buffalo")       # Buffalo, NY
    ]
    
    # Convert cities to GeoDataFrame
    city_points = [{"geometry": gpd.points_from_xy([lon], [lat])[0], "name": name} for lat, lon, name in major_cities]
    cities_gdf = gpd.GeoDataFrame(city_points)
    cities_gdf.set_crs("EPSG:4326", inplace=True)
    cities_gdf = cities_gdf.to_crs("EPSG:3857")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5))
    us_map.plot(ax=ax, color="none", edgecolor="black", linewidth=0.7)
    us_states.plot(ax=ax, color="none", edgecolor="grey", linewidth=0.5)
    
    # Add points for 2023 and 2024 data
    lat_lon_2023_gdf.plot(ax=ax, color='orangered', markersize=100, marker='o', edgecolor='black')
    # lat_lon_2024_gdf.plot(ax=ax, color='darkslategray', label='2024', markersize=15, marker='^', edgecolor='black', alpha=0.8)
    
    # Add city points and labels
    for idx, row in cities_gdf.iterrows():
        # Plot city point
        x, y = row.geometry.x, row.geometry.y
        ax.scatter(x, y, c='blue', s=30, zorder=5)
        
        # Add city label with white background for readability
        plt.annotate(
            text=row['name'],
            xy=(x, y),
            xytext=(5, 5),  # Offset text slightly from point
            textcoords="offset points",
            fontsize=13,
            fontweight='bold',
            color='black',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7)
        )
    
    # Set plot limits to fit the northeastern/midwestern region
    ax.set_xlim(-10000000, -7500000)  # Extended westward to include Chicago
    ax.set_ylim(4200000, 5500000)
    ax.set_axis_off()
    
    # Add legend
    plt.legend(loc="lower left")
    
    # Add title
    
    plt.tight_layout()
    plt.savefig("/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/route_with_cities_and_states.png", dpi=300)
    plt.savefig("/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/route_with_cities_and_states.pdf")
    plt.close()



if 1:
    # plot_on_map_2()
    if 0:
        # with LTE and others
        for key in main_sa_nsa_lte_tech_dist_dict_1.keys():
            main_sa_nsa_lte_tech_dist_dict_1[key] = sum(main_sa_nsa_lte_tech_dist_dict_1[key])
        main_sa_nsa_lte_tech_dist_dict_1.pop('Others', None) 
        main_sa_nsa_lte_tech_dist_dict_1 = dict(sorted(main_sa_nsa_lte_tech_dist_dict_1.items()))

        for key in main_sa_nsa_lte_tech_dist_dict_3.keys():
            main_sa_nsa_lte_tech_dist_dict_3[key] = sum(main_sa_nsa_lte_tech_dist_dict_3[key])
        main_sa_nsa_lte_tech_dist_dict_3.pop('Others', None) 
        main_sa_nsa_lte_tech_dist_dict_3 = dict(sorted(main_sa_nsa_lte_tech_dist_dict_3.items()))

        # for key in main_sa_nsa_lte_tech_dist_dict_3.keys():
        #     main_sa_nsa_lte_tech_dist_dict_3[key]+=main_sa_nsa_lte_tech_dist_dict_3[key]
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        ax[0].pie(main_sa_nsa_lte_tech_dist_dict_1.values(), colors=['salmon', 'slategrey', 'darkseagreen'], labels=main_sa_nsa_lte_tech_dist_dict_1.keys(), autopct='%1.1f%%', textprops={'fontsize': 14})
        ax[1].pie(main_sa_nsa_lte_tech_dist_dict_3.values(), colors=['salmon', 'slategrey', 'darkseagreen'], labels=main_sa_nsa_lte_tech_dist_dict_3.keys(), autopct='%1.1f%%', textprops={'fontsize': 14})
        ax[0].set_title('2023', fontweight='bold', fontsize=18)
        ax[1].set_title('2024', fontweight='bold', fontsize=18)

        patch_list = []
        for k, v in zip(list(main_sa_nsa_lte_tech_dist_dict_3.keys()), ['salmon', 'slategrey', 'darkseagreen']):
            patch_list.append(mpatches.Patch(color=v, label=k))
        # ax[1].legend(handles=patch_list, loc='center left', bbox_to_anchor=(1.1, 0.6), fontsize=17)
        plt.tight_layout()
        plt.savefig("/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/sa_nsa_coverage_all.pdf")
        plt.close()
        sys.exit(1)
    
    if 1:
        # with LTE and others - First create copies to avoid modifying original data
        data_2023 = main_sa_nsa_lte_tech_dist_dict_1.copy()
        data_2024 = main_sa_nsa_lte_tech_dist_dict_3.copy()

        # Convert to sums if they're still lists
        for key in data_2023.keys():
            if isinstance(data_2023[key], list):
                data_2023[key] = sum(data_2023[key])
        data_2023.pop('Others', None)
        data_2023 = dict(sorted(data_2023.items()))

        for key in data_2024.keys():
            if isinstance(data_2024[key], list):
                data_2024[key] = sum(data_2024[key])
        data_2024.pop('Others', None)
        data_2024 = dict(sorted(data_2024.items()))

        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(4, 4))

        # Prepare data for stacked bars
        years = ['2023', '2024']

        # Calculate totals for percentage conversion
        total_2023 = sum(data_2023.values())
        total_2024 = sum(data_2024.values())

        # Convert to percentages
        data_2023_pct = {k: (v / total_2023) * 100 for k, v in data_2023.items()}
        data_2024_pct = {k: (v / total_2024) * 100 for k, v in data_2024.items()}

        # Get all unique technologies and ensure consistent ordering
        all_techs = sorted(set(list(data_2023.keys()) + list(data_2024.keys())))
        colors = ['salmon', 'slategrey', 'darkseagreen']
        color_map = {tech: colors[i] for i, tech in enumerate(all_techs)}

        # Prepare percentage data for each technology
        tech_data_pct = {}
        for tech in all_techs:
            tech_data_pct[tech] = [
                data_2023_pct.get(tech, 0),  # 2023 percentage
                data_2024_pct.get(tech, 0)   # 2024 percentage
            ]

        # Create stacked bars
        x_pos = range(len(years))
        bottom = [0, 0]  # Starting bottom for each year

        for tech in all_techs:
            values = tech_data_pct[tech]
            bars = ax.bar(x_pos, values, bottom=bottom, 
                        label=tech, color=color_map[tech], 
                        edgecolor='black', linewidth=1, width=0.3)
            
            # Add percentage labels on bars (only if > 5%)
            for i, (bar, value) in enumerate(zip(bars, values)):
                if value > 5:  # Only show label if > 5%
                    height = bar.get_height()
                    y_pos = bottom[i] + height/2
                    # ax.text(bar.get_x() + bar.get_width()/2, y_pos, 
                    #        f'{value:.1f}%', ha='center', va='center',
                    #        fontsize=12, fontweight='bold', color='white')
            
            # Update bottom for next stack
            bottom = [b + v for b, v in zip(bottom, values)]

        # Customize the plot
        ax.set_xlabel('Year', fontsize=14, fontweight='bold')
        ax.set_ylabel('%', fontsize=14, fontweight='bold')

        # Set x-axis
        ax.set_xticks(x_pos)
        ax.set_xticklabels(years, fontsize=14, fontweight='bold')

        # Set y-axis to percentage (0-100%)
        ax.set_ylim(0, 100)

        # Add grid for better readability
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add legend
        ax.legend(loc='upper center', fontsize=11.5)

        # Adjust layout
        plt.tight_layout()

        # Save figure
        plt.savefig("../plots/yearwise/sa_nsa_coverage_all_stacked.pdf")
        plt.close()
        sys.exit(1)


    #overall distance tech distribution 
    for key in main_sa_nsa_lte_tech_dist_dict_1.keys():
        main_sa_nsa_lte_tech_dist_dict_1[key] = sum(main_sa_nsa_lte_tech_dist_dict_1[key])
    main_sa_nsa_lte_tech_dist_dict_1.pop('Others', None) 
    main_sa_nsa_lte_tech_dist_dict_1.pop('LTE', None) 
    main_sa_nsa_lte_tech_dist_dict_1 = dict(sorted(main_sa_nsa_lte_tech_dist_dict_1.items()))

    for key in main_sa_nsa_lte_tech_dist_dict_3.keys():
        main_sa_nsa_lte_tech_dist_dict_3[key] = sum(main_sa_nsa_lte_tech_dist_dict_3[key])
    main_sa_nsa_lte_tech_dist_dict_3.pop('Others', None) 
    main_sa_nsa_lte_tech_dist_dict_3.pop('LTE', None) 
    main_sa_nsa_lte_tech_dist_dict_3 = dict(sorted(main_sa_nsa_lte_tech_dist_dict_3.items()))

    # for key in main_sa_nsa_lte_tech_dist_dict_3.keys():
    #     main_sa_nsa_lte_tech_dist_dict_3[key]+=main_sa_nsa_lte_tech_dist_dict_3[key]
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].pie(main_sa_nsa_lte_tech_dist_dict_1.values(), colors=['salmon', 'slategrey'], labels=main_sa_nsa_lte_tech_dist_dict_1.keys(), autopct='%1.1f%%', textprops={'fontsize': 14})
    ax[1].pie(main_sa_nsa_lte_tech_dist_dict_3.values(), colors=['salmon', 'slategrey'], labels=main_sa_nsa_lte_tech_dist_dict_3.keys(), autopct='%1.1f%%', textprops={'fontsize': 14})
    ax[0].set_title('2023', fontweight='bold', fontsize=18)
    ax[1].set_title('2024', fontweight='bold', fontsize=18)

    patch_list = []
    for k, v in zip(list(main_sa_nsa_lte_tech_dist_dict_3.keys()), ['salmon', 'slategrey']):
        patch_list.append(mpatches.Patch(color=v, label=k))
    ax[1].legend(handles=patch_list, loc='center left', bbox_to_anchor=(1.1, 0.6), fontsize=17)
    plt.tight_layout()
    plt.savefig("/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/sa_nsa_coverage.pdf")
    plt.close()

    for key in main_sa_nsa_lte_tech_dist_dict_chic_bos_3.keys():
        main_sa_nsa_lte_tech_dist_dict_chic_bos_3[key] = sum(main_sa_nsa_lte_tech_dist_dict_chic_bos_3[key])
    main_sa_nsa_lte_tech_dist_dict_chic_bos_3.pop('Others', None) 
    main_sa_nsa_lte_tech_dist_dict_chic_bos_3.pop('LTE', None) 
    main_sa_nsa_lte_tech_dist_dict_chic_bos_3 = dict(sorted(main_sa_nsa_lte_tech_dist_dict_chic_bos_3.items()))

    # for key in main_sa_nsa_lte_tech_dist_dict_chic_bos_3.keys():
    #     main_sa_nsa_lte_tech_dist_dict_chic_bos_3[key]+=main_sa_nsa_lte_tech_dist_dict_chic_bos_3[key]

    for key in main_sa_nsa_lte_tech_dist_dict_trip_2.keys():
        main_sa_nsa_lte_tech_dist_dict_trip_2[key] = sum(main_sa_nsa_lte_tech_dist_dict_trip_2[key])
    main_sa_nsa_lte_tech_dist_dict_trip_2.pop('Others', None) 
    main_sa_nsa_lte_tech_dist_dict_trip_2.pop('LTE', None) 
    main_sa_nsa_lte_tech_dist_dict_trip_2 = dict(sorted(main_sa_nsa_lte_tech_dist_dict_trip_2.items()))

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].pie(main_sa_nsa_lte_tech_dist_dict_trip_2.values(), colors=['salmon', 'slategrey'], labels=main_sa_nsa_lte_tech_dist_dict_trip_2.keys(),         autopct='%1.1f%%', textprops={'fontsize': 16})
    ax[1].pie(main_sa_nsa_lte_tech_dist_dict_chic_bos_3.values(), colors=['salmon', 'slategrey'], labels=main_sa_nsa_lte_tech_dist_dict_chic_bos_3.keys(), autopct='%1.1f%%', textprops={'fontsize': 16})
    ax[0].set_title('2023', fontweight='bold', fontsize=18)
    ax[1].set_title('2024', fontweight='bold', fontsize=18)

    patch_list = []
    for k, v in zip(list(main_sa_nsa_lte_tech_dist_dict_chic_bos_3.keys()), ['salmon', 'slategrey']):
        patch_list.append(mpatches.Patch(color=v, label=k))
    # ax[1].legend(handles=patch_list, loc='center left', bbox_to_anchor=(1.1, 0.6), fontsize=13)
    plt.tight_layout()
    #plt.savefig("/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/sa_nsa_coverage_chic_boston.pdf")
    plt.close()

    fh = open("../pkls/data_2023/coverage_data.pkl", "rb")
    main_lat_lon_tech_df_1, main_sa_nsa_lte_tech_time_dict_1, main_sa_nsa_lte_tech_dist_dict_1, main_sa_nsa_lte_tech_dist_dict_city_1, main_sa_nsa_lte_tech_band_dict_1, main_sa_nsa_lte_tech_ca_dict_1, main_sa_nsa_lte_tech_ca_ul_dict_1, main_sa_nsa_lte_tech_ca_band_combo_dict_1, main_sa_nsa_lte_tech_ca_ul_band_combo_dict_1, main_sa_nsa_tx_power_dict_1, main_sa_nsa_tx_power_control_dict_1, main_sa_nsa_rx_power_dict_1, main_sa_nsa_pathloss_dict_1, main_sa_nsa_lte_tech_dl_mimo_dict_1, main_sa_nsa_lte_tech_dl_mimo_layer_dict_1, main_sa_nsa_lte_tech_city_ca_dict_1 = pkl.load(fh)
    fh.close()

    fh = open("/home/moinakgh/csv_ho/nsa_sa_analysis_perf/pkls/driving_trip_lax_bos_2024/coverage_data.pkl", "rb")
    main_lat_lon_tech_df_3, main_sa_nsa_lte_tech_time_dict_3, main_sa_nsa_lte_tech_dist_dict_3, main_sa_nsa_lte_tech_dist_dict_tz_3, main_sa_nsa_lte_tech_dist_dict_city_3, main_sa_nsa_lte_tech_dist_dict_chic_bos_3, main_sa_nsa_lte_tech_band_dict_3, main_sa_nsa_lte_tech_ca_dict_3, main_sa_nsa_lte_tech_ca_ul_dict_3, main_sa_nsa_lte_tech_ca_band_combo_dict_3, main_sa_nsa_lte_tech_ca_ul_band_combo_dict_3, main_sa_nsa_tx_power_dict_3, main_sa_nsa_tx_power_control_dict_3, main_sa_nsa_rx_power_dict_3, main_sa_nsa_pathloss_dict_3, main_sa_nsa_lte_tech_dl_mimo_dict_3, main_sa_nsa_lte_tech_dl_mimo_layer_dict_3, main_sa_nsa_lte_tech_city_ca_dict_3 = pkl.load(fh)
    fh.close()

    # Count occurrences of each combination
    counts_1 = city_tech_df_1.groupby(['Event Technology', 'city_info']).size().reset_index(name='count')

    # Calculate total counts for each Event Technology
    total_counts = city_tech_df_1['Event Technology'].value_counts().reset_index()
    total_counts.columns = ['Event Technology', 'total_count']

    # Merge counts with total counts
    counts_1 = counts_1.merge(total_counts, on='Event Technology')

    # Calculate percentage
    counts_1['percentage'] = (counts_1['count'] / counts_1['total_count']) * 100

    # Display results
    print(counts_1)

   
    # Count occurrences of each combination
    counts_3 = city_tech_df_3.groupby(['Event Technology', 'city_info']).size().reset_index(name='count')

    # Calculate total counts for each Event Technology
    total_counts = city_tech_df_3['Event Technology'].value_counts().reset_index()
    total_counts.columns = ['Event Technology', 'total_count']

    # Merge counts with total counts
    counts_3 = counts_3.merge(total_counts, on='Event Technology')

    # Calculate percentage
    counts_3['percentage'] = (counts_3['count'] / counts_3['total_count']) * 100

    # Display results
    print(counts_3)

    # plot big city vs non-big city 
    big_city_2023 = {'5G (NSA)' : counts_1.loc[0]['count'], '5G (SA)' : counts_1.loc[3]['count']}
    big_city_2024 = {'5G (NSA)' : counts_3.loc[0]['count'], '5G (SA)' : counts_3.loc[3]['count']}
    non_big_city_2023 = {'5G (NSA)' : counts_1.loc[1]['count'] , '5G (SA)' : counts_1.loc[4]['count']}
    non_big_city_2024 = {'5G (NSA)' : counts_3.loc[1]['count'] , '5G (SA)' : counts_3.loc[4]['count']}


    # Calculate total values for each category
    total_2023_big = big_city_2023['5G (NSA)'] + big_city_2023['5G (SA)']
    total_2024_big = big_city_2024['5G (NSA)'] + big_city_2024['5G (SA)']
    total_2023_non_big = non_big_city_2023['5G (NSA)'] + non_big_city_2023['5G (SA)']
    total_2024_non_big = non_big_city_2024['5G (NSA)'] + non_big_city_2024['5G (SA)']

    # Calculate percentages for stacking
    big_city_2023_perc = [big_city_2023['5G (NSA)'] / total_2023_big * 100, big_city_2023['5G (SA)'] / total_2023_big * 100]
    big_city_2024_perc = [big_city_2024['5G (NSA)'] / total_2024_big * 100, big_city_2024['5G (SA)'] / total_2024_big * 100]
    non_big_city_2023_perc = [non_big_city_2023['5G (NSA)'] / total_2023_non_big * 100, non_big_city_2023['5G (SA)'] / total_2023_non_big * 100]
    non_big_city_2024_perc = [non_big_city_2024['5G (NSA)'] / total_2024_non_big * 100, non_big_city_2024['5G (SA)'] / total_2024_non_big * 100]

    try:
        del(main_sa_nsa_lte_tech_dist_dict_city_1['5G (SA)']['unclassified'])
        del(main_sa_nsa_lte_tech_dist_dict_city_3['5G (SA)']['unclassified'])

        del(main_sa_nsa_lte_tech_dist_dict_city_1['5G (NSA)']['unclassified'])
        del(main_sa_nsa_lte_tech_dist_dict_city_3['5G (NSA)']['unclassified'])
    except:
        pass 

    for tech in main_sa_nsa_lte_tech_dist_dict_city_1.keys():
        for city in main_sa_nsa_lte_tech_dist_dict_city_1[tech].keys():
            main_sa_nsa_lte_tech_dist_dict_city_1[tech][city] = sum(main_sa_nsa_lte_tech_dist_dict_city_1[tech][city])

    for tech in main_sa_nsa_lte_tech_dist_dict_city_3.keys():
        for city in main_sa_nsa_lte_tech_dist_dict_city_3[tech].keys():
            main_sa_nsa_lte_tech_dist_dict_city_3[tech][city] = sum(main_sa_nsa_lte_tech_dist_dict_city_3[tech][city])

    big_city_2023_perc = [main_sa_nsa_lte_tech_dist_dict_city_1['5G (NSA)']['big-city'] / (main_sa_nsa_lte_tech_dist_dict_city_1['5G (NSA)']['big-city'] + main_sa_nsa_lte_tech_dist_dict_city_1['5G (SA)']['big-city']) * 100, main_sa_nsa_lte_tech_dist_dict_city_1['5G (SA)']['big-city'] / (main_sa_nsa_lte_tech_dist_dict_city_1['5G (NSA)']['big-city'] + main_sa_nsa_lte_tech_dist_dict_city_1['5G (SA)']['big-city']) * 100]
    big_city_2024_perc = [main_sa_nsa_lte_tech_dist_dict_city_3['5G (NSA)']['big-city'] / (main_sa_nsa_lte_tech_dist_dict_city_3['5G (NSA)']['big-city'] + main_sa_nsa_lte_tech_dist_dict_city_3['5G (SA)']['big-city']) * 100, main_sa_nsa_lte_tech_dist_dict_city_3['5G (SA)']['big-city'] / (main_sa_nsa_lte_tech_dist_dict_city_3['5G (NSA)']['big-city'] + main_sa_nsa_lte_tech_dist_dict_city_3['5G (SA)']['big-city']) * 100]
    non_big_city_2023_perc = [main_sa_nsa_lte_tech_dist_dict_city_1['5G (NSA)']['not-big-city'] / (main_sa_nsa_lte_tech_dist_dict_city_1['5G (NSA)']['not-big-city'] + main_sa_nsa_lte_tech_dist_dict_city_1['5G (SA)']['not-big-city']) * 100, main_sa_nsa_lte_tech_dist_dict_city_1['5G (SA)']['not-big-city'] / (main_sa_nsa_lte_tech_dist_dict_city_1['5G (NSA)']['not-big-city'] + main_sa_nsa_lte_tech_dist_dict_city_1['5G (SA)']['not-big-city']) * 100]
    non_big_city_2024_perc = [main_sa_nsa_lte_tech_dist_dict_city_3['5G (NSA)']['not-big-city'] / (main_sa_nsa_lte_tech_dist_dict_city_3['5G (NSA)']['not-big-city'] + main_sa_nsa_lte_tech_dist_dict_city_3['5G (SA)']['not-big-city']) * 100, main_sa_nsa_lte_tech_dist_dict_city_3['5G (SA)']['not-big-city'] / (main_sa_nsa_lte_tech_dist_dict_city_3['5G (NSA)']['not-big-city'] + main_sa_nsa_lte_tech_dist_dict_city_3['5G (SA)']['not-big-city']) * 100]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(5, 4))

    # Define bar positions and width
    bar_width = 0.1
    x = np.array([0, 0.3])  # Position for 2023 and 2024

    # Plot stacked bars for each group
    # 2023 data
    ax.bar(x[0] - bar_width / 2, big_city_2023_perc[0], width=bar_width, label='5G (NSA) - Big City', color='salmon', hatch='', edgecolor='black')
    ax.bar(x[0] - bar_width / 2, big_city_2023_perc[1], width=bar_width, bottom=big_city_2023_perc[0], color='slategrey', hatch='', edgecolor='black')

    ax.bar(x[0] + bar_width / 2, non_big_city_2023_perc[0], width=bar_width, label='5G (NSA) - Non-Big City', color='salmon', hatch='\\\\\\', edgecolor='black')
    ax.bar(x[0] + bar_width / 2, non_big_city_2023_perc[1], width=bar_width, bottom=non_big_city_2023_perc[0], color='slategrey', hatch='\\\\\\', edgecolor='black')

    # 2024 data
    ax.bar(x[1] - bar_width / 2, big_city_2024_perc[0], width=bar_width, color='salmon', hatch='', edgecolor='black')
    ax.bar(x[1] - bar_width / 2, big_city_2024_perc[1], width=bar_width, bottom=big_city_2024_perc[0], color='slategrey', hatch='', edgecolor='black')

    ax.bar(x[1] + bar_width / 2, non_big_city_2024_perc[0], width=bar_width, color='salmon', hatch='\\\\\\', edgecolor='black')
    ax.bar(x[1] + bar_width / 2, non_big_city_2024_perc[1], width=bar_width, bottom=non_big_city_2024_perc[0], color='slategrey', hatch='\\\\\\', edgecolor='black')

    # Labels and titles
    ax.set_xticks(x)
    ax.set_xticklabels(['2023', '2024'])
    ax.set_ylabel('Fraction of miles\n covered (%)')
    ax.set_xlabel('Year')
    ax.set_ylim(0, 160)
    ax.legend(['5G (NSA) - Big City', '5G (SA) - Big City', '5G (NSA) - Non-Big City', '5G (SA) - Non-Big City'], loc='upper center', fontsize=12)
    plt.tight_layout()
    plt.savefig("/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/sa_nsa_city_breakdown.pdf")
    plt.close()

    merged_data = pd.concat([city_tech_df_1, city_tech_df_3])
    grouped = merged_data.groupby(['Timezone', 'Event Technology'])
    group_summary = grouped.size().reset_index(name='count')

    tz_tech_dist = {'America/Los_Angeles' : {}, 'America/Denver' : {}, 'America/Chicago' : {}, 'America/New_York' : {}}

    # Iterate over the rows of group_summary, which is now a DataFrame
    for _, row in group_summary.iterrows():
        # Skip rows where 'Event Technology' is 'LTE'
        if row['Event Technology'] == 'LTE':
            continue
        # Update the dictionary with the count
        tz_tech_dist[row['Timezone']][row['Event Technology']] = row['count']

    data = tz_tech_dist.copy()

    # xput cdf
    if 0:
        # overall xput 
        color_dict = {'NSA' : 'salmon', 'SA' : 'slategrey', 'LTE' : 'green', 'LTE (DC)' : 'lime'}
        fig, ax = plt.subplots(figsize=(6, 4), sharey=True)
        for tech in main_xput_tech_dl_dict_1.keys():
            if tech == 'Skip' or 'gsm' in tech.lower() or 'service' in tech.lower() or ('lte' in tech.lower() and 'DC' not in tech):
                continue
            data = main_xput_tech_dl_dict_1[tech].copy()
            sorted_data = np.sort(data)
            ax.plot(sorted_data, np.linspace(0, 1, sorted_data.size), label=tech + " (2023)", color=color_dict[tech], lw=3)

            data = main_xput_tech_dl_dict_3[tech].copy()
            sorted_data = np.sort(data)
            ax.plot(sorted_data, np.linspace(0, 1, sorted_data.size), label=tech + " (2024)", color=color_dict[tech], ls='--', lw=3)

        ax.set_ylabel('CDF')
        ax.set_xlabel('Throughput (Mbps)')
        ax.legend(loc='best')
        ax.set_ylim(0, 1)
        ax.set_xlim(xmin=0, xmax=1200)
        plt.tight_layout()
        plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/dl_sa_nsa_xput_overall.pdf')
        plt.close()


        fig, ax = plt.subplots(figsize=(6, 4))
        for tech in main_xput_tech_ul_dict_1.keys():
            if tech == 'Skip' or 'gsm' in tech.lower() or 'service' in tech.lower() or ('lte' in tech.lower() and 'DC' not in tech):
                continue
            data = main_xput_tech_ul_dict_1[tech].copy()
            sorted_data = np.sort(data)
            ax.plot(sorted_data, np.linspace(0, 1, sorted_data.size), label=tech + " (2023)", color=color_dict[tech], lw=3)

            data = main_xput_tech_ul_dict_3[tech].copy()
            sorted_data = np.sort(data)
            ax.plot(sorted_data, np.linspace(0, 1, sorted_data.size), label=tech + " (2024)", color=color_dict[tech], ls='--', lw=3)

        ax.set_ylabel('CDF')
        ax.set_xlabel('Throughput (Mbps)')
        ax.legend(loc='best')
        ax.set_ylim(0, 1)
        ax.set_xlim(xmin=0, xmax=120)
        plt.tight_layout()
        plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ul_sa_nsa_xput_overall.pdf')
        plt.close()

    # 2023 dl ul bar
    if 0:
        dl_avg_xput_dict_1 = {}
        dl_std_xput_dict_1 = {}
        ul_avg_xput_dict_1 = {}
        ul_std_xput_dict_1 = {}

        fig, ax = plt.subplots()
        for tech in fiveg_xput_tech_dl_dict_1.keys():
            if tech == 'Skip' or 'gsm' in tech.lower() or 'service' in tech.lower():
                continue
            data = fiveg_xput_tech_dl_dict_1[tech].copy()
            sorted_data = np.sort(data)
            dl_avg_xput_dict_1[tech] = np.mean(sorted_data)
            dl_std_xput_dict_1[tech] = np.std(sorted_data)
            ax.plot(sorted_data, np.linspace(0, 1, sorted_data.size), label=tech, color=color_dict[tech])

        for tech in lte_xput_tech_dl_dict_1.keys():
            if tech == 'Skip' or 'gsm' in tech.lower() or tech == 'SA' or 'service' in tech.lower() or 'lte' in tech.lower():
                continue
            data = lte_xput_tech_dl_dict_1[tech].copy()
            sorted_data = np.sort(data)
            if tech == 'NSA':
                tech = 'LTE (DC)'
            dl_avg_xput_dict_1[tech] = np.mean(sorted_data)
            dl_std_xput_dict_1[tech] = np.std(sorted_data)
            ax.plot(sorted_data, np.linspace(0, 1, sorted_data.size), label=tech, color=color_dict[tech])

        ax.set_ylabel('CDF')
        ax.set_xlabel('DL Throughput (Mbps)')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.set_xlim(xmin=0, xmax=1600)
        plt.tight_layout()
        #plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/dl_sa_nsa_xput_break_2023.pdf')
        plt.close()


        fig, ax = plt.subplots()
        for tech in fiveg_xput_tech_ul_dict_1.keys():
            if tech == 'Skip' or 'gsm' in tech.lower() or 'service' in tech.lower():
                continue
            data = fiveg_xput_tech_ul_dict_1[tech].copy()
            sorted_data = np.sort(data)
            ul_avg_xput_dict_1[tech] = np.mean(sorted_data)
            ul_std_xput_dict_1[tech] = np.std(sorted_data)
            ax.plot(sorted_data, np.linspace(0, 1, sorted_data.size), label=tech, color=color_dict[tech])

        for tech in lte_xput_tech_ul_dict_1.keys():
            if tech == 'Skip' or 'gsm' in tech.lower() or tech == 'SA' or 'service' in tech.lower() or 'lte' in tech.lower():
                continue
            data = lte_xput_tech_ul_dict_1[tech].copy()
            sorted_data = np.sort(data)

            if tech == 'NSA':
                tech = 'LTE (DC)'
            ul_avg_xput_dict_1[tech] = np.mean(sorted_data)
            ul_std_xput_dict_1[tech] = np.std(sorted_data)
            ax.plot(sorted_data, np.linspace(0, 1, sorted_data.size), label=tech, color=color_dict[tech])

        ax.set_ylabel('CDF')
        ax.set_xlabel('UL Throughput (Mbps)')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.set_xlim(xmin=0, xmax=160)
        plt.tight_layout()
        #plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ul_sa_nsa_xput_break_2023.pdf')
        plt.close()


        # Extract values for plotting
        x_labels = ['NSA', 'SA']
        nsa_val = dl_avg_xput_dict_1['NSA']
        sa_val = dl_avg_xput_dict_1['SA']
        lte_dc_val = dl_avg_xput_dict_1['LTE (DC)']

        nsa_val_std = dl_std_xput_dict_1['NSA']
        sa_val_std = dl_std_xput_dict_1['SA']
        lte_dc_val_std = dl_std_xput_dict_1['LTE (DC)']

        # Values for the bar heights
        nsa_vals = [nsa_val]  # NSA values
        lte_dc_vals = [lte_dc_val]  # LTE (DC) to stack on NSA
        sa_vals = [sa_val]  # SA values

        # Define the figure and axis
        fig, ax = plt.subplots(figsize=(4, 5))

        # X positions for the bars
        x = range(len(x_labels))

        # Plotting
        ax.bar(x[0], nsa_vals, label='NSA (5G)', color='brown', width=0.35, edgecolor='black')  # NSA Bar
        ax.bar(x[0], lte_dc_vals, bottom=nsa_vals, label='NSA (LTE)', color='rosybrown', width=0.35, edgecolor='black')  # LTE (DC) stacked on NSA
        ax.bar(x[1], sa_vals, label='SA (5G)', color='slategrey', width=0.35, edgecolor='black')  # SA Bar

        # Adding labels and title
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel('Avg. DL Throughput (Mbps)')
        ax.set_xlabel('Technology')
        ax.legend()
        ax.set_ylim(ymax=600)

        # Show the plot
        plt.tight_layout()
        plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/dl_sa_nsa_bar_2023.pdf')
        plt.close()


        # Extract values for plotting
        x_labels = ['NSA', 'SA']
        nsa_val = ul_avg_xput_dict_1['NSA']
        sa_val = ul_avg_xput_dict_1['SA']
        lte_dc_val = ul_avg_xput_dict_1['LTE (DC)']

        nsa_val_std = ul_std_xput_dict_1['NSA']
        sa_val_std = ul_std_xput_dict_1['SA']
        lte_dc_val_std = ul_std_xput_dict_1['LTE (DC)']

        # Values for the bar heights
        nsa_vals = [nsa_val]  # NSA values
        lte_dc_vals = [lte_dc_val]  # LTE (DC) to stack on NSA
        sa_vals = [sa_val]  # SA values

        # Define the figure and axis
        fig, ax = plt.subplots(figsize=(4, 5))

        # X positions for the bars
        x = range(len(x_labels))

        # Plotting
        ax.bar(x[0], nsa_vals, label='NSA (5G)', color='brown', width=0.35, edgecolor='black')  # NSA Bar
        ax.bar(x[0], lte_dc_vals, bottom=nsa_vals, label='NSA (LTE)', color='rosybrown', width=0.35, edgecolor='black')  # LTE (DC) stacked on NSA
        ax.bar(x[1], sa_vals, label='SA (5G)', color='slategrey', width=0.35, edgecolor='black')  # SA Bar

        # Adding labels and title
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel('Avg. UL Throughput (Mbps)')
        ax.set_xlabel('Technology')
        ax.legend()
        ax.set_ylim(ymax=60)

        # Show the plot
        plt.tight_layout()
        plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ul_sa_nsa_bar_2023.pdf')
        plt.close()


    # 2024 dl ul bar
    if 0:
        dl_avg_xput_dict_1 = {}
        dl_std_xput_dict_1 = {}
        ul_avg_xput_dict_1 = {}
        ul_std_xput_dict_1 = {}

        fig, ax = plt.subplots()
        for tech in fiveg_xput_tech_dl_dict_3.keys():
            if tech == 'Skip' or 'gsm' in tech.lower() or 'service' in tech.lower():
                continue
            data = fiveg_xput_tech_dl_dict_3[tech].copy()
            sorted_data = np.sort(data)
            dl_avg_xput_dict_1[tech] = np.mean(sorted_data)
            dl_std_xput_dict_1[tech] = np.std(sorted_data)
            ax.plot(sorted_data, np.linspace(0, 1, sorted_data.size), label=tech, color=color_dict[tech])

        for tech in lte_xput_tech_dl_dict_3.keys():
            if tech == 'Skip' or 'gsm' in tech.lower() or tech == 'SA' or 'service' in tech.lower() or 'lte' in tech.lower():
                continue
            data = lte_xput_tech_dl_dict_3[tech].copy()
            sorted_data = np.sort(data)
            if tech == 'NSA':
                tech = 'LTE (DC)'
            dl_avg_xput_dict_1[tech] = np.mean(sorted_data)
            dl_std_xput_dict_1[tech] = np.std(sorted_data)
            ax.plot(sorted_data, np.linspace(0, 1, sorted_data.size), label=tech, color=color_dict[tech])

        ax.set_ylabel('CDF')
        ax.set_xlabel('DL Throughput (Mbps)')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.set_xlim(xmin=0, xmax=1600)
        plt.tight_layout()
        #plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/dl_sa_nsa_xput_break_2024.pdf')
        plt.close()


        fig, ax = plt.subplots()
        for tech in fiveg_xput_tech_ul_dict_3.keys():
            if tech == 'Skip' or 'gsm' in tech.lower() or 'service' in tech.lower():
                continue
            data = fiveg_xput_tech_ul_dict_3[tech].copy()
            sorted_data = np.sort(data)
            ul_avg_xput_dict_1[tech] = np.mean(sorted_data)
            ul_std_xput_dict_1[tech] = np.std(sorted_data)
            ax.plot(sorted_data, np.linspace(0, 1, sorted_data.size), label=tech, color=color_dict[tech])

        for tech in lte_xput_tech_ul_dict_3.keys():
            if tech == 'Skip' or 'gsm' in tech.lower() or tech == 'SA' or 'service' in tech.lower() or 'lte' in tech.lower():
                continue
            data = lte_xput_tech_ul_dict_3[tech].copy()
            sorted_data = np.sort(data)

            if tech == 'NSA':
                tech = 'LTE (DC)'
            ul_avg_xput_dict_1[tech] = np.mean(sorted_data)
            ul_std_xput_dict_1[tech] = np.std(sorted_data)
            ax.plot(sorted_data, np.linspace(0, 1, sorted_data.size), label=tech, color=color_dict[tech])

        ax.set_ylabel('CDF')
        ax.set_xlabel('UL Throughput (Mbps)')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.set_xlim(xmin=0, xmax=160)
        plt.tight_layout()
        #plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ul_sa_nsa_xput_break_2024.pdf')
        plt.close()


        # Extract values for plotting
        x_labels = ['NSA', 'SA']
        nsa_val = dl_avg_xput_dict_1['NSA']
        sa_val = dl_avg_xput_dict_1['SA']
        lte_dc_val = dl_avg_xput_dict_1['LTE (DC)']

        nsa_val_std = dl_std_xput_dict_1['NSA']
        sa_val_std = dl_std_xput_dict_1['SA']
        lte_dc_val_std = dl_std_xput_dict_1['LTE (DC)']

        # Values for the bar heights
        nsa_vals = [nsa_val]  # NSA values
        lte_dc_vals = [lte_dc_val]  # LTE (DC) to stack on NSA
        sa_vals = [sa_val]  # SA values

        # Define the figure and axis
        fig, ax = plt.subplots(figsize=(4, 5))

        # X positions for the bars
        x = range(len(x_labels))

        # Plotting
        ax.bar(x[0], nsa_vals, label='NSA (5G)', color='brown', width=0.35, edgecolor='black')  # NSA Bar
        ax.bar(x[0], lte_dc_vals, bottom=nsa_vals, label='NSA (LTE)', color='rosybrown', width=0.35, edgecolor='black')  # LTE (DC) stacked on NSA
        ax.bar(x[1], sa_vals, label='SA (5G)', color='slategrey', width=0.35, edgecolor='black')  # SA Bar

        # Adding labels and title
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel('Avg. DL Throughput (Mbps)')
        ax.set_xlabel('Technology')
        ax.legend()
        ax.set_ylim(ymax=600)

        # Show the plot
        plt.tight_layout()
        plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/dl_sa_nsa_bar_2024.pdf')
        plt.close()


        # Extract values for plotting
        x_labels = ['NSA', 'SA']
        nsa_val = ul_avg_xput_dict_1['NSA']
        sa_val = ul_avg_xput_dict_1['SA']
        lte_dc_val = ul_avg_xput_dict_1['LTE (DC)']

        nsa_val_std = ul_std_xput_dict_1['NSA']
        sa_val_std = ul_std_xput_dict_1['SA']
        lte_dc_val_std = ul_std_xput_dict_1['LTE (DC)']

        # Values for the bar heights
        nsa_vals = [nsa_val]  # NSA values
        lte_dc_vals = [lte_dc_val]  # LTE (DC) to stack on NSA
        sa_vals = [sa_val]  # SA values

        # Define the figure and axis
        fig, ax = plt.subplots(figsize=(4, 5))

        # X positions for the bars
        x = range(len(x_labels))

        # Plotting
        ax.bar(x[0], nsa_vals, label='NSA (5G)', color='brown', width=0.35, edgecolor='black')  # NSA Bar
        ax.bar(x[0], lte_dc_vals, bottom=nsa_vals, label='NSA (LTE)', color='rosybrown', width=0.35, edgecolor='black')  # LTE (DC) stacked on NSA
        ax.bar(x[1], sa_vals, label='SA (5G)', color='slategrey', width=0.35, edgecolor='black')  # SA Bar

        # Adding labels and title
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel('Avg. UL Throughput (Mbps)')
        ax.set_xlabel('Technology')
        ax.legend()
        ax.set_ylim(ymax=60)

        # Show the plot
        plt.tight_layout()
        plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ul_sa_nsa_bar_2024.pdf')
        plt.close()

    # carrier aggregation  - 2024
    if 1:
        # Process the data to combine bands with < 5% as "Others"
        def process_bands(category_data):
            processed_data = {}
            for ca, bands in category_data.items():
                processed_data[ca] = {}
                others_total = 0
                
                # First pass: identify bands < 5%
                for band, percentage in bands.items():
                    if percentage < 5:
                        others_total += percentage
                    else:
                        processed_data[ca][band] = percentage
                
                # Add "Others" category if needed
                if others_total > 0:
                    processed_data[ca]["Others"] = others_total
                    
            return processed_data

        # Create a mapping of all unique bands to colors
        def get_band_color_mapping(data):
            all_bands = set()
            for category in data.values():
                for ca_data in category.values():
                    all_bands.update(ca_data.keys())
            
            # Remove "Others" from the set for special handling
            if "Others" in all_bands:
                all_bands.remove("Others")
            
            # Sort bands for consistent ordering
            all_bands = sorted(list(all_bands))
            
            # Create color map using a color spectrum (excluding gray for "Others")
            import matplotlib.cm as cm
            cmap = cm.viridis
            colors = cmap(np.linspace(0, 1, len(all_bands)))
            
            # Create the mapping
            color_map = {band: colors[i] for i, band in enumerate(all_bands)}
            # Add "Others" with gray color
            color_map["Others"] = (0.7, 0.7, 0.7, 1.0)  # Gray for "Others"
            
            return color_map


        if 1:
            # calculate overall ca distribution 
            dl_ca_overall_distribution = {}
            dl_ca_count = {}
            for tech in main_sa_nsa_lte_tech_ca_band_xput_dict_3.keys():
                dl_ca_overall_distribution[tech] = {}
                dl_ca_count[tech] = {}
                temp = {}
                for ca in main_sa_nsa_lte_tech_ca_band_xput_dict_3[tech].keys():
                    if ca == 0:
                        continue 
                    temp[ca] = 0
                    dl_ca_count[tech][ca] = []
                    for band_combo in main_sa_nsa_lte_tech_ca_band_xput_dict_3[tech][ca].keys():
                        if band_combo.count(':') != ca or 'nan' in band_combo:
                            continue 
                        temp[ca] += len(main_sa_nsa_lte_tech_ca_band_xput_dict_3[tech][ca][band_combo])
                        dl_ca_count[tech][ca].extend([band_combo[:-1]] * len(main_sa_nsa_lte_tech_ca_band_xput_dict_3[tech][ca][band_combo]))
                    
                    dl_ca_count[tech][ca] = calculate_percentage_of_occurrence(dl_ca_count[tech][ca])
                
                temp = {k: (v / sum(temp.values())) * 100 for k, v in temp.items()}
                dl_ca_overall_distribution[tech] = temp.copy()

            ul_ca_overall_distribution = {}
            ul_ca_count = {}
            for tech in main_sa_nsa_lte_tech_ca_ul_band_xput_dict_3.keys():
                ul_ca_overall_distribution[tech] = {}
                ul_ca_count[tech] = {}
                temp = {}
                for ca in main_sa_nsa_lte_tech_ca_ul_band_xput_dict_3[tech].keys():
                    if ca == 0:
                        continue 
                    temp[ca] = 0
                    ul_ca_count[tech][ca] = []
                    for band_combo in main_sa_nsa_lte_tech_ca_ul_band_xput_dict_3[tech][ca].keys():
                        if band_combo.count(':') != ca or 'nan' in band_combo:
                            continue 
                        temp[ca] += len(main_sa_nsa_lte_tech_ca_ul_band_xput_dict_3[tech][ca][band_combo])
                        ul_ca_count[tech][ca].extend([band_combo[:-1]] * len(main_sa_nsa_lte_tech_ca_ul_band_xput_dict_3[tech][ca][band_combo]))

                    ul_ca_count[tech][ca] = calculate_percentage_of_occurrence(ul_ca_count[tech][ca])

                temp = {k: (v / sum(temp.values())) * 100 for k, v in temp.items()}
                ul_ca_overall_distribution[tech] = temp.copy()
            
            # overall CA distribution 
            if 1:
                if 0:
                    # Set up the figure and axis
                    fig, ax = plt.subplots(figsize=(5, 4))
                    # Set width of bars
                    bar_width = 0.2
                    # Positions for the bars
                    positions = np.array([0, 1])

                    # Create a gradient color spectrum (blue to red)
                    import matplotlib.cm as cm
                    # Get all unique CA values to determine color mapping
                    all_ca_values = set()
                    for distribution in [dl_ca_overall_distribution, ul_ca_overall_distribution]:
                        for category in distribution.values():
                            all_ca_values.update(category.keys())
                    num_colors = len(all_ca_values)
                    # Create a color map from a blue-to-red spectrum
                    colors = cm.coolwarm(np.linspace(0, 1, num_colors))

                    # Hatches for NSA and SA
                    hatches = ['', '\\\\\\']

                    # Process and plot DL data
                    for i, category in enumerate(['NSA', 'SA']):
                        bottom = 0
                        ca_values = dl_ca_overall_distribution[category]
                        # Calculate total for percentage
                        total = sum(ca_values.values())
                        
                        for j, (ca, value) in enumerate(sorted(ca_values.items())):
                            # Convert to percentage
                            percentage = (value / total) * 100
                            
                            bar = ax.bar(positions[0] + (i - 0.5) * bar_width, percentage, bar_width,
                                        bottom=bottom, color=colors[sorted(list(all_ca_values)).index(ca)],
                                        edgecolor='black', hatch=hatches[i])
                            bottom += percentage
                            
                            # Add value label in the middle of each segment
                            # if percentage > 0:
                            #     height = percentage / 2 + bottom - percentage
                            #     ax.text(positions[0] + (i - 0.5) * bar_width, height,
                            #             f'CA {ca}\n{percentage:.1f}%', ha='center', va='center', fontsize=9)

                    # Process and plot UL data
                    for i, category in enumerate(['NSA', 'SA']):
                        bottom = 0
                        ca_values = ul_ca_overall_distribution[category]
                        # Calculate total for percentage
                        total = sum(ca_values.values())
                        
                        for j, (ca, value) in enumerate(sorted(ca_values.items())):
                            # Convert to percentage
                            percentage = (value / total) * 100
                            
                            bar = ax.bar(positions[1] + (i - 0.5) * bar_width, percentage, bar_width,
                                        bottom=bottom, color=colors[sorted(list(all_ca_values)).index(ca)],
                                        edgecolor='black', hatch=hatches[i])
                            bottom += percentage
                            
                            # Add value label in the middle of each segment
                            # if percentage > 0:
                            #     height = percentage / 2 + bottom - percentage
                            #     ax.text(positions[1] + (i - 0.5) * bar_width, height,
                            #             f'CA {ca}\n{percentage:.1f}%', ha='center', va='center', fontsize=9)

                    # Add legend for CA values
                    ca_values = set()
                    for distribution in [dl_ca_overall_distribution, ul_ca_overall_distribution]:
                        for category in distribution.values():
                            ca_values.update(category.keys())

                    # Create custom legend for CA values
                    ca_legend_elements = [plt.Rectangle((0, 0), 1, 1, 
                                                    facecolor=colors[sorted(list(all_ca_values)).index(ca)],
                                                    edgecolor='black', label=f'CA {ca}')
                                        for ca in sorted(ca_values)]

                    # Create custom legend for NSA and SA (with hatches)
                    hatch_legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor='white',
                                                        edgecolor='black', hatch=hatches[i],
                                                        label=category)
                                            for i, category in enumerate(['NSA', 'SA'])]

                    # Add both legends
                    # ax.legend(handles=ca_legend_elements + hatch_legend_elements,
                    #         loc='center', bbox_to_anchor=(1, 1))

                    ax.legend(handles=ca_legend_elements + hatch_legend_elements,
                            loc='best')
                    # Set labels and title
                    ax.set_xticks(positions)
                    ax.set_xticklabels(['DL', 'UL'])
                    ax.set_ylabel('%')  # Changed to % as requested
                    # Title removed as requested

                    # Add a grid for better readability
                    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

                    # Adjust layout
                    plt.tight_layout()

                    # Show the plot
                    plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/sa_nsa_ca_overall_distribution.pdf')
                    plt.close()

                if 1:
                    # Set up the figure and axis
                    fig, ax = plt.subplots(figsize=(5, 4))

                    # Set width of bars
                    bar_width = 0.2

                    # Positions for the bars
                    positions = np.array([0, 1])

                    # Colors for NSA and SA
                    colors = {'NSA': 'salmon', 'SA': 'slategrey'}

                    # Get all unique CA values to determine hatch mapping
                    all_ca_values = set()
                    for distribution in [dl_ca_overall_distribution, ul_ca_overall_distribution]:
                        for category in distribution.values():
                            all_ca_values.update(category.keys())

                    # Create hatch patterns for different CA values
                    ca_hatches = ['', '///', '\\\\\\', '|||', '...', '+++', 'xxx', 'ooo']
                    ca_hatch_map = {ca: ca_hatches[i % len(ca_hatches)] for i, ca in enumerate(sorted(all_ca_values))}

                    # Process and plot DL data
                    for i, category in enumerate(['NSA', 'SA']):
                        bottom = 0
                        ca_values = dl_ca_overall_distribution[category]
                        # Calculate total for percentage
                        total = sum(ca_values.values())
                        
                        for j, (ca, value) in enumerate(sorted(ca_values.items())):
                            # Convert to percentage
                            percentage = (value / total) * 100
                            bar = ax.bar(positions[0] + (i - 0.5) * bar_width, percentage, bar_width,
                                        bottom=bottom, color=colors[category],
                                        edgecolor='black', hatch=ca_hatch_map[ca])
                            bottom += percentage

                    # Process and plot UL data
                    for i, category in enumerate(['NSA', 'SA']):
                        bottom = 0
                        ca_values = ul_ca_overall_distribution[category]
                        # Calculate total for percentage
                        total = sum(ca_values.values())
                        
                        for j, (ca, value) in enumerate(sorted(ca_values.items())):
                            # Convert to percentage
                            percentage = (value / total) * 100
                            bar = ax.bar(positions[1] + (i - 0.5) * bar_width, percentage, bar_width,
                                        bottom=bottom, color=colors[category],
                                        edgecolor='black', hatch=ca_hatch_map[ca])
                            bottom += percentage

                    # Get all CA values that actually appear in the data
                    ca_values = set()
                    for distribution in [dl_ca_overall_distribution, ul_ca_overall_distribution]:
                        for category in distribution.values():
                            ca_values.update(category.keys())

                    # Create custom legend for CA values (with hatches)
                    ca_legend_elements = [plt.Rectangle((0, 0), 1, 1,
                                                    facecolor='white',
                                                    edgecolor='black', 
                                                    hatch=ca_hatch_map[ca],
                                                    label=f'CA {ca}')
                                        for ca in sorted(ca_values)]

                    # Create custom legend for NSA and SA (with colors)
                    color_legend_elements = [plt.Rectangle((0, 0), 1, 1, 
                                                        facecolor=colors[category],
                                                        edgecolor='black', 
                                                        label=category)
                                            for category in ['NSA', 'SA']]

                    # Add both legends
                    ax.legend(handles=ca_legend_elements + color_legend_elements, loc='best')

                    # Set labels and title
                    ax.set_xticks(positions)
                    ax.set_xticklabels(['DL', 'UL'])
                    ax.set_ylabel('%')

                    # Add a grid for better readability
                    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

                    # Adjust layout
                    plt.tight_layout()

                    # Show the plot
                    plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/sa_nsa_ca_overall_distribution_mod.pdf')
                    plt.close()
                    a = 1

            # plot band distribution - DL
            if 0:
                # Process both NSA and SA data
                processed_dl_ca_count = {
                    'NSA': process_bands(dl_ca_count['NSA']),
                    'SA': process_bands(dl_ca_count['SA'])
                }

                # Get color mapping
                band_colors = get_band_color_mapping(processed_dl_ca_count)

                # Find all unique CA values
                all_cas = set()
                for category in processed_dl_ca_count.values():
                    all_cas.update(category.keys())
                max_ca = max(all_cas) if all_cas else 0

                # Create figure with 2 rows (NSA, SA) and max_ca columns
                fig, axes = plt.subplots(2, max_ca, figsize=(5*max_ca, 8))
                fig.patch.set_edgecolor('black')
                fig.patch.set_linewidth(2)
                # If max_ca is 1, axes will not be a 2D array, so convert to 2D
                if max_ca == 1:
                    axes = np.array(axes).reshape(2, 1)

                # Categories to plot
                categories = ['NSA', 'SA']

                # Plot pie charts
                for row, category in enumerate(categories):
                    for col in range(max_ca):
                        ca = col + 1  # CA values start from 1
                        
                        # If this CA exists for this category
                        if ca in processed_dl_ca_count[category]:
                            ax = axes[row, col]
                            
                            # Get band data for this CA
                            bands_data = processed_dl_ca_count[category][ca]
                            labels = list(bands_data.keys())
                            sizes = list(bands_data.values())
                            
                            # Create label text with percentages
                            # label_texts = [f"{label}\n({size:.1f}%)" for label, size in zip(labels, sizes)]
                            label_texts = [label for label, size in zip(labels, sizes)]
                            
                            # Get colors for each band
                            colors = [band_colors[band] for band in labels]
                            
                            # Use matplotlib's built-in pie chart with external labels
                            wedges, texts = ax.pie(
                                sizes,
                                colors=colors,
                                labels=label_texts,
                                labeldistance=1.1,  # Position labels just outside the pie
                                wedgeprops={'edgecolor': 'w', 'linewidth': 1},
                                textprops={'fontsize': 16, 'ha': 'center'},  # Increased font size from 8 to 10
                                startangle=90,
                                radius=0.8  # Make pie slightly smaller to leave room for labels
                            )
                            
                            # Draw connecting lines from pie to labels
                            for i, wedge in enumerate(wedges):
                                if sizes[i] < 3:  # Skip very small wedges to reduce clutter
                                    texts[i].set_visible(False)
                                    continue
                                
                                # Adjust text alignment for better readability
                                angle = (wedge.theta1 + wedge.theta2) / 2
                                if 90 < angle < 270:  # Left side of the pie
                                    texts[i].set_ha('right')
                                else:  # Right side of the pie
                                    texts[i].set_ha('left')
                            
                            # Modified title format: "CA # (NSA/SA)" and made larger and bold
                            ax.set_title(f"{ca} CA ({category})", fontsize=16, fontweight='bold')
                        else:
                            # If this CA doesn't exist for this category, hide the axes
                            axes[row, col].axis('off')

                # Adjust layout with extra padding for labels
                plt.tight_layout(pad=2.0)

                # Save figure
                plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ca_band_dl_distribution.pdf', 
                            bbox_inches='tight')
                plt.close()
                a = 1

            # plot band distribution - DL - stacked
            if 0:

                # Color mapping function
                def get_band_color_mapping(data):
                    import matplotlib.cm as cm
                    
                    # Get all unique bands
                    all_bands = set()
                    for category in data.values():
                        for ca_data in category.values():
                            all_bands.update(ca_data.keys())
                    
                    # Remove "Others" from the set for special handling
                    if "Others" in all_bands:
                        all_bands.remove("Others")
                    
                    # Sort bands for consistent ordering
                    all_bands = sorted(list(all_bands))
                    
                    # Define color maps for different starting bands
                    colormap_assignments = {
                        'n25': ('Reds', cm.Reds),
                        'n71': ('Blues', cm.Blues),
                        'n41': ('Greens', cm.Greens),
                        'n260': ('Oranges', cm.Oranges),
                        'n261': ('Purples', cm.Purples),
                        'n77': ('YlOrRd', cm.YlOrRd),
                        'n78': ('YlGn', cm.YlGn),
                        'b2': ('BuPu', cm.BuPu),
                        'b4': ('GnBu', cm.GnBu),
                        'b12': ('RdPu', cm.RdPu),
                    }
                    
                    # Default colormap for bands that don't match any prefix
                    default_colormap_name = 'viridis'
                    default_colormap = cm.viridis
                    
                    # Group bands by their primary band (first band in combination)
                    band_groups = {}
                    
                    for band in all_bands:
                        # Get the primary band (first band in combination)
                        if ':' in band:
                            primary_band = band.split(':')[0]
                            complexity = band.count(':') + 1  # Number of bands in combination
                        else:
                            primary_band = band
                            complexity = 1
                        
                        # Find colormap for this primary band
                        assigned_colormap_name = default_colormap_name
                        assigned_colormap = default_colormap
                        
                        for prefix, (cmap_name, cmap) in colormap_assignments.items():
                            if primary_band.startswith(prefix):
                                assigned_colormap_name = cmap_name
                                assigned_colormap = cmap
                                break
                        
                        # Group by primary band and colormap
                        if primary_band not in band_groups:
                            band_groups[primary_band] = {
                                'colormap': assigned_colormap,
                                'bands_by_complexity': {}
                            }
                        
                        # Group by complexity within each primary band group
                        if complexity not in band_groups[primary_band]['bands_by_complexity']:
                            band_groups[primary_band]['bands_by_complexity'][complexity] = []
                        
                        band_groups[primary_band]['bands_by_complexity'][complexity].append(band)
                    
                    # Create the color mapping
                    color_map = {}
                    
                    for primary_band, group_data in band_groups.items():
                        cmap = group_data['colormap']
                        complexity_groups = group_data['bands_by_complexity']
                        
                        # Get all complexities for this primary band, sorted
                        complexities = sorted(complexity_groups.keys())
                        max_complexity = max(complexities)
                        
                        # Assign colors based on complexity
                        for complexity in complexities:
                            bands = sorted(complexity_groups[complexity])  # Sort alphabetically within complexity
                            
                            # Calculate color intensity based on complexity
                            # Single bands (complexity=1) get lighter colors (0.3-0.5)
                            # Higher complexity gets progressively darker (up to 0.9)
                            if max_complexity == 1:
                                # If only single bands exist, use middle range
                                color_intensity = 0.6
                            else:
                                # Map complexity to color intensity: 1->0.3, max->0.9
                                min_intensity = 0.3
                                max_intensity = 0.9
                                color_intensity = min_intensity + (complexity - 1) / (max_complexity - 1) * (max_intensity - min_intensity)
                            
                            # If multiple bands have the same complexity, spread them around the target intensity
                            if len(bands) > 1:
                                intensity_range = 0.1  # Small range around target intensity
                                intensities = np.linspace(
                                    max(0.2, color_intensity - intensity_range/2), 
                                    min(0.95, color_intensity + intensity_range/2), 
                                    len(bands)
                                )
                            else:
                                intensities = [color_intensity]
                            
                            # Assign colors to bands
                            for i, band in enumerate(bands):
                                color_map[band] = cmap(intensities[i])
                    
                    # Add "Others" with gray color
                    color_map["Others"] = (0.7, 0.7, 0.7, 1.0)
                    
                    return color_map

                # Main plotting code
                if 1:
                    # Process both NSA and SA data
                    processed_dl_ca_count = {
                        'NSA': process_bands(dl_ca_count['NSA']),
                        'SA': process_bands(dl_ca_count['SA'])
                    }

                    # Get color mapping
                    band_colors = get_band_color_mapping(processed_dl_ca_count)

                    # Find all unique CA values
                    all_cas = set()
                    for category in processed_dl_ca_count.values():
                        all_cas.update(category.keys())
                    max_ca = max(all_cas) if all_cas else 0

                    # Find all unique bands across all categories and CA levels
                    all_bands_set = set()
                    for category_data in processed_dl_ca_count.values():
                        for ca_data in category_data.values():
                            all_bands_set.update(ca_data.keys())
                    
                    # Custom sorting function for legend order
                    def sort_bands_for_legend(bands):
                        # Separate "Others" for special handling
                        others = [b for b in bands if b == "Others"]
                        regular_bands = [b for b in bands if b != "Others"]
                        
                        # Group bands by their primary band (first band in the combination)
                        band_groups = {}
                        for band in regular_bands:
                            if ':' in band:
                                primary = band.split(':')[0]
                            else:
                                primary = band
                            
                            if primary not in band_groups:
                                band_groups[primary] = []
                            band_groups[primary].append(band)
                        
                        # Sort each group by complexity (number of bands) in ASCENDING order (1, 2, 3 bands)
                        sorted_bands = []
                        for primary in sorted(band_groups.keys()):
                            group_bands = band_groups[primary]
                            # Sort by number of bands (complexity) ascending, then alphabetically
                            group_bands.sort(key=lambda x: (x.count(':'), x))
                            sorted_bands.extend(group_bands)
                        
                        # Add "Others" at the end
                        sorted_bands.extend(others)
                        
                        return sorted_bands

                    # Apply custom sorting to all_bands
                    all_bands = sort_bands_for_legend(list(all_bands_set))

                    # Create figure with stacked bar plot
                    fig, ax = plt.subplots(figsize=(12, 8))
                    fig.patch.set_edgecolor('black')
                    fig.patch.set_linewidth(2)

                    # Prepare data for stacked bars
                    categories = ['NSA', 'SA']
                    x_labels = []
                    bar_data = {band: [] for band in all_bands}

                    # Build x-axis labels and data
                    for category in categories:
                        for ca in range(1, max_ca + 1):
                            x_labels.append(f"{ca} CA\n({category})")
                            
                            # Get data for this CA and category
                            if ca in processed_dl_ca_count[category]:
                                ca_data = processed_dl_ca_count[category][ca]
                                for band in all_bands:
                                    bar_data[band].append(ca_data.get(band, 0))
                            else:
                                # If CA doesn't exist for this category, add zeros
                                for band in all_bands:
                                    bar_data[band].append(0)

                    # Create stacked bars in the sorted order
                    x_pos = range(len(x_labels))
                    bottom = [0] * len(x_labels)

                    for band in all_bands:  # This now uses the sorted order
                        ax.bar(x_pos, bar_data[band], bottom=bottom, 
                            label=band, color=band_colors[band], 
                            edgecolor='white', linewidth=1)
                        
                        # Update bottom for next stack
                        bottom = [b + v for b, v in zip(bottom, bar_data[band])]

                    # Customize the plot
                    ax.set_xlabel('CA Level and Technology', fontsize=14, fontweight='bold')
                    ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')

                    # Set x-axis
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(x_labels, fontsize=12)
                    
                    # Set y-axis to percentage
                    ax.set_ylim(0, 100)
                    
                    # Add grid for better readability
                    ax.grid(axis='y', alpha=0.3, linestyle='--')
                    
                    # Add legend
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                    
                    # Add value labels on bars (optional - only for segments > 5%)
                    for i, x in enumerate(x_pos):
                        cumulative = 0
                        for band in all_bands:
                            value = bar_data[band][i]
                            if value > 5:  # Only show labels for segments > 5%
                                y_pos = cumulative + value/2
                                ax.text(x, y_pos, f'{value:.1f}%', ha='center', va='center', 
                                    fontsize=8, fontweight='bold', color='white')
                            cumulative += value

                    # Adjust layout
                    plt.tight_layout()

                    # Save figure
                    plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ca_band_dl_distribution_stacked.pdf', 
                                bbox_inches='tight')
                    plt.close()
                    a= 1
            
            # plot band distribution - DL - stacked - hatched
            if 1:

                # Hatch mapping function
                def get_band_hatch_mapping(data):
                    # Get all unique bands
                    all_bands = set()
                    for category in data.values():
                        for ca_data in category.values():
                            all_bands.update(ca_data.keys())
                    
                    # Remove "Others" from the set for special handling
                    if "Others" in all_bands:
                        all_bands.remove("Others")
                    
                    # Sort bands for consistent ordering
                    all_bands = sorted(list(all_bands))
                    
                    # Define base colors for different starting bands
                    base_colors = {
                        'n25': 'lightcoral',
                        'n71': 'cyan', 
                        'n41': 'palegreen',
                        'n260': 'orange',
                        'n261': 'purple',
                        'n77': 'brown',
                        'n78': 'pink',
                        'b2': 'cyan',
                        'b4': 'magenta',
                        'b12': 'yellow',
                    }
                    
                    # Default color for bands that don't match any prefix
                    default_color = 'gray'
                    
                    # Define extensive hatch patterns for complexity levels
                    complexity_hatches = [
                        '',           # No hatch for single bands
                        '///',        # Diagonal lines
                        '\\\\\\',     # Reverse diagonal
                        '|||',        # Vertical lines
                        '---',        # Horizontal lines
                        '+++',        # Plus signs
                        'xxx',        # X marks
                        '...',        # Dots
                        'ooo',        # Circles
                        '***',        # Stars
                        '//o',        # Diagonal with circles
                        '\\\\*',      # Reverse diagonal with stars
                        '||+',        # Vertical with plus
                        '--.',        # Horizontal with dots
                        '//x',        # Diagonal with x
                        '\\\\.',      # Reverse diagonal with dots
                        '++o',        # Plus with circles
                        'x*x',        # X with stars
                        'o-o',        # Circles with horizontal
                        '|*|'         # Vertical with stars
                    ]
                    
                    # Group bands by their primary band (first band in combination)
                    band_groups = {}
                    
                    for band in all_bands:
                        # Get the primary band (first band in combination)
                        if ':' in band:
                            primary_band = band.split(':')[0]
                            complexity = band.count(':') + 1  # Number of bands in combination
                        else:
                            primary_band = band
                            complexity = 1
                        
                        # Find base color for this primary band
                        assigned_color = default_color
                        for prefix, color in base_colors.items():
                            if primary_band.startswith(prefix):
                                assigned_color = color
                                break
                        
                        # Group by primary band
                        if primary_band not in band_groups:
                            band_groups[primary_band] = {
                                'color': assigned_color,
                                'bands_by_complexity': {}
                            }
                        
                        # Group by complexity within each primary band group
                        if complexity not in band_groups[primary_band]['bands_by_complexity']:
                            band_groups[primary_band]['bands_by_complexity'][complexity] = []
                        
                        band_groups[primary_band]['bands_by_complexity'][complexity].append(band)
                    
                    # Create the color and hatch mapping
                    color_map = {}
                    hatch_map = {}
                    
                    for primary_band, group_data in band_groups.items():
                        base_color = group_data['color']
                        complexity_groups = group_data['bands_by_complexity']
                        
                        # Assign colors and hatches based on complexity
                        for complexity, bands in complexity_groups.items():
                            bands = sorted(bands)  # Sort alphabetically within complexity
                            hatch_pattern = complexity_hatches[min(complexity - 1, len(complexity_hatches) - 1)]
                            
                            # If there are multiple bands with same complexity, give them slightly different hatches
                            for i, band in enumerate(bands):
                                color_map[band] = base_color
                                if len(bands) > 1 and i > 0:
                                    # Use a different hatch variation for multiple bands of same complexity
                                    additional_hatch_idx = min(complexity - 1 + i, len(complexity_hatches) - 1)
                                    hatch_map[band] = complexity_hatches[additional_hatch_idx]
                                else:
                                    hatch_map[band] = hatch_pattern
                    
                    # Add "Others" with gray color and no hatch
                    color_map["Others"] = 'lightgray'
                    hatch_map["Others"] = ''
                    
                    return color_map, hatch_map

                if 1:
                    # Process both NSA and SA data
                    processed_dl_ca_count = {
                        'NSA': process_bands(dl_ca_count['NSA']),
                        'SA': process_bands(dl_ca_count['SA'])
                    }

                    # Get color and hatch mapping
                    band_colors, band_hatches = get_band_hatch_mapping(processed_dl_ca_count)
                    band_hatches['n41:n71:n25'] = '||'
                    band_hatches['n41:n71:n41'] = '|'
                    band_hatches['n41:n25:n41'] = '////'
                    band_hatches['n25:n71:n41:n41'] = '/'
                    band_hatches['n41:n25:n71'] = '//'
                    band_hatches['n41:n25:n25:n41'] = '/'
                    band_hatches['n41:n41:n25'] = '|||||'
                    # Find all unique CA values
                    all_cas = set()
                    for category in processed_dl_ca_count.values():
                        all_cas.update(category.keys())
                    max_ca = max(all_cas) if all_cas else 0

                    # Find all unique bands across all categories and CA levels
                    all_bands_set = set()
                    for category_data in processed_dl_ca_count.values():
                        for ca_data in category_data.values():
                            all_bands_set.update(ca_data.keys())

                    # Custom sorting function for legend order
                    def sort_bands_for_legend(bands):
                        # Separate "Others" for special handling
                        others = [b for b in bands if b == "Others"]
                        regular_bands = [b for b in bands if b != "Others"]
                        
                        # Group bands by their primary band (first band in the combination)
                        band_groups = {}
                        for band in regular_bands:
                            if ':' in band:
                                primary = band.split(':')[0]
                            else:
                                primary = band
                            
                            if primary not in band_groups:
                                band_groups[primary] = []
                            band_groups[primary].append(band)
                        
                        # Sort each group by complexity (number of bands) in ASCENDING order (1, 2, 3 bands)
                        sorted_bands = []
                        for primary in sorted(band_groups.keys()):
                            group_bands = band_groups[primary]
                            # Sort by number of bands (complexity) ascending, then alphabetically
                            group_bands.sort(key=lambda x: (x.count(':'), x))
                            sorted_bands.extend(group_bands)
                        
                        # Add "Others" at the end
                        sorted_bands.extend(others)
                        
                        return sorted_bands

                    # Apply custom sorting to all_bands
                    all_bands = sort_bands_for_legend(list(all_bands_set))

                    # Create figure with stacked bar plot
                    fig, ax = plt.subplots(figsize=(10, 7))  # Increased height to accommodate legend
                    fig.patch.set_edgecolor('black')
                    fig.patch.set_linewidth(2)

                    # Prepare data for stacked bars
                    categories = ['NSA', 'SA']
                    x_labels = []
                    bar_data = {band: [] for band in all_bands}

                    # Build x-axis labels and data
                    for category in categories:
                        for ca in range(1, max_ca + 1):
                            x_labels.append(f"{ca} CA\n({category})")
                            
                            # Get data for this CA and category
                            if ca in processed_dl_ca_count[category]:
                                ca_data = processed_dl_ca_count[category][ca]
                                for band in all_bands:
                                    bar_data[band].append(ca_data.get(band, 0))
                            else:
                                # If CA doesn't exist for this category, add zeros
                                for band in all_bands:
                                    bar_data[band].append(0)

                    # Create stacked bars in the sorted order
                    x_pos = range(len(x_labels))
                    bottom = [0] * len(x_labels)

                    for band in all_bands:  # This now uses the sorted order
                        ax.bar(x_pos, bar_data[band], bottom=bottom, 
                            label=band, color=band_colors[band], hatch=band_hatches[band],
                            edgecolor='black', linewidth=0.5)
                        
                        # Update bottom for next stack
                        bottom = [b + v for b, v in zip(bottom, bar_data[band])]

                    # Customize the plot
                    ax.set_xlabel('CA Level and Technology', fontsize=14, fontweight='bold')
                    ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')

                    # Set x-axis
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(x_labels, fontsize=12)
                    
                    # Set y-axis to percentage
                    ax.set_ylim(0, 100)
                    
                    # Add grid for better readability
                    ax.grid(axis='y', alpha=0.3, linestyle='--')
                    
                    # Create custom legend with colors and hatches - positioned at top in 5 rows
                    legend_elements = []
                    for band in all_bands:
                        legend_elements.append(plt.Rectangle((0, 0), 1, 1, 
                                                        facecolor=band_colors[band], 
                                                        hatch=band_hatches[band],
                                                        edgecolor='black',
                                                        label=band))
                    
                    # Calculate number of columns based on number of bands and desired 5 rows
                    ncols = max(1, len(all_bands) // 5 + (1 if len(all_bands) % 5 > 0 else 0))
                    
                    # Place legend at the top of the figure
                    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.3), 
                            ncol=ncols, fontsize=10, frameon=True, handlelength=2.5, handleheight=1.5)

                    # Add value labels on bars (optional - only for segments > 5%)
                    for i, x in enumerate(x_pos):
                        cumulative = 0
                        for band in all_bands:
                            value = bar_data[band][i]
                            if value > 5:  # Only show labels for segments > 5%
                                y_pos = cumulative + value/2
                                cumulative += value

                    # Adjust layout to make room for the legend at the top
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.8)  # Make room for legend at the top

                    # Save figure
                    plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ca_band_dl_distribution_stacked_hatched.pdf',
                                bbox_inches='tight')
                    plt.close()
                    a = 1

            # plot band distribution - UL
            if 0:
                # Process both NSA and SA data
                processed_ul_ca_count = {
                    'NSA': process_bands(ul_ca_count['NSA']),
                    'SA': process_bands(ul_ca_count['SA'])
                }

                # Get color mapping
                band_colors = get_band_color_mapping(processed_ul_ca_count)

                # Find all unique CA values
                all_cas = set()
                for category in processed_ul_ca_count.values():
                    all_cas.update(category.keys())
                max_ca = max(all_cas) if all_cas else 0

                # Create figure with 2 rows (NSA, SA) and max_ca columns
                fig, axes = plt.subplots(2, max_ca, figsize=(5*max_ca, 8))

                fig.patch.set_edgecolor('black')
                fig.patch.set_linewidth(2)
                # If max_ca is 1, axes will not be a 2D array, so convert to 2D
                if max_ca == 1:
                    axes = np.array(axes).reshape(2, 1)

                # Categories to plot
                categories = ['NSA', 'SA']

                # Plot pie charts
                for row, category in enumerate(categories):
                    for col in range(max_ca):
                        ca = col + 1  # CA values start from 1
                        
                        # If this CA exists for this category
                        if ca in processed_ul_ca_count[category]:
                            ax = axes[row, col]
                            
                            # Get band data for this CA
                            bands_data = processed_ul_ca_count[category][ca]
                            labels = list(bands_data.keys())
                            sizes = list(bands_data.values())
                            
                            # Create label text with percentages
                            # label_texts = [f"{label}\n({size:.1f}%)" for label, size in zip(labels, sizes)]
                            label_texts = [label for label, size in zip(labels, sizes)]
                            # Get colors for each band
                            colors = [band_colors[band] for band in labels]
                            
                            # Use matplotlib's built-in pie chart with external labels
                            wedges, texts = ax.pie(
                                sizes,
                                colors=colors,
                                labels=label_texts,
                                labeldistance=1.1,  # Position labels just outside the pie
                                wedgeprops={'edgecolor': 'w', 'linewidth': 1},
                                textprops={'fontsize': 16, 'ha': 'center'},  # Increased font size from 8 to 10
                                startangle=90,
                                radius=0.8  # Make pie slightly smaller to leave room for labels
                            )
                            
                            # Draw connecting lines from pie to labels
                            for i, wedge in enumerate(wedges):
                                if sizes[i] < 3:  # Skip very small wedges to reduce clutter
                                    texts[i].set_visible(False)
                                    continue
                                
                                # Adjust text alignment for better readability
                                angle = (wedge.theta1 + wedge.theta2) / 2
                                if 90 < angle < 270:  # Left side of the pie
                                    texts[i].set_ha('right')
                                else:  # Right side of the pie
                                    texts[i].set_ha('left')
                            
                            # Modified title format: "CA # (NSA/SA)" and made larger and bold
                            ax.set_title(f"{ca} CA ({category})", fontsize=16, fontweight='bold')
                        else:
                            # If this CA doesn't exist for this category, hide the axes
                            axes[row, col].axis('off')

                # Adjust layout with extra padding for labels
                plt.tight_layout(pad=2.0)

                # Save figure
                plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ca_band_ul_distribution.pdf', 
                            bbox_inches='tight')
                plt.close()
                a = 1

            # plot band distribution - UL - hatched
            if 1:
                # def get_band_hatch_mapping(band_colors):
                #     # Define extensive hatch patterns for different bands
                #     hatch_patterns = [
                #         '',           # No hatch
                #         '///',        # Diagonal lines
                #         '\\\\\\',     # Reverse diagonal
                #         '|||',        # Vertical lines
                #         '---',        # Horizontal lines
                #         '+++',        # Plus signs
                #         'xxx',        # X marks
                #         '...',        # Dots
                #         'ooo',        # Circles
                #         '***',        # Stars
                #         '//o',        # Diagonal with circles
                #         '\\\\*',      # Reverse diagonal with stars
                #         '||+',        # Vertical with plus
                #         '--.',        # Horizontal with dots
                #         '//x',        # Diagonal with x
                #         '\\\\.',      # Reverse diagonal with dots
                #         '++o',        # Plus with circles
                #         'x*x',        # X with stars
                #         'o-o',        # Circles with horizontal
                #         '|*|'         # Vertical with stars
                #     ]
                    
                #     # Create hatch mapping based on band complexity and alphabetical order
                #     band_hatch_map = {}
                #     sorted_bands = sorted([band for band in band_colors.keys() if band != 'Others'])
                    
                #     for i, band in enumerate(sorted_bands):
                #         # Determine complexity (number of colons + 1)
                #         if ':' in band:
                #             complexity = band.count(':') + 1
                #         else:
                #             complexity = 1
                        
                #         # Assign hatch based on complexity and position
                #         hatch_idx = min(complexity - 1 + (i % 3), len(hatch_patterns) - 1)
                #         band_hatch_map[band] = hatch_patterns[hatch_idx]
                    
                #     # Others gets no hatch
                #     band_hatch_map['Others'] = ''
                    
                #     return band_hatch_map
                
                def get_band_hatch_mapping(data):
                    # Get all unique bands
                    all_bands = set()
                    for category in data.values():
                        for ca_data in category.values():
                            all_bands.update(ca_data.keys())
                    
                    # Remove "Others" from the set for special handling
                    if "Others" in all_bands:
                        all_bands.remove("Others")
                    
                    # Sort bands for consistent ordering
                    all_bands = sorted(list(all_bands))
                    
                    # Define base colors for different starting bands
                    base_colors = {
                        'n25': 'lightcoral',
                        'n71': 'cyan', 
                        'n41': 'palegreen',
                        'n260': 'orange',
                        'n261': 'purple',
                        'n77': 'brown',
                        'n78': 'pink',
                        'b2': 'cyan',
                        'b4': 'magenta',
                        'b12': 'yellow',
                    }
                    
                    # Default color for bands that don't match any prefix
                    default_color = 'gray'
                    
                    # Define extensive hatch patterns for complexity levels
                    complexity_hatches = [
                        '',           # No hatch for single bands
                        '///',        # Diagonal lines
                        '\\\\\\',     # Reverse diagonal
                        '|||',        # Vertical lines
                        '---',        # Horizontal lines
                        '+++',        # Plus signs
                        'xxx',        # X marks
                        '...',        # Dots
                        'ooo',        # Circles
                        '***',        # Stars
                        '//o',        # Diagonal with circles
                        '\\\\*',      # Reverse diagonal with stars
                        '||+',        # Vertical with plus
                        '--.',        # Horizontal with dots
                        '//x',        # Diagonal with x
                        '\\\\.',      # Reverse diagonal with dots
                        '++o',        # Plus with circles
                        'x*x',        # X with stars
                        'o-o',        # Circles with horizontal
                        '|*|'         # Vertical with stars
                    ]
                    
                    # Group bands by their primary band (first band in combination)
                    band_groups = {}
                    
                    for band in all_bands:
                        # Get the primary band (first band in combination)
                        if ':' in band:
                            primary_band = band.split(':')[0]
                            complexity = band.count(':') + 1  # Number of bands in combination
                        else:
                            primary_band = band
                            complexity = 1
                        
                        # Find base color for this primary band
                        assigned_color = default_color
                        for prefix, color in base_colors.items():
                            if primary_band.startswith(prefix):
                                assigned_color = color
                                break
                        
                        # Group by primary band
                        if primary_band not in band_groups:
                            band_groups[primary_band] = {
                                'color': assigned_color,
                                'bands_by_complexity': {}
                            }
                        
                        # Group by complexity within each primary band group
                        if complexity not in band_groups[primary_band]['bands_by_complexity']:
                            band_groups[primary_band]['bands_by_complexity'][complexity] = []
                        
                        band_groups[primary_band]['bands_by_complexity'][complexity].append(band)
                    
                    # Create the color and hatch mapping
                    color_map = {}
                    hatch_map = {}
                    
                    for primary_band, group_data in band_groups.items():
                        base_color = group_data['color']
                        complexity_groups = group_data['bands_by_complexity']
                        
                        # Assign colors and hatches based on complexity
                        for complexity, bands in complexity_groups.items():
                            bands = sorted(bands)  # Sort alphabetically within complexity
                            hatch_pattern = complexity_hatches[min(complexity - 1, len(complexity_hatches) - 1)]
                            
                            # If there are multiple bands with same complexity, give them slightly different hatches
                            for i, band in enumerate(bands):
                                color_map[band] = base_color
                                if len(bands) > 1 and i > 0:
                                    # Use a different hatch variation for multiple bands of same complexity
                                    additional_hatch_idx = min(complexity - 1 + i, len(complexity_hatches) - 1)
                                    hatch_map[band] = complexity_hatches[additional_hatch_idx]
                                else:
                                    hatch_map[band] = hatch_pattern
                    
                    # Add "Others" with gray color and no hatch
                    color_map["Others"] = 'lightgray'
                    hatch_map["Others"] = ''
                    
                    return color_map, hatch_map

                # Process both NSA and SA data
                processed_ul_ca_count = {
                    'NSA': process_bands(ul_ca_count['NSA']),
                    'SA': process_bands(ul_ca_count['SA'])
                }

                # Get color and hatch mapping
                # band_colors, band_hatches = get_band_hatch_mapping(processed_ul_ca_count)

                # Find all unique CA values
                all_cas = set()
                for category in processed_ul_ca_count.values():
                    all_cas.update(category.keys())
                max_ca = max(all_cas) if all_cas else 0

                # Find all unique bands across all categories and CA levels
                all_bands_set = set()
                for category_data in processed_ul_ca_count.values():
                    for ca_data in category_data.values():
                        all_bands_set.update(ca_data.keys())
                
                # Custom sorting function for legend order
                def sort_bands_for_legend(bands):
                    # Separate "Others" for special handling
                    others = [b for b in bands if b == "Others"]
                    regular_bands = [b for b in bands if b != "Others"]
                    
                    # Group bands by their primary band (first band in the combination)
                    band_groups = {}
                    for band in regular_bands:
                        if ':' in band:
                            primary = band.split(':')[0]
                        else:
                            primary = band
                        
                        if primary not in band_groups:
                            band_groups[primary] = []
                        band_groups[primary].append(band)
                    
                    # Sort each group by complexity (number of bands) in ASCENDING order (1, 2, 3 bands)
                    sorted_bands = []
                    for primary in sorted(band_groups.keys()):
                        group_bands = band_groups[primary]
                        # Sort by number of bands (complexity) ascending, then alphabetically
                        group_bands.sort(key=lambda x: (x.count(':'), x))
                        sorted_bands.extend(group_bands)
                    
                    # Add "Others" at the end
                    sorted_bands.extend(others)
                    
                    return sorted_bands

                # Apply custom sorting to all_bands
                all_bands = sort_bands_for_legend(list(all_bands_set))

                # Create figure with stacked bar plot
                fig, ax = plt.subplots(figsize=(5, 6))
                fig.patch.set_edgecolor('black')
                fig.patch.set_linewidth(2)

                # Prepare data for stacked bars
                categories = ['NSA', 'SA']
                x_labels = []
                bar_data = {band: [] for band in all_bands}

                # Build x-axis labels and data
                for category in categories:
                    for ca in range(1, max_ca + 1):
                        x_labels.append(f"{ca} CA\n({category})")
                        
                        # Get data for this CA and category
                        if ca in processed_ul_ca_count[category]:
                            ca_data = processed_ul_ca_count[category][ca]
                            for band in all_bands:
                                bar_data[band].append(ca_data.get(band, 0))
                        else:
                            # If CA doesn't exist for this category, add zeros
                            for band in all_bands:
                                bar_data[band].append(0)

                # Create stacked bars in the sorted order
                x_pos = range(len(x_labels))
                bottom = [0] * len(x_labels)

                for band in all_bands:  # This now uses the sorted order
                    ax.bar(x_pos, bar_data[band], bottom=bottom, 
                        label=band, color=band_colors[band], hatch=band_hatches[band],
                        edgecolor='black', linewidth=0.5)
                    
                    # Update bottom for next stack
                    bottom = [b + v for b, v in zip(bottom, bar_data[band])]

                # Customize the plot
                ax.set_xlabel('CA Level and Technology', fontsize=14, fontweight='bold')
                ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')

                # Set x-axis
                ax.set_xticks(x_pos)
                ax.set_xticklabels(x_labels, fontsize=12)
                
                # Set y-axis to percentage
                ax.set_ylim(0, 100)
                
                # Add grid for better readability
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                
                # Create custom legend with colors and hatches - positioned at top in 2 rows
                # legend_elements = []
                # for band in all_bands:
                #     legend_elements.append(plt.Rectangle((0, 0), 1, 1, 
                #                                     facecolor=band_colors[band], 
                #                                     hatch=band_hatches[band],
                #                                     edgecolor='black',
                #                                     label=band))
                
                # # Calculate number of columns for 2 rows
                # ncols = max(1, len(all_bands) // 2 + (1 if len(all_bands) % 2 > 0 else 0))
                
                # # Place legend at the top of the figure in 2 rows
                # fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.99), 
                #         ncol=ncols, fontsize=14, frameon=True)

                # Add value labels on bars (optional - only for segments > 5%)
                for i, x in enumerate(x_pos):
                    cumulative = 0
                    for band in all_bands:
                        value = bar_data[band][i]
                        if value > 5:  # Only show labels for segments > 5%
                            y_pos = cumulative + value/2
                        cumulative += value

                # Adjust layout to make room for the legend at the top
                plt.tight_layout()
                # plt.subplots_adjust(top=0.85)  # Make room for legend at the top

                # Save figure
                plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ca_band_ul_distribution_stacked_hatched.pdf', 
                            bbox_inches='tight')
                plt.close()
                a = 1
        # bw - dl
        if 1:
            dl_bw_overall_distribution = {}
            dl_bw_xput_distribution = {}


            for tech in main_tech_xput_dl_bandwidth_dict_3.keys():
                dl_bw_overall_distribution[tech] = {}
                dl_bw_xput_distribution[tech] = {}
                temp = {}
                for bw in main_tech_xput_dl_bandwidth_dict_3[tech].keys():
                    if bw > 0:
                        temp[bw] = len(main_tech_xput_dl_bandwidth_dict_3[tech][bw])
                        dl_bw_xput_distribution[tech][int(bw)] = main_tech_xput_dl_bandwidth_dict_3[tech][bw].copy()

                temp = {k: (v / sum(temp.values())) * 100 for k, v in temp.items()}
                dl_bw_overall_distribution[tech] = temp.copy()


            # plot DL 
            if 1:
                # Get all unique bandwidth values across both NSA and SA
                all_bandwidths = set()
                for category in dl_bw_xput_distribution.values():
                    all_bandwidths.update(category.keys())
                all_bandwidths = sorted(all_bandwidths)

                # Create a mapping from bandwidth to x-position
                bw_to_pos = {bw: i for i, bw in enumerate(all_bandwidths)}

                # Create figure and subplots (stacked vertically)
                fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True, sharey=True)

                # Plot NSA data (top subplot)
                nsa_data = []
                nsa_positions = []
                nsa_labels = []
                nsa_sample_sizes = []

                # Calculate the total number of NSA samples for percentage calculation
                total_nsa_samples = 0
                for bw in all_bandwidths:
                    if bw in dl_bw_xput_distribution['NSA']:
                        total_nsa_samples += len(dl_bw_xput_distribution['NSA'][bw])

                for bw in all_bandwidths:
                    if bw in dl_bw_xput_distribution['NSA']:
                        nsa_data.append(dl_bw_xput_distribution['NSA'][bw])
                        nsa_positions.append(bw_to_pos[bw])
                        nsa_labels.append(str(bw))
                        # Calculate percentage instead of absolute sample size
                        sample_count = len(dl_bw_xput_distribution['NSA'][bw])
                        percentage = (sample_count / total_nsa_samples) * 100
                        nsa_sample_sizes.append(percentage)

                if nsa_data:  # Check if there's data to plot
                    # Store the boxplot result to access flier positions
                    nsa_boxplot = axes[0].boxplot(nsa_data, positions=nsa_positions, patch_artist=True,
                                boxprops=dict(facecolor='lightblue'))
                    axes[0].set_ylabel('NSA\nThroughput')
                    axes[0].grid(True, linestyle='--', alpha=0.7)
                    
                    # Annotate sample percentages above each box
                    for i, (pos, percentage) in enumerate(zip(nsa_positions, nsa_sample_sizes)):
                        if nsa_data[i]:  # Ensure there's data to plot
                            data_array = np.array(nsa_data[i])
                            
                            # Find the max value, including outliers
                            max_y = np.max(data_array)
                            
                            # Check for outliers/fliers
                            fliers = nsa_boxplot['fliers'][i].get_ydata()
                            if len(fliers) > 0:
                                max_y = max(max_y, np.max(fliers))
                            
                            # Add buffer above the highest point (outlier or whisker)
                            y_buffer = 0.02 * (axes[0].get_ylim()[1] - axes[0].get_ylim()[0])
                            
                            # Add the annotation with percentage
                            axes[0].annotate(f'{percentage:.1f}%', xy=(pos, max_y + y_buffer), 
                                        xytext=(0, 5), textcoords='offset points',
                                        ha='center', va='bottom', fontsize=9, fontweight='bold')

                # Plot SA data (bottom subplot)
                sa_data = []
                sa_positions = []
                sa_labels = []
                sa_sample_sizes = []

                # Calculate the total number of SA samples for percentage calculation
                total_sa_samples = 0
                for bw in all_bandwidths:
                    if bw in dl_bw_xput_distribution['SA']:
                        total_sa_samples += len(dl_bw_xput_distribution['SA'][bw])

                for bw in all_bandwidths:
                    if bw in dl_bw_xput_distribution['SA']:
                        sa_data.append(dl_bw_xput_distribution['SA'][bw])
                        sa_positions.append(bw_to_pos[bw])
                        sa_labels.append(str(bw))
                        # Calculate percentage instead of absolute sample size
                        sample_count = len(dl_bw_xput_distribution['SA'][bw])
                        percentage = (sample_count / total_sa_samples) * 100
                        sa_sample_sizes.append(percentage)

                if sa_data:  # Check if there's data to plot
                    # Store the boxplot result to access flier positions
                    sa_boxplot = axes[1].boxplot(sa_data, positions=sa_positions, patch_artist=True,
                                boxprops=dict(facecolor='lightblue'))
                    axes[1].set_ylabel('SA\nThroughput')
                    axes[1].set_xlabel('Sum Bandwidth (MHz)')
                    axes[1].grid(True, linestyle='--', alpha=0.7)
                    
                    # Annotate sample percentages above each box
                    for i, (pos, percentage) in enumerate(zip(sa_positions, sa_sample_sizes)):
                        if sa_data[i]:  # Ensure there's data to plot
                            data_array = np.array(sa_data[i])
                            
                            # Find the max value, including outliers
                            max_y = np.max(data_array)
                            
                            # Check for outliers/fliers
                            fliers = sa_boxplot['fliers'][i].get_ydata()
                            if len(fliers) > 0:
                                max_y = max(max_y, np.max(fliers))
                            
                            # Add buffer above the highest point (outlier or whisker)
                            y_buffer = 0.02 * (axes[1].get_ylim()[1] - axes[1].get_ylim()[0])
                            
                            # Add the annotation with percentage
                            axes[1].annotate(f'{percentage:.1f}%', xy=(pos, max_y + y_buffer), 
                                        xytext=(0, 5), textcoords='offset points',
                                        ha='center', va='bottom', fontsize=9, fontweight='bold')

                # Set x-ticks and labels to be the same for both plots
                axes[1].set_xticks(range(len(all_bandwidths)))
                axes[1].set_xticklabels([str(bw) for bw in all_bandwidths], rotation=45, ha='right')
                axes[1].set_ylim(0, 1900)
                # Adjust layout
                plt.tight_layout()

                # Save the figure
                plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/bw_dl_xput_distribution.pdf')
                plt.close()

        # bw - ul 
        if 0:
            ul_bw_overall_distribution = {}
            ul_bw_xput_distribution = {}

            for tech in main_tech_xput_ul_bandwidth_dict_3.keys():
                ul_bw_overall_distribution[tech] = {}
                ul_bw_xput_distribution[tech] = {}
                temp = {}
                for bw in main_tech_xput_ul_bandwidth_dict_3[tech].keys():
                    if bw > 0:
                        temp[bw] = len(main_tech_xput_ul_bandwidth_dict_3[tech][bw])
                        ul_bw_xput_distribution[tech][int(bw)] = main_tech_xput_ul_bandwidth_dict_3[tech][bw].copy()

                temp = {k: (v / sum(temp.values())) * 100 for k, v in temp.items()}
                ul_bw_overall_distribution[tech] = temp.copy()

            # plot ul 
            if 1:
                # Get all unique bandwidth values across both NSA and SA
                all_bandwidths = set()
                for category in ul_bw_xput_distribution.values():
                    all_bandwidths.update(category.keys())
                all_bandwidths = sorted(all_bandwidths)

                # Create a mapping from bandwidth to x-position
                bw_to_pos = {bw: i for i, bw in enumerate(all_bandwidths)}

                # Create figure and subplots (stacked vertically)
                fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, sharey=True)

                # Plot NSA data (top subplot)
                nsa_data = []
                nsa_positions = []
                nsa_labels = []
                nsa_sample_sizes = []

                # Calculate total NSA samples for percentage calculation
                total_nsa_samples = 0
                for bw in all_bandwidths:
                    if bw in ul_bw_xput_distribution['NSA']:
                        total_nsa_samples += len(ul_bw_xput_distribution['NSA'][bw])

                for bw in all_bandwidths:
                    if bw in ul_bw_xput_distribution['NSA']:
                        nsa_data.append(ul_bw_xput_distribution['NSA'][bw])
                        nsa_positions.append(bw_to_pos[bw])
                        nsa_labels.append(str(bw))
                        # Calculate percentage instead of absolute count
                        sample_count = len(ul_bw_xput_distribution['NSA'][bw])
                        percentage = (sample_count / total_nsa_samples) * 100
                        nsa_sample_sizes.append(percentage)

                if nsa_data:  # Check if there's data to plot
                    # Store the boxplot result to access flier positions
                    nsa_boxplot = axes[0].boxplot(nsa_data, positions=nsa_positions, patch_artist=True,
                                boxprops=dict(facecolor='lightblue'))
                    axes[0].set_ylabel('NSA\nThroughput')
                    axes[0].grid(True, linestyle='--', alpha=0.7)
                    
                    # Annotate percentages above each box
                    for i, (pos, percentage) in enumerate(zip(nsa_positions, nsa_sample_sizes)):
                        if nsa_data[i]:  # Ensure there's data to plot
                            data_array = np.array(nsa_data[i])
                            
                            # Find the max value, including outliers
                            max_y = np.max(data_array)
                            
                            # Check for outliers/fliers
                            fliers = nsa_boxplot['fliers'][i].get_ydata()
                            if len(fliers) > 0:
                                max_y = max(max_y, np.max(fliers))
                            
                            # Add buffer above the highest point (outlier or whisker)
                            y_buffer = 0.02 * (axes[0].get_ylim()[1] - axes[0].get_ylim()[0])
                            
                            # Add the annotation with percentage format
                            axes[0].annotate(f'{percentage:.1f}%', xy=(pos, max_y + y_buffer), 
                                        xytext=(0, 5), textcoords='offset points',
                                        ha='center', va='bottom', fontsize=9, fontweight='bold')

                # Plot SA data (bottom subplot)
                sa_data = []
                sa_positions = []
                sa_labels = []
                sa_sample_sizes = []

                # Calculate total SA samples for percentage calculation
                total_sa_samples = 0
                for bw in all_bandwidths:
                    if bw in ul_bw_xput_distribution['SA']:
                        total_sa_samples += len(ul_bw_xput_distribution['SA'][bw])

                for bw in all_bandwidths:
                    if bw in ul_bw_xput_distribution['SA']:
                        sa_data.append(ul_bw_xput_distribution['SA'][bw])
                        sa_positions.append(bw_to_pos[bw])
                        sa_labels.append(str(bw))
                        # Calculate percentage instead of absolute count
                        sample_count = len(ul_bw_xput_distribution['SA'][bw])
                        percentage = (sample_count / total_sa_samples) * 100
                        sa_sample_sizes.append(percentage)

                if sa_data:  # Check if there's data to plot
                    # Store the boxplot result to access flier positions
                    sa_boxplot = axes[1].boxplot(sa_data, positions=sa_positions, patch_artist=True,
                                boxprops=dict(facecolor='lightblue'))
                    axes[1].set_ylabel('SA\nThroughput')
                    axes[1].set_xlabel('Sum Bandwidth (MHz)')
                    axes[1].grid(True, linestyle='--', alpha=0.7)
                    
                    # Annotate percentages above each box
                    for i, (pos, percentage) in enumerate(zip(sa_positions, sa_sample_sizes)):
                        if sa_data[i]:  # Ensure there's data to plot
                            data_array = np.array(sa_data[i])
                            
                            # Find the max value, including outliers
                            max_y = np.max(data_array)
                            
                            # Check for outliers/fliers
                            fliers = sa_boxplot['fliers'][i].get_ydata()
                            if len(fliers) > 0:
                                max_y = max(max_y, np.max(fliers))
                            
                            # Add buffer above the highest point (outlier or whisker)
                            y_buffer = 0.02 * (axes[1].get_ylim()[1] - axes[1].get_ylim()[0])
                            
                            # Add the annotation with percentage format
                            axes[1].annotate(f'{percentage:.1f}%', xy=(pos, max_y + y_buffer), 
                                        xytext=(0, 5), textcoords='offset points',
                                        ha='center', va='bottom', fontsize=9, fontweight='bold')

                # Set x-ticks and labels to be the same for both plots
                axes[1].set_xticks(range(len(all_bandwidths)))
                axes[1].set_xticklabels([str(bw) for bw in all_bandwidths], rotation=45, ha='right')
                axes[1].set_ylim(0, 250)
                # Adjust layout
                plt.tight_layout()

                # Save the figure
                plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/bw_ul_xput_distribution.pdf')
                plt.close()

        # bw + ca - 3D
        if 0:
            # plot 
            # Use default settings inside this block
            with mpl.rc_context(rc={}):  # empty rc will reset to default
                # Optionally reinitialize defaults manually
                mpl.rcdefaults()
                # DL
                # main_tech_xput_dl_ca_bandwidth_dict
                dl_ca_bw_xput_dict = {}
                for tech in main_tech_xput_dl_ca_bandwidth_dict_3.keys():
                    dl_ca_bw_xput_dict[tech] = []
                    dl_ca_list = []
                    dl_bw_list = []
                    dl_xput_list = []
                    for ca_bw in main_tech_xput_dl_ca_bandwidth_dict_3[tech].keys():
                        if ca_bw[2] > 0:
                            ca = ca_bw[0]
                            band = ca_bw[1]
                            bw = ca_bw[2]
                            if ca != len(band.split(":")) - 1:
                                continue
                            dl_xput_list.extend(main_tech_xput_dl_ca_bandwidth_dict_3[tech][ca_bw])
                            dl_ca_list.extend([ca] * len(main_tech_xput_dl_ca_bandwidth_dict_3[tech][ca_bw]))
                            dl_bw_list.extend([bw] * len(main_tech_xput_dl_ca_bandwidth_dict_3[tech][ca_bw]))
                    dl_ca_bw_xput_dict[tech] = pd.DataFrame({'CA': dl_ca_list, 'BW': dl_bw_list, 'Xput': dl_xput_list})

                    df = dl_ca_bw_xput_dict[tech]
                    # Plot
                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111, projection='3d')

                    # Color mapping based on throughput
                    sc = ax.scatter(df['BW'], df['CA'], df['Xput'], 
                                    c=df['Xput'], cmap='viridis', s=50)

                    # Axis labels
                    ax.set_xlabel('Bandwidth (BW)')
                    ax.set_ylabel('Carrier Aggregation (CA)')
                    ax.set_yticks([1, 2, 3, 4])
                    ax.set_zlabel('Throughput (Xput)')
                    ax.set_zlim(0, 1600)

                    # Add color bar to show the throughput scale
                    cb = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
                    cb.set_label('Throughput')


                    # Show plot
                    plt.tight_layout()
                    plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/dl_ca_bw_xput_%s_2024.png' %tech, dpi=300)
                    plt.close()


                # ul
                # main_tech_xput_ul_ca_bandwidth_dict
                ul_ca_bw_xput_dict = {}
                for tech in main_tech_xput_ul_ca_bandwidth_dict_3.keys():
                    ul_ca_bw_xput_dict[tech] = []
                    ul_ca_list = []
                    ul_bw_list = []
                    ul_xput_list = []
                    for ca_bw in main_tech_xput_ul_ca_bandwidth_dict_3[tech].keys():
                        if ca_bw[2] > 0:
                            ca = ca_bw[0]
                            band = ca_bw[1]
                            bw = ca_bw[2]
                            if ca != len(band.split(":")) - 1:
                                continue
                            ul_xput_list.extend(main_tech_xput_ul_ca_bandwidth_dict_3[tech][ca_bw])
                            ul_ca_list.extend([ca] * len(main_tech_xput_ul_ca_bandwidth_dict_3[tech][ca_bw]))
                            ul_bw_list.extend([bw] * len(main_tech_xput_ul_ca_bandwidth_dict_3[tech][ca_bw]))
                    ul_ca_bw_xput_dict[tech] = pd.DataFrame({'CA': ul_ca_list, 'BW': ul_bw_list, 'Xput': ul_xput_list})

                    df = ul_ca_bw_xput_dict[tech]
                    # Plot
                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111, projection='3d')

                    # Color mapping based on throughput
                    sc = ax.scatter(df['BW'], df['CA'], df['Xput'], 
                                    c=df['Xput'], cmap='viridis', s=50)

                    # Axis labels
                    ax.set_xlabel('Bandwidth (BW)')
                    ax.set_ylabel('Carrier Aggregation (CA)')
                    ax.set_yticks([1, 2])
                    ax.set_zlabel('Throughput (Xput)')
                    ax.set_zlim(0, 200)

                    # Add color bar to show the throughput scale
                    cb = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
                    cb.set_label('Throughput')


                    # Show plot
                    plt.tight_layout()
                    plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ul_ca_bw_xput_%s_2024.png' %tech, dpi=300)
                    plt.close()

        # bw + ca + xput - 2D
        if 0:
            # plot 
            # Define markers for different CA values
            ca_markers = {1: 'o', 2: 's', 3: '^', 4: 'D', 5: 'v', 6: '<', 7: '>', 8: 'p'}
            ca_colors = {1: 'blue', 2: 'red', 3: 'green', 4: 'orange', 5: 'purple', 6: 'brown', 7: 'pink', 8: 'gray'}
            
            # Define marker sizes - largest for 1-CA, smallest for highest CA
            def get_marker_size(ca_value, max_ca):
                # Start with base size 120 for 1-CA, decrease by 15 for each higher CA
                ca_marker_size_dict = {1: 120, 2: 90, 3: 50, 4: 20}
                # return max(120 - (ca_value - 1) * 15, 30)  # Minimum size of 30
                return ca_marker_size_dict[ca_value]
            
            # DL
            dl_ca_bw_xput_dict = {}
            for tech in main_tech_xput_dl_ca_bandwidth_dict_3.keys():
                dl_ca_bw_xput_dict[tech] = []
                dl_ca_list = []
                dl_bw_list = []
                dl_xput_list = []
                for ca_bw in main_tech_xput_dl_ca_bandwidth_dict_3[tech].keys():
                    if ca_bw[2] > 0:
                        ca = ca_bw[0]
                        band = ca_bw[1]
                        bw = ca_bw[2]
                        if ca != len(band.split(":")) - 1:
                            continue
                        dl_xput_list.extend(main_tech_xput_dl_ca_bandwidth_dict_3[tech][ca_bw])
                        dl_ca_list.extend([ca] * len(main_tech_xput_dl_ca_bandwidth_dict_3[tech][ca_bw]))
                        dl_bw_list.extend([bw] * len(main_tech_xput_dl_ca_bandwidth_dict_3[tech][ca_bw]))
                dl_ca_bw_xput_dict[tech] = pd.DataFrame({'CA': dl_ca_list, 'BW': dl_bw_list, 'Xput': dl_xput_list})

                df = dl_ca_bw_xput_dict[tech]
                
                # Create 2D scatter plot
                fig, ax = plt.subplots(figsize=(6, 3.5))
                
                # Get max CA value for size calculation
                max_ca = df['CA'].max()
                
                # Plot each CA value with different markers and sizes
                # Plot in ascending order of CA so higher CA (smaller markers) are plotted on top
                for ca_value in sorted(df['CA'].unique()):
                    ca_data = df[df['CA'] == ca_value]
                    marker_size = get_marker_size(ca_value, max_ca)
                    
                    ax.scatter(ca_data['BW'], ca_data['Xput'], 
                            marker=ca_markers.get(ca_value, 'o'), 
                            color=ca_colors.get(ca_value, 'black'),
                            s=marker_size, alpha=0.7, edgecolor='black', linewidth=0.5,
                            label=f'{ca_value} CA')
                
                # Axis labels and formatting
                ax.set_xlabel('Bandwidth (MHz)', fontsize=18)
                # ax.set_ylabel('Throughput (Mbps)', fontsize=18)
                # ax.set_title(f'Downlink: Bandwidth vs Throughput by CA - {tech}', fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best')
                
                # Set reasonable limits
                ax.set_ylim(0, 1750)
                ax.set_xlim(0, 240)
                
                plt.tight_layout()
                plt.savefig(f'/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/dl_ca_bw_xput_{tech}_2024_2d.png', dpi=300)
                # plt.savefig(f'/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/dl_ca_bw_xput_{tech}_2024_2d.pdf')
                plt.close()

            # UL
            ul_ca_bw_xput_dict = {}
            for tech in main_tech_xput_ul_ca_bandwidth_dict_3.keys():
                ul_ca_bw_xput_dict[tech] = []
                ul_ca_list = []
                ul_bw_list = []
                ul_xput_list = []
                for ca_bw in main_tech_xput_ul_ca_bandwidth_dict_3[tech].keys():
                    if ca_bw[2] > 0:
                        ca = ca_bw[0]
                        band = ca_bw[1]
                        bw = ca_bw[2]
                        if ca != len(band.split(":")) - 1:
                            continue
                        ul_xput_list.extend(main_tech_xput_ul_ca_bandwidth_dict_3[tech][ca_bw])
                        ul_ca_list.extend([ca] * len(main_tech_xput_ul_ca_bandwidth_dict_3[tech][ca_bw]))
                        ul_bw_list.extend([bw] * len(main_tech_xput_ul_ca_bandwidth_dict_3[tech][ca_bw]))
                ul_ca_bw_xput_dict[tech] = pd.DataFrame({'CA': ul_ca_list, 'BW': ul_bw_list, 'Xput': ul_xput_list})

                df = ul_ca_bw_xput_dict[tech]
                
                # Create 2D scatter plot
                fig, ax = plt.subplots(figsize=(8, 4))
                
                # Get max CA value for size calculation
                max_ca = df['CA'].max()
                
                # Plot each CA value with different markers and sizes
                # Plot in ascending order of CA so higher CA (smaller markers) are plotted on top
                for ca_value in sorted(df['CA'].unique()):
                    ca_data = df[df['CA'] == ca_value]
                    marker_size = get_marker_size(ca_value, max_ca)
                    
                    ax.scatter(ca_data['BW'], ca_data['Xput'], 
                            marker=ca_markers.get(ca_value, 'o'), 
                            color=ca_colors.get(ca_value, 'black'),
                            s=marker_size, alpha=0.7, edgecolor='black', linewidth=0.5,
                            label=f'{ca_value} CA')
                
                # Axis labels and formatting
                ax.set_xlabel('Bandwidth (MHz)', fontsize=12)
                ax.set_ylabel('Throughput (Mbps)', fontsize=12)
                # ax.set_title(f'Uplink: Bandwidth vs Throughput by CA - {tech}', fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best')
                
                # Set reasonable limits
                # Set reasonable limits
                ax.set_ylim(0, 250)
                ax.set_xlim(0, 130)
                
                plt.tight_layout()
                plt.savefig(f'/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ul_ca_bw_xput_{tech}_2024_2d.png', dpi=300)
                plt.close()

        # bw + ca - 2D
        if 1:
            # plot 
            # Define markers for different CA values
            ca_markers = {1: 'o', 2: 's', 3: '^', 4: 'D', 5: 'v', 6: '<', 7: '>', 8: 'p'}
            ca_colors = {1: 'blue', 2: 'red', 3: 'green', 4: 'orange', 5: 'purple', 6: 'brown', 7: 'pink', 8: 'gray'}
            
            # Define marker sizes - same size for all CA values since we're not emphasizing throughput
            marker_size = 60
            
            # DL
            dl_ca_bw_dict = {}
            for tech in main_tech_xput_dl_ca_bandwidth_dict_3.keys():
                dl_ca_bw_dict[tech] = []
                dl_ca_list = []
                dl_bw_list = []
                for ca_bw in main_tech_xput_dl_ca_bandwidth_dict_3[tech].keys():
                    if ca_bw[2] > 0:
                        ca = ca_bw[0]
                        band = ca_bw[1]
                        bw = ca_bw[2]
                        if ca != len(band.split(":")) - 1:
                            continue
                        # Add one entry per unique CA-BW combination (no throughput data needed)
                        num_samples = len(main_tech_xput_dl_ca_bandwidth_dict_3[tech][ca_bw])
                        dl_ca_list.extend([ca] * num_samples)
                        dl_bw_list.extend([bw] * num_samples)
                dl_ca_bw_dict[tech] = pd.DataFrame({'CA': dl_ca_list, 'BW': dl_bw_list})

                df = dl_ca_bw_dict[tech]
                
                # Create 2D scatter plot
                fig, ax = plt.subplots(figsize=(8, 5))
                
                # Plot each CA value with different markers and colors
                for ca_value in sorted(df['CA'].unique()):
                    ca_data = df[df['CA'] == ca_value]
                    
                    ax.scatter(ca_data['BW'], ca_data['CA'], 
                            marker=ca_markers.get(ca_value, 'o'), 
                            color=ca_colors.get(ca_value, 'black'),
                            s=marker_size, alpha=0.7, edgecolor='black', linewidth=0.5,
                            label=f'{ca_value} CA')
                
                # Axis labels and formatting
                ax.set_xlabel('Bandwidth (MHz)', fontsize=22)
                ax.set_ylabel('Carrier Aggregation (CA)', fontsize=22)
                # ax.set_title(f'Downlink: CA vs Bandwidth - {tech}', fontsize=16)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best', fontsize=16)
                
                # Set reasonable limits
                ax.set_ylim(0.5, df['CA'].max() + 0.5)  # Give some padding around CA values
                ax.set_xlim(0, df['BW'].max() + 10)     # Give some padding for bandwidth
                
                # Set y-axis to show integer CA values
                ax.set_yticks(sorted(df['CA'].unique()))
                
                plt.tight_layout()
                plt.savefig(f'/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/dl_ca_vs_bw_{tech}_2024.png', dpi=300)
                plt.savefig(f'/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/dl_ca_vs_bw_{tech}_2024.pdf')
                plt.close()

            # UL
            ul_ca_bw_dict = {}
            for tech in main_tech_xput_ul_ca_bandwidth_dict_3.keys():
                ul_ca_bw_dict[tech] = []
                ul_ca_list = []
                ul_bw_list = []
                for ca_bw in main_tech_xput_ul_ca_bandwidth_dict_3[tech].keys():
                    if ca_bw[2] > 0:
                        ca = ca_bw[0]
                        band = ca_bw[1]
                        bw = ca_bw[2]
                        if ca != len(band.split(":")) - 1:
                            continue
                        # Add one entry per unique CA-BW combination (no throughput data needed)
                        num_samples = len(main_tech_xput_ul_ca_bandwidth_dict_3[tech][ca_bw])
                        ul_ca_list.extend([ca] * num_samples)
                        ul_bw_list.extend([bw] * num_samples)
                ul_ca_bw_dict[tech] = pd.DataFrame({'CA': ul_ca_list, 'BW': ul_bw_list})

                df = ul_ca_bw_dict[tech]
                
                # Create 2D scatter plot
                fig, ax = plt.subplots(figsize=(8, 5))
                
                # Plot each CA value with different markers and colors
                for ca_value in sorted(df['CA'].unique()):
                    ca_data = df[df['CA'] == ca_value]
                    
                    ax.scatter(ca_data['BW'], ca_data['CA'], 
                            marker=ca_markers.get(ca_value, 'o'), 
                            color=ca_colors.get(ca_value, 'black'),
                            s=marker_size, alpha=0.7, edgecolor='black', linewidth=0.5,
                            label=f'{ca_value} CA')
                
                # Axis labels and formatting
                ax.set_xlabel('Bandwidth (MHz)', fontsize=22)
                ax.set_ylabel('Carrier Aggregation (CA)', fontsize=22)
                # ax.set_title(f'Uplink: CA vs Bandwidth - {tech}', fontsize=16)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best', fontsize=16)
                
                # Set reasonable limits
                ax.set_ylim(0.5, df['CA'].max() + 0.5)  # Give some padding around CA values
                ax.set_xlim(0, df['BW'].max() + 10)     # Give some padding for bandwidth
                
                # Set y-axis to show integer CA values
                ax.set_yticks(sorted(df['CA'].unique()))
                
                plt.tight_layout()
                plt.savefig(f'/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ul_ca_vs_bw_{tech}_2024.png', dpi=300)
                plt.savefig(f'/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ul_ca_vs_bw_{tech}_2024.pdf')
                plt.close()


    # carrier aggregation  - 2023
    if 1:
        a = 1
        # Process the data to combine bands with < 5% as "Others"
        def process_bands(category_data):
            processed_data = {}
            for ca, bands in category_data.items():
                processed_data[ca] = {}
                others_total = 0
                
                # First pass: identify bands < 5%
                for band, percentage in bands.items():
                    if percentage < 5:
                        others_total += percentage
                    else:
                        processed_data[ca][band] = percentage
                
                # Add "Others" category if needed
                if others_total > 0:
                    processed_data[ca]["Others"] = others_total
                    
            return processed_data

        # Create a mapping of all unique bands to colors
        def get_band_color_mapping(data):
            all_bands = set()
            for category in data.values():
                for ca_data in category.values():
                    all_bands.update(ca_data.keys())
            
            # Remove "Others" from the set for special handling
            if "Others" in all_bands:
                all_bands.remove("Others")
            
            # Sort bands for consistent ordering
            all_bands = sorted(list(all_bands))
            
            # Create color map using a color spectrum (excluding gray for "Others")
            import matplotlib.cm as cm
            cmap = cm.viridis
            colors = cmap(np.linspace(0, 1, len(all_bands)))

            # Create the mapping
            color_map = {band: colors[i] for i, band in enumerate(all_bands)}
            # Add "Others" with gray color
            color_map["Others"] = (0.7, 0.7, 0.7, 1.0)  # Gray for "Others"
            
            return color_map

        # CA 
        if 1:
            # calculate overall ca distribution 
            dl_ca_overall_distribution = {}
            dl_ca_count = {}
            for tech in main_sa_nsa_lte_tech_ca_band_xput_dict_1.keys():
                dl_ca_overall_distribution[tech] = {}
                dl_ca_count[tech] = {}
                temp = {}
                for ca in main_sa_nsa_lte_tech_ca_band_xput_dict_1[tech].keys():
                    if ca == 0:
                        continue 
                    temp[ca] = 0
                    dl_ca_count[tech][ca] = []
                    for band_combo in main_sa_nsa_lte_tech_ca_band_xput_dict_1[tech][ca].keys():
                        if band_combo.count(':') != ca or 'nan' in band_combo:
                            continue 
                        temp[ca] += len(main_sa_nsa_lte_tech_ca_band_xput_dict_1[tech][ca][band_combo])
                        dl_ca_count[tech][ca].extend([band_combo[:-1]] * len(main_sa_nsa_lte_tech_ca_band_xput_dict_1[tech][ca][band_combo]))
                    
                    dl_ca_count[tech][ca] = calculate_percentage_of_occurrence(dl_ca_count[tech][ca])
                
                temp = {k: (v / sum(temp.values())) * 100 for k, v in temp.items()}
                dl_ca_overall_distribution[tech] = temp.copy()

                ul_ca_overall_distribution = {}
                ul_ca_count = {}
                for tech in main_sa_nsa_lte_tech_ca_ul_band_xput_dict_1.keys():
                    ul_ca_overall_distribution[tech] = {}
                    ul_ca_count[tech] = {}
                    temp = {}
                    for ca in main_sa_nsa_lte_tech_ca_ul_band_xput_dict_1[tech].keys():
                        if ca == 0:
                            continue 
                        temp[ca] = 0
                        ul_ca_count[tech][ca] = []
                        for band_combo in main_sa_nsa_lte_tech_ca_ul_band_xput_dict_1[tech][ca].keys():
                            if band_combo.count(':') != ca or 'nan' in band_combo:
                                continue 
                            temp[ca] += len(main_sa_nsa_lte_tech_ca_ul_band_xput_dict_1[tech][ca][band_combo])
                            ul_ca_count[tech][ca].extend([band_combo[:-1]] * len(main_sa_nsa_lte_tech_ca_ul_band_xput_dict_1[tech][ca][band_combo]))

                        ul_ca_count[tech][ca] = calculate_percentage_of_occurrence(ul_ca_count[tech][ca])

                    temp = {k: (v / sum(temp.values())) * 100 for k, v in temp.items()}
                    ul_ca_overall_distribution[tech] = temp.copy()
                
            # overall CA distribution 
            if 1:
                if 0:
                    # Set up the figure and axis
                    fig, ax = plt.subplots(figsize=(5, 4))
                    # Set width of bars
                    bar_width = 0.2
                    # Positions for the bars
                    positions = np.array([0, 1])

                    # Create a gradient color spectrum (blue to red)
                    import matplotlib.cm as cm
                    # Get all unique CA values to determine color mapping
                    all_ca_values = set()
                    for distribution in [dl_ca_overall_distribution, ul_ca_overall_distribution]:
                        for category in distribution.values():
                            all_ca_values.update(category.keys())
                    num_colors = len(all_ca_values)
                    # Create a color map from a blue-to-red spectrum
                    colors = cm.coolwarm(np.linspace(0, 1, 4))

                    # Hatches for NSA and SA
                    hatches = ['', '\\\\\\']

                    # Process and plot DL data
                    for i, category in enumerate(['NSA', 'SA']):
                        bottom = 0
                        ca_values = dl_ca_overall_distribution[category]
                        # Calculate total for percentage
                        total = sum(ca_values.values())
                        
                        for j, (ca, value) in enumerate(sorted(ca_values.items())):
                            # Convert to percentage
                            percentage = (value / total) * 100
                            
                            bar = ax.bar(positions[0] + (i - 0.5) * bar_width, percentage, bar_width,
                                        bottom=bottom, color=colors[sorted(list(all_ca_values)).index(ca)],
                                        edgecolor='black', hatch=hatches[i])
                            bottom += percentage
                            
                            # Add value label in the middle of each segment
                            # if percentage > 0:
                            #     height = percentage / 2 + bottom - percentage
                            #     ax.text(positions[0] + (i - 0.5) * bar_width, height,
                            #             f'CA {ca}\n{percentage:.1f}%', ha='center', va='center', fontsize=9)

                    # Process and plot UL data
                    for i, category in enumerate(['NSA', 'SA']):
                        bottom = 0
                        ca_values = ul_ca_overall_distribution[category]
                        # Calculate total for percentage
                        total = sum(ca_values.values())
                        
                        for j, (ca, value) in enumerate(sorted(ca_values.items())):
                            # Convert to percentage
                            percentage = (value / total) * 100
                            
                            bar = ax.bar(positions[1] + (i - 0.5) * bar_width, percentage, bar_width,
                                        bottom=bottom, color=colors[sorted(list(all_ca_values)).index(ca)],
                                        edgecolor='black', hatch=hatches[i])
                            bottom += percentage
                            
                            # Add value label in the middle of each segment
                            # if percentage > 0:
                            #     height = percentage / 2 + bottom - percentage
                            #     ax.text(positions[1] + (i - 0.5) * bar_width, height,
                            #             f'CA {ca}\n{percentage:.1f}%', ha='center', va='center', fontsize=9)

                    # Add legend for CA values
                    ca_values = set()
                    for distribution in [dl_ca_overall_distribution, ul_ca_overall_distribution]:
                        for category in distribution.values():
                            ca_values.update(category.keys())

                    # Create custom legend for CA values
                    ca_legend_elements = [plt.Rectangle((0, 0), 1, 1, 
                                                    facecolor=colors[sorted(list(all_ca_values)).index(ca)],
                                                    edgecolor='black', label=f'CA {ca}')
                                        for ca in sorted(ca_values)]

                    # Create custom legend for NSA and SA (with hatches)
                    hatch_legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor='white',
                                                        edgecolor='black', hatch=hatches[i],
                                                        label=category)
                                            for i, category in enumerate(['NSA', 'SA'])]

                    # Add both legends
                    # ax.legend(handles=ca_legend_elements + hatch_legend_elements,
                    #         loc='center', bbox_to_anchor=(1, 1))

                    ax.legend(handles=ca_legend_elements + hatch_legend_elements,
                            loc='best')
                    # Set labels and title
                    ax.set_xticks(positions)
                    ax.set_xticklabels(['DL', 'UL'])
                    ax.set_ylabel('%')  # Changed to % as requested
                    # Title removed as requested

                    # Add a grid for better readability
                    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

                    # Adjust layout
                    plt.tight_layout()

                    # Show the plot
                    plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/sa_nsa_ca_overall_distribution_2023.pdf')
                    plt.close()

                if 1:
                    # Set up the figure and axis
                    fig, ax = plt.subplots(figsize=(5, 4))
                    # Set width of bars
                    bar_width = 0.2
                    # Positions for the bars
                    positions = np.array([0, 1])

                    # Colors for NSA and SA
                    category_colors = {'NSA': 'salmon', 'SA': 'slategrey'}

                    # Get all unique CA values to determine hatch mapping
                    all_ca_values = set()
                    for distribution in [dl_ca_overall_distribution, ul_ca_overall_distribution]:
                        for category in distribution.values():
                            all_ca_values.update(category.keys())

                    # Create hatch patterns for different CA values
                    ca_hatches = ['', '///', '\\\\\\', '|||', '...', '+++', 'xxx', 'ooo']
                    ca_hatch_map = {ca: ca_hatches[i % len(ca_hatches)] for i, ca in enumerate(sorted(all_ca_values))}

                    # Process and plot DL data
                    for i, category in enumerate(['NSA', 'SA']):
                        bottom = 0
                        ca_values = dl_ca_overall_distribution[category]
                        # Calculate total for percentage
                        total = sum(ca_values.values())
                        
                        for j, (ca, value) in enumerate(sorted(ca_values.items())):
                            # Convert to percentage
                            percentage = (value / total) * 100
                            
                            bar = ax.bar(positions[0] + (i - 0.5) * bar_width, percentage, bar_width,
                                        bottom=bottom, color=category_colors[category],
                                        edgecolor='black', hatch=ca_hatch_map[ca])
                            bottom += percentage
                            
                            # Add value label in the middle of each segment
                            # if percentage > 0:
                            #     height = percentage / 2 + bottom - percentage
                            #     ax.text(positions[0] + (i - 0.5) * bar_width, height,
                            #             f'CA {ca}\n{percentage:.1f}%', ha='center', va='center', fontsize=9)

                    # Process and plot UL data
                    for i, category in enumerate(['NSA', 'SA']):
                        bottom = 0
                        ca_values = ul_ca_overall_distribution[category]
                        # Calculate total for percentage
                        total = sum(ca_values.values())
                        
                        for j, (ca, value) in enumerate(sorted(ca_values.items())):
                            # Convert to percentage
                            percentage = (value / total) * 100
                            
                            bar = ax.bar(positions[1] + (i - 0.5) * bar_width, percentage, bar_width,
                                        bottom=bottom, color=category_colors[category],
                                        edgecolor='black', hatch=ca_hatch_map[ca])
                            bottom += percentage
                            
                            # Add value label in the middle of each segment
                            # if percentage > 0:
                            #     height = percentage / 2 + bottom - percentage
                            #     ax.text(positions[1] + (i - 0.5) * bar_width, height,
                            #             f'CA {ca}\n{percentage:.1f}%', ha='center', va='center', fontsize=9)

                    # Add legend for CA values
                    ca_values = set()
                    for distribution in [dl_ca_overall_distribution, ul_ca_overall_distribution]:
                        for category in distribution.values():
                            ca_values.update(category.keys())

                    # Create custom legend for CA values (with hatch patterns)
                    ca_legend_elements = [plt.Rectangle((0, 0), 1, 1, 
                                                    facecolor='white',
                                                    edgecolor='black', 
                                                    hatch=ca_hatch_map[ca],
                                                    label=f'CA {ca}')
                                        for ca in sorted(ca_values)]

                    # Create custom legend for NSA and SA (with colors)
                    category_legend_elements = [plt.Rectangle((0, 0), 1, 1, 
                                                            facecolor=category_colors[category],
                                                            edgecolor='black',
                                                            label=category)
                                            for category in ['NSA', 'SA']]

                    # Add both legends
                    ax.legend(handles=ca_legend_elements + category_legend_elements,
                            loc='best')

                    # Set labels and title
                    ax.set_xticks(positions)
                    ax.set_xticklabels(['DL', 'UL'])
                    ax.set_ylabel('%')

                    # Add a grid for better readability
                    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

                    # Adjust layout
                    plt.tight_layout()

                    # Show the plot
                    plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/sa_nsa_ca_overall_distribution_2023.pdf')
                    plt.close()

            # plot band distribution - DL
            if 0:
                # Process both NSA and SA data
                processed_dl_ca_count = {
                    'NSA': process_bands(dl_ca_count['NSA']),
                    'SA': process_bands(dl_ca_count['SA'])
                }

                # Get color mapping
                band_colors = get_band_color_mapping(processed_dl_ca_count)
                band_colors = {'Others': (0.7, 0.7, 0.7, 1.0),
                'n25':np.array([0.267004, 0.004874, 0.329415, 1.      ]),
                'n25:n41:n41':np.array([0.280267, 0.073417, 0.397163, 1.      ]),
                'n25:n71':np.array([0.282623, 0.140926, 0.457517, 1.      ]),
                'n25:n71:n41':np.array([0.273006, 0.20452 , 0.501721, 1.      ]),
                'n25:n71:n41:n41':np.array([0.253935, 0.265254, 0.529983, 1.      ]),
                'n41':np.array([0.229739, 0.322361, 0.545706, 1.      ]),
                'n41:n25':np.array([0.206756, 0.371758, 0.553117, 1.      ]),
                'n41:n25:n25:n41':np.array([0.183898, 0.422383, 0.556944, 1.      ]),
                'n41:n25:n41':np.array([0.163625, 0.471133, 0.558148, 1.      ]),
                'n41:n25:n71':np.array([0.144759, 0.519093, 0.556572, 1.      ]),
                'n41:n41':np.array([0.127568, 0.566949, 0.550556, 1.      ]),
                'n41:n41:n25':np.array([0.119423, 0.611141, 0.538982, 1.      ]),
                'n41:n41:n71:n25':np.array([0.134692, 0.658636, 0.517649, 1.      ]),
                'n41:n71':np.array([0.185783, 0.704891, 0.485273, 1.      ]),
                'n41:n71:n25':np.array([0.266941, 0.748751, 0.440573, 1.      ]),
                'n41:n71:n25:n41':np.array([0.369214, 0.788888, 0.382914, 1.      ]),
                'n41:n71:n41':np.array([0.477504, 0.821444, 0.318195, 1.      ]),
                'n41:n71:n41:n25':np.array([0.606045, 0.850733, 0.236712, 1.      ]),
                'n71':np.array([0.741388, 0.873449, 0.149561, 1.      ]),
                'n71:n25':np.array([0.876168, 0.891125, 0.09525 , 1.      ]),
                'n71:n41':np.array([0.993248, 0.906157, 0.143936, 1.      ])}
                # Find all unique CA values
                all_cas = set()
                for category in processed_dl_ca_count.values():
                    all_cas.update(category.keys())
                max_ca = max(all_cas) if all_cas else 0

                # Create figure with 2 rows (NSA, SA) and max_ca columns
                fig, axes = plt.subplots(2, max_ca, figsize=(5*max_ca, 8))
                fig.patch.set_edgecolor('black')
                fig.patch.set_linewidth(2)

                # If max_ca is 1, axes will not be a 2D array, so convert to 2D
                if max_ca == 1:
                    axes = np.array(axes).reshape(2, 1)

                # Categories to plot
                categories = ['NSA', 'SA']

                # Plot pie charts
                for row, category in enumerate(categories):
                    for col in range(max_ca):
                        ca = col + 1  # CA values start from 1
                        
                        # If this CA exists for this category
                        if ca in processed_dl_ca_count[category]:
                            ax = axes[row, col]
                            
                            # Get band data for this CA
                            bands_data = processed_dl_ca_count[category][ca]
                            labels = list(bands_data.keys())
                            sizes = list(bands_data.values())
                            
                            # Create label text with percentages
                            # label_texts = [f"{label}\n({size:.1f}%)" for label, size in zip(labels, sizes)]
                            label_texts = [label for label, size in zip(labels, sizes)]
                            
                            # Get colors for each band
                            colors = [band_colors[band] for band in labels]
                            
                            # Use matplotlib's built-in pie chart with external labels
                            wedges, texts = ax.pie(
                                sizes,
                                colors=colors,
                                labels=label_texts,
                                labeldistance=1.1,  # Position labels just outside the pie
                                wedgeprops={'edgecolor': 'w', 'linewidth': 1},
                                textprops={'fontsize': 16, 'ha': 'center'},  # Increased font size from 8 to 10
                                startangle=90,
                                radius=0.8  # Make pie slightly smaller to leave room for labels
                            )
                            
                            # Draw connecting lines from pie to labels
                            for i, wedge in enumerate(wedges):
                                if sizes[i] < 3:  # Skip very small wedges to reduce clutter
                                    texts[i].set_visible(False)
                                    continue
                                
                                # Adjust text alignment for better readability
                                angle = (wedge.theta1 + wedge.theta2) / 2
                                if 90 < angle < 270:  # Left side of the pie
                                    texts[i].set_ha('right')
                                else:  # Right side of the pie
                                    texts[i].set_ha('left')
                            
                            # Modified title format: "CA # (NSA/SA)" and made larger and bold
                            ax.set_title(f"{ca} CA ({category})", fontsize=16, fontweight='bold')
                        else:
                            # If this CA doesn't exist for this category, hide the axes
                            axes[row, col].axis('off')

                # Adjust layout with extra padding for labels
                plt.tight_layout(pad=2.0)

                # Save figure
                plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ca_band_dl_distribution_2023.pdf', 
                            bbox_inches='tight')
                plt.close()

            # plot band distribution - DL - Stacked - hatched
            if 1:
                # Hatch mapping function
                def get_band_hatch_mapping(data):
                    # Get all unique bands
                    all_bands = set()
                    for category in data.values():
                        for ca_data in category.values():
                            all_bands.update(ca_data.keys())
                    
                    # Remove "Others" from the set for special handling
                    if "Others" in all_bands:
                        all_bands.remove("Others")
                    
                    # Sort bands for consistent ordering
                    all_bands = sorted(list(all_bands))
                    
                    # Define base colors for different starting bands
                    base_colors = {
                        'n25': 'lightcoral',
                        'n71': 'cyan', 
                        'n41': 'palegreen',
                        'n260': 'orange',
                        'n261': 'purple',
                        'n77': 'brown',
                        'n78': 'pink',
                        'b2': 'cyan',
                        'b4': 'magenta',
                        'b12': 'yellow',
                    }
                    
                    # Default color for bands that don't match any prefix
                    default_color = 'gray'
                    
                    # Define extensive hatch patterns for complexity levels
                    complexity_hatches = [
                        '',           # No hatch for single bands
                        '///',        # Diagonal lines
                        '\\\\\\',     # Reverse diagonal
                        '|||',        # Vertical lines
                        '---',        # Horizontal lines
                        '+++',        # Plus signs
                        'xxx',        # X marks
                        '...',        # Dots
                        'ooo',        # Circles
                        '***',        # Stars
                        '//o',        # Diagonal with circles
                        '\\\\*',      # Reverse diagonal with stars
                        '||+',        # Vertical with plus
                        '--.',        # Horizontal with dots
                        '//x',        # Diagonal with x
                        '\\\\.',      # Reverse diagonal with dots
                        '++o',        # Plus with circles
                        'x*x',        # X with stars
                        'o-o',        # Circles with horizontal
                        '|*|'         # Vertical with stars
                    ]
                    
                    # Group bands by their primary band (first band in combination)
                    band_groups = {}
                    
                    for band in all_bands:
                        # Get the primary band (first band in combination)
                        if ':' in band:
                            primary_band = band.split(':')[0]
                            complexity = band.count(':') + 1  # Number of bands in combination
                        else:
                            primary_band = band
                            complexity = 1
                        
                        # Find base color for this primary band
                        assigned_color = default_color
                        for prefix, color in base_colors.items():
                            if primary_band.startswith(prefix):
                                assigned_color = color
                                break
                        
                        # Group by primary band
                        if primary_band not in band_groups:
                            band_groups[primary_band] = {
                                'color': assigned_color,
                                'bands_by_complexity': {}
                            }
                        
                        # Group by complexity within each primary band group
                        if complexity not in band_groups[primary_band]['bands_by_complexity']:
                            band_groups[primary_band]['bands_by_complexity'][complexity] = []
                        
                        band_groups[primary_band]['bands_by_complexity'][complexity].append(band)
                    
                    # Create the color and hatch mapping
                    color_map = {}
                    hatch_map = {}
                    
                    for primary_band, group_data in band_groups.items():
                        base_color = group_data['color']
                        complexity_groups = group_data['bands_by_complexity']
                        
                        # Assign colors and hatches based on complexity
                        for complexity, bands in complexity_groups.items():
                            bands = sorted(bands)  # Sort alphabetically within complexity
                            hatch_pattern = complexity_hatches[min(complexity - 1, len(complexity_hatches) - 1)]
                            
                            # If there are multiple bands with same complexity, give them slightly different hatches
                            for i, band in enumerate(bands):
                                color_map[band] = base_color
                                if len(bands) > 1 and i > 0:
                                    # Use a different hatch variation for multiple bands of same complexity
                                    additional_hatch_idx = min(complexity - 1 + i, len(complexity_hatches) - 1)
                                    hatch_map[band] = complexity_hatches[additional_hatch_idx]
                                else:
                                    hatch_map[band] = hatch_pattern
                    
                    # Add "Others" with gray color and no hatch
                    color_map["Others"] = 'lightgray'
                    hatch_map["Others"] = ''
                    
                    return color_map, hatch_map
                

                processed_dl_ca_count = {'NSA': process_bands(dl_ca_count['NSA']), 'SA': process_bands(dl_ca_count['SA'])}

                # Get color and hatch mapping
                # band_colors, band_hatches = get_band_hatch_mapping(processed_dl_ca_count)

                # Find all unique CA values
                all_cas = set()
                for category in processed_dl_ca_count.values():
                    all_cas.update(category.keys())
                max_ca = max(all_cas) if all_cas else 0

                # Find all unique bands across all categories and CA levels
                all_bands_set = set()
                for category_data in processed_dl_ca_count.values():
                    for ca_data in category_data.values():
                        all_bands_set.update(ca_data.keys())

                # Custom sorting function for legend order
                def sort_bands_for_legend(bands):
                    # Separate "Others" for special handling
                    others = [b for b in bands if b == "Others"]
                    regular_bands = [b for b in bands if b != "Others"]
                    
                    # Group bands by their primary band (first band in the combination)
                    band_groups = {}
                    for band in regular_bands:
                        if ':' in band:
                            primary = band.split(':')[0]
                        else:
                            primary = band
                        
                        if primary not in band_groups:
                            band_groups[primary] = []
                        band_groups[primary].append(band)
                    
                    # Sort each group by complexity (number of bands) in ASCENDING order (1, 2, 3 bands)
                    sorted_bands = []
                    for primary in sorted(band_groups.keys()):
                        group_bands = band_groups[primary]
                        # Sort by number of bands (complexity) ascending, then alphabetically
                        group_bands.sort(key=lambda x: (x.count(':'), x))
                        sorted_bands.extend(group_bands)
                    
                    # Add "Others" at the end
                    sorted_bands.extend(others)
                    
                    return sorted_bands

                # Apply custom sorting to all_bands
                all_bands = sort_bands_for_legend(list(all_bands_set))

                # Create figure with stacked bar plot
                fig, ax = plt.subplots(figsize=(5, 6))  # Adjusted height for top legend
                fig.patch.set_edgecolor('black')
                fig.patch.set_linewidth(2)

                # Prepare data for stacked bars
                categories = ['NSA', 'SA']
                x_labels = []
                bar_data = {band: [] for band in all_bands}

                # Build x-axis labels and data
                for category in categories:
                    for ca in range(1, max_ca + 1):
                        x_labels.append(f"{ca} CA\n({category})")
                        
                        # Get data for this CA and category
                        if ca in processed_dl_ca_count[category]:
                            ca_data = processed_dl_ca_count[category][ca]
                            for band in all_bands:
                                bar_data[band].append(ca_data.get(band, 0))
                        else:
                            # If CA doesn't exist for this category, add zeros
                            for band in all_bands:
                                bar_data[band].append(0)

                # Create stacked bars in the sorted order
                x_pos = range(len(x_labels))
                bottom = [0] * len(x_labels)

                for band in all_bands:  # This now uses the sorted order
                    ax.bar(x_pos, bar_data[band], bottom=bottom, 
                        label=band, color=band_colors[band], hatch=band_hatches[band],
                        edgecolor='black', linewidth=0.5)
                    
                    # Update bottom for next stack
                    bottom = [b + v for b, v in zip(bottom, bar_data[band])]

                # Customize the plot
                ax.set_xlabel('CA Level and Technology', fontsize=14, fontweight='bold')
                ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')

                # Set x-axis
                ax.set_xticks(x_pos)
                ax.set_xticklabels(x_labels, fontsize=12)

                # Set y-axis to percentage
                ax.set_ylim(0, 100)

                # Add grid for better readability
                ax.grid(axis='y', alpha=0.3, linestyle='--')

                # Create custom legend with colors and hatches - positioned at top in 2 rows
                # legend_elements = []
                # for band in all_bands:
                #     legend_elements.append(plt.Rectangle((0, 0), 1, 1, 
                #                                     facecolor=band_colors[band], 
                #                                     hatch=band_hatches[band],
                #                                     edgecolor='black',
                #                                     label=band))

                # # Calculate number of columns for 2 rows
                # ncols = max(1, len(all_bands) // 2 + (1 if len(all_bands) % 2 > 0 else 0))

                # # Place legend at the top of the figure in 2 rows
                # fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.99), 
                #         ncol=ncols, fontsize=14, frameon=True)

                # Add value labels on bars (optional - only for segments > 5%)
                for i, x in enumerate(x_pos):
                    cumulative = 0
                    for band in all_bands:
                        value = bar_data[band][i]
                        if value > 5:  # Only show labels for segments > 5%
                            y_pos = cumulative + value/2
                        cumulative += value

                # Adjust layout to make room for the legend at the top
                plt.tight_layout()
                # plt.subplots_adjust(top=0.85)  # Make room for legend at the top

                # Save figure
                plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ca_band_dl_distribution_2023_stacked_hatched.pdf',
                            bbox_inches='tight')
                plt.close()

            # plot band distribution - UL
            if 0:
                # Process both NSA and SA data
                processed_ul_ca_count = {
                    'NSA': process_bands(ul_ca_count['NSA']),
                    'SA': process_bands(ul_ca_count['SA'])
                }

                # Get color mapping
                band_colors = get_band_color_mapping(processed_ul_ca_count)
                band_colors = {'Others': (0.7, 0.7, 0.7, 1.0),
                'n25':np.array([0.267004, 0.004874, 0.329415, 1.      ]),
                'n41':np.array([0.190631, 0.407061, 0.556089, 1.      ]),
                'n41:n25':np.array([0.20803 , 0.718701, 0.472873, 1.      ]),
                'n71':np.array([0.993248, 0.906157, 0.143936, 1.      ])}
                # Find all unique CA values
                all_cas = set()
                for category in processed_ul_ca_count.values():
                    all_cas.update(category.keys())
                max_ca = max(all_cas) if all_cas else 0

                # Create figure with 2 rows (NSA, SA) and max_ca columns
                fig, axes = plt.subplots(2, max_ca, figsize=(5*max_ca, 8))
                fig.patch.set_edgecolor('black')
                fig.patch.set_linewidth(2)

                # If max_ca is 1, axes will not be a 2D array, so convert to 2D
                if max_ca == 1:
                    axes = np.array(axes).reshape(2, 1)

                # Categories to plot
                categories = ['NSA', 'SA']

                # Plot pie charts
                for row, category in enumerate(categories):
                    for col in range(max_ca):
                        ca = col + 1  # CA values start from 1
                        
                        # If this CA exists for this category
                        if ca in processed_ul_ca_count[category]:
                            ax = axes[row, col]
                            
                            # Get band data for this CA
                            bands_data = processed_ul_ca_count[category][ca]
                            labels = list(bands_data.keys())
                            sizes = list(bands_data.values())
                            
                            # Create label text with percentages
                            # label_texts = [f"{label}\n({size:.1f}%)" for label, size in zip(labels, sizes)]
                            label_texts = [label for label, size in zip(labels, sizes)]

                            # Get colors for each band
                            colors = [band_colors[band] for band in labels]
                            
                            # Use matplotlib's built-in pie chart with external labels
                            wedges, texts = ax.pie(
                                sizes,
                                colors=colors,
                                labels=label_texts,
                                labeldistance=1.1,  # Position labels just outside the pie
                                wedgeprops={'edgecolor': 'w', 'linewidth': 1},
                                textprops={'fontsize': 16, 'ha': 'center'},  # Increased font size from 8 to 10
                                startangle=90,
                                radius=0.8  # Make pie slightly smaller to leave room for labels
                            )
                            
                            # Draw connecting lines from pie to labels
                            for i, wedge in enumerate(wedges):
                                if sizes[i] < 3:  # Skip very small wedges to reduce clutter
                                    texts[i].set_visible(False)
                                    continue
                                
                                # Adjust text alignment for better readability
                                angle = (wedge.theta1 + wedge.theta2) / 2
                                if 90 < angle < 270:  # Left side of the pie
                                    texts[i].set_ha('right')
                                else:  # Right side of the pie
                                    texts[i].set_ha('left')
                            
                            # Modified title format: "CA # (NSA/SA)" and made larger and bold
                            ax.set_title(f"{ca} CA ({category})", fontsize=16, fontweight='bold')
                        else:
                            # If this CA doesn't exist for this category, hide the axes
                            axes[row, col].axis('off')

                # Adjust layout with extra padding for labels
                plt.tight_layout(pad=2.0)

                # Save figure
                plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ca_band_ul_distribution_2023.pdf', 
                            bbox_inches='tight')
                plt.close()
                a = 1

            # plot band distribution - UL -- stacked hatched
            if 1:
                def get_band_hatch_mapping(data):
                    # Get all unique bands
                    all_bands = set()
                    for category in data.values():
                        for ca_data in category.values():
                            all_bands.update(ca_data.keys())
                    
                    # Remove "Others" from the set for special handling
                    if "Others" in all_bands:
                        all_bands.remove("Others")
                    
                    # Sort bands for consistent ordering
                    all_bands = sorted(list(all_bands))
                    
                    # Define base colors for different starting bands
                    base_colors = {
                        'n25': 'lightcoral',
                        'n71': 'cyan', 
                        'n41': 'palegreen',
                        'n260': 'orange',
                        'n261': 'purple',
                        'n77': 'brown',
                        'n78': 'pink',
                        'b2': 'cyan',
                        'b4': 'magenta',
                        'b12': 'yellow',
                    }
                    
                    # Default color for bands that don't match any prefix
                    default_color = 'gray'
                    
                    # Define extensive hatch patterns for complexity levels
                    complexity_hatches = [
                        '',           # No hatch for single bands
                        '///',        # Diagonal lines
                        '\\\\\\',     # Reverse diagonal
                        '|||',        # Vertical lines
                        '---',        # Horizontal lines
                        '+++',        # Plus signs
                        'xxx',        # X marks
                        '...',        # Dots
                        'ooo',        # Circles
                        '***',        # Stars
                        '//o',        # Diagonal with circles
                        '\\\\*',      # Reverse diagonal with stars
                        '||+',        # Vertical with plus
                        '--.',        # Horizontal with dots
                        '//x',        # Diagonal with x
                        '\\\\.',      # Reverse diagonal with dots
                        '++o',        # Plus with circles
                        'x*x',        # X with stars
                        'o-o',        # Circles with horizontal
                        '|*|'         # Vertical with stars
                    ]
                    
                    # Group bands by their primary band (first band in combination)
                    band_groups = {}
                    
                    for band in all_bands:
                        # Get the primary band (first band in combination)
                        if ':' in band:
                            primary_band = band.split(':')[0]
                            complexity = band.count(':') + 1  # Number of bands in combination
                        else:
                            primary_band = band
                            complexity = 1
                        
                        # Find base color for this primary band
                        assigned_color = default_color
                        for prefix, color in base_colors.items():
                            if primary_band.startswith(prefix):
                                assigned_color = color
                                break
                        
                        # Group by primary band
                        if primary_band not in band_groups:
                            band_groups[primary_band] = {
                                'color': assigned_color,
                                'bands_by_complexity': {}
                            }
                        
                        # Group by complexity within each primary band group
                        if complexity not in band_groups[primary_band]['bands_by_complexity']:
                            band_groups[primary_band]['bands_by_complexity'][complexity] = []
                        
                        band_groups[primary_band]['bands_by_complexity'][complexity].append(band)
                    
                    # Create the color and hatch mapping
                    color_map = {}
                    hatch_map = {}
                    
                    for primary_band, group_data in band_groups.items():
                        base_color = group_data['color']
                        complexity_groups = group_data['bands_by_complexity']
                        
                        # Assign colors and hatches based on complexity
                        for complexity, bands in complexity_groups.items():
                            bands = sorted(bands)  # Sort alphabetically within complexity
                            hatch_pattern = complexity_hatches[min(complexity - 1, len(complexity_hatches) - 1)]
                            
                            # If there are multiple bands with same complexity, give them slightly different hatches
                            for i, band in enumerate(bands):
                                color_map[band] = base_color
                                if len(bands) > 1 and i > 0:
                                    # Use a different hatch variation for multiple bands of same complexity
                                    additional_hatch_idx = min(complexity - 1 + i, len(complexity_hatches) - 1)
                                    hatch_map[band] = complexity_hatches[additional_hatch_idx]
                                else:
                                    hatch_map[band] = hatch_pattern
                    
                    # Add "Others" with gray color and no hatch
                    color_map["Others"] = 'lightgray'
                    hatch_map["Others"] = ''
                    
                    return color_map, hatch_map
                

                if 1:
                    # Process both NSA and SA data
                    processed_ul_ca_count = {
                        'NSA': process_bands(ul_ca_count['NSA']),
                        'SA': process_bands(ul_ca_count['SA'])
                    }

                    # Get color mapping
                    original_band_colors = band_colors.copy()
                    band_colors = get_band_color_mapping(processed_ul_ca_count)
                    band_colors = {'Others': 'lightgray',
                    'n25': 'lightcoral',
                    'n41': 'palegreen',
                    'n41:n25': 'palegreen',
                    'n71': 'cyan'}

                    # Create simple hatch mapping for the bands in band_colors
                    def create_band_hatches(band_colors):
                        hatch_patterns = ['', '///', '\\\\\\', '|||', '---', '+++', 'xxx', '...', 'ooo', '***']
                        band_hatches = {}
                        
                        # Sort bands by complexity (number of colons)
                        sorted_bands = sorted([band for band in band_colors.keys() if band != 'Others'], 
                                            key=lambda x: (x.count(':'), x))
                        
                        for i, band in enumerate(sorted_bands):
                            band_hatches[band] = hatch_patterns[i % len(hatch_patterns)]
                        
                        # Others gets no hatch
                        band_hatches['Others'] = ''
                        
                        return band_hatches
                    
                    # Get hatch mapping
                    original_band_hatches = band_hatches.copy()
                    band_hatches = create_band_hatches(band_colors)

                    for band, hatch in band_hatches.items():
                        if band in original_band_hatches.keys():
                            band_hatches[band] = original_band_hatches[band]
                    # Find all unique CA values
                    all_cas = set()
                    for category in processed_ul_ca_count.values():
                        all_cas.update(category.keys())
                    max_ca = max(all_cas) if all_cas else 0

                    # Find all unique bands across all categories and CA levels
                    all_bands_set = set()
                    for category_data in processed_ul_ca_count.values():
                        for ca_data in category_data.values():
                            all_bands_set.update(ca_data.keys())
                    
                    # Custom sorting function for legend order
                    def sort_bands_for_legend(bands):
                        # Separate "Others" for special handling
                        others = [b for b in bands if b == "Others"]
                        regular_bands = [b for b in bands if b != "Others"]
                        
                        # Group bands by their primary band (first band in the combination)
                        band_groups = {}
                        for band in regular_bands:
                            if ':' in band:
                                primary = band.split(':')[0]
                            else:
                                primary = band
                            
                            if primary not in band_groups:
                                band_groups[primary] = []
                            band_groups[primary].append(band)
                        
                        # Sort each group by complexity (number of bands) in ASCENDING order (1, 2, 3 bands)
                        sorted_bands = []
                        for primary in sorted(band_groups.keys()):
                            group_bands = band_groups[primary]
                            # Sort by number of bands (complexity) ascending, then alphabetically
                            group_bands.sort(key=lambda x: (x.count(':'), x))
                            sorted_bands.extend(group_bands)
                        
                        # Add "Others" at the end
                        sorted_bands.extend(others)
                        
                        return sorted_bands

                    # Apply custom sorting to all_bands
                    all_bands = sort_bands_for_legend(list(all_bands_set))

                    # Create figure with stacked bar plot
                    fig, ax = plt.subplots(figsize=(4, 6))
                    fig.patch.set_edgecolor('black')
                    fig.patch.set_linewidth(2)

                    # Prepare data for stacked bars
                    categories = ['NSA', 'SA']
                    x_labels = []
                    bar_data = {band: [] for band in all_bands}

                    # Build x-axis labels and data
                    for category in categories:
                        for ca in range(1, max_ca + 1):
                            x_labels.append(f"{ca} CA\n({category})")
                            
                            # Get data for this CA and category
                            if ca in processed_ul_ca_count[category]:
                                ca_data = processed_ul_ca_count[category][ca]
                                for band in all_bands:
                                    bar_data[band].append(ca_data.get(band, 0))
                            else:
                                # If CA doesn't exist for this category, add zeros
                                for band in all_bands:
                                    bar_data[band].append(0)

                    # Create stacked bars in the sorted order
                    x_pos = range(len(x_labels))
                    bottom = [0] * len(x_labels)

                    for band in all_bands:  # This now uses the sorted order
                        ax.bar(x_pos, bar_data[band], bottom=bottom, 
                            label=band, color=band_colors[band], hatch=band_hatches[band],
                            edgecolor='black', linewidth=0.5)
                        
                        # Update bottom for next stack
                        bottom = [b + v for b, v in zip(bottom, bar_data[band])]

                    # Customize the plot
                    ax.set_xlabel('CA Level and Technology', fontsize=14, fontweight='bold')
                    ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')

                    # Set x-axis
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(x_labels, fontsize=12)
                    
                    # Set y-axis to percentage
                    ax.set_ylim(0, 100)
                    
                    # Add grid for better readability
                    ax.grid(axis='y', alpha=0.3, linestyle='--')
                    
                    # Create custom legend with colors and hatches - positioned at top in 2 rows
                    # legend_elements = []
                    # for band in all_bands:
                    #     legend_elements.append(plt.Rectangle((0, 0), 1, 1, 
                    #                                     facecolor=band_colors[band], 
                    #                                     hatch=band_hatches[band],
                    #                                     edgecolor='black',
                    #                                     label=band))
                    
                    # # Calculate number of columns for 2 rows
                    # ncols = max(1, len(all_bands) // 2 + (1 if len(all_bands) % 2 > 0 else 0))
                    
                    # # Place legend at the top of the figure in 2 rows
                    # fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.99), 
                    #         ncol=ncols, fontsize=14, frameon=True)

                    # Add value labels on bars (optional - only for segments > 5%)
                    for i, x in enumerate(x_pos):
                        cumulative = 0
                        for band in all_bands:
                            value = bar_data[band][i]
                            if value > 5:  # Only show labels for segments > 5%
                                y_pos = cumulative + value/2
                            cumulative += value

                    # Adjust layout to make room for the legend at the top
                    plt.tight_layout()
                    # plt.subplots_adjust(top=0.85)  # Make room for legend at the top

                    # Save figure
                    plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ca_band_ul_distribution_2023_stacked_hatched.pdf', 
                                bbox_inches='tight')
                    plt.close()
                    a = 1
                a = 1

        # bw - dl
        if 1:
            dl_bw_overall_distribution = {}
            dl_bw_xput_distribution = {}


            for tech in main_tech_xput_dl_bandwidth_dict_1.keys():
                dl_bw_overall_distribution[tech] = {}
                dl_bw_xput_distribution[tech] = {}
                temp = {}
                for bw in main_tech_xput_dl_bandwidth_dict_1[tech].keys():
                    if bw > 0:
                        temp[bw] = len(main_tech_xput_dl_bandwidth_dict_1[tech][bw])
                        dl_bw_xput_distribution[tech][int(bw)] = main_tech_xput_dl_bandwidth_dict_1[tech][bw].copy()

                temp = {k: (v / sum(temp.values())) * 100 for k, v in temp.items()}
                dl_bw_overall_distribution[tech] = temp.copy()


            # plot DL 
            if 1:
                # Get all unique bandwidth values across both NSA and SA
                all_bandwidths = set()
                for category in dl_bw_xput_distribution.values():
                    all_bandwidths.update(category.keys())
                all_bandwidths = sorted(all_bandwidths)

                # Create a mapping from bandwidth to x-position
                bw_to_pos = {bw: i for i, bw in enumerate(all_bandwidths)}

                # Create figure and subplots (stacked vertically)
                fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True, sharey=True)

                # Plot NSA data (top subplot)
                nsa_data = []
                nsa_positions = []
                nsa_labels = []
                nsa_sample_sizes = []

                # Calculate the total number of NSA samples for percentage calculation
                total_nsa_samples = 0
                for bw in all_bandwidths:
                    if bw in dl_bw_xput_distribution['NSA']:
                        total_nsa_samples += len(dl_bw_xput_distribution['NSA'][bw])

                for bw in all_bandwidths:
                    if bw in dl_bw_xput_distribution['NSA']:
                        nsa_data.append(dl_bw_xput_distribution['NSA'][bw])
                        nsa_positions.append(bw_to_pos[bw])
                        nsa_labels.append(str(bw))
                        # Calculate percentage instead of absolute sample size
                        sample_count = len(dl_bw_xput_distribution['NSA'][bw])
                        percentage = (sample_count / total_nsa_samples) * 100
                        nsa_sample_sizes.append(percentage)

                if nsa_data:  # Check if there's data to plot
                    # Store the boxplot result to access flier positions
                    nsa_boxplot = axes[0].boxplot(nsa_data, positions=nsa_positions, patch_artist=True,
                                boxprops=dict(facecolor='lightblue'))
                    axes[0].set_ylabel('NSA\nThroughput', fontsize=13, fontweight='bold')
                    axes[0].grid(True, linestyle='--', alpha=0.7)
                    
                    # Annotate sample percentages above each box
                    for i, (pos, percentage) in enumerate(zip(nsa_positions, nsa_sample_sizes)):
                        if nsa_data[i]:  # Ensure there's data to plot
                            data_array = np.array(nsa_data[i])
                            
                            # Find the max value, including outliers
                            max_y = np.max(data_array)
                            
                            # Check for outliers/fliers
                            fliers = nsa_boxplot['fliers'][i].get_ydata()
                            if len(fliers) > 0:
                                max_y = max(max_y, np.max(fliers))
                            
                            # Add buffer above the highest point (outlier or whisker)
                            y_buffer = 0.02 * (axes[0].get_ylim()[1] - axes[0].get_ylim()[0])
                            
                            # Add the annotation with percentage
                            axes[0].annotate(f'{percentage:.1f}%', xy=(pos, max_y + y_buffer), 
                                        xytext=(0, 5), textcoords='offset points',
                                        ha='center', va='bottom', fontsize=9, fontweight='bold')

                # Plot SA data (bottom subplot)
                sa_data = []
                sa_positions = []
                sa_labels = []
                sa_sample_sizes = []

                # Calculate the total number of SA samples for percentage calculation
                total_sa_samples = 0
                for bw in all_bandwidths:
                    if bw in dl_bw_xput_distribution['SA']:
                        total_sa_samples += len(dl_bw_xput_distribution['SA'][bw])

                for bw in all_bandwidths:
                    if bw in dl_bw_xput_distribution['SA']:
                        sa_data.append(dl_bw_xput_distribution['SA'][bw])
                        sa_positions.append(bw_to_pos[bw])
                        sa_labels.append(str(bw))
                        # Calculate percentage instead of absolute sample size
                        sample_count = len(dl_bw_xput_distribution['SA'][bw])
                        percentage = (sample_count / total_sa_samples) * 100
                        sa_sample_sizes.append(percentage)

                if sa_data:  # Check if there's data to plot
                    # Store the boxplot result to access flier positions
                    sa_boxplot = axes[1].boxplot(sa_data, positions=sa_positions, patch_artist=True,
                                boxprops=dict(facecolor='lightblue'))
                    axes[1].set_ylabel('SA\nThroughput', fontsize=13, fontweight='bold')
                    axes[1].set_xlabel('Sum Bandwidth (MHz)', fontsize=13, fontweight='bold')
                    axes[1].grid(True, linestyle='--', alpha=0.7)
                    
                    # Annotate sample percentages above each box
                    for i, (pos, percentage) in enumerate(zip(sa_positions, sa_sample_sizes)):
                        if sa_data[i]:  # Ensure there's data to plot
                            data_array = np.array(sa_data[i])
                            
                            # Find the max value, including outliers
                            max_y = np.max(data_array)
                            
                            # Check for outliers/fliers
                            fliers = sa_boxplot['fliers'][i].get_ydata()
                            if len(fliers) > 0:
                                max_y = max(max_y, np.max(fliers))
                            
                            # Add buffer above the highest point (outlier or whisker)
                            y_buffer = 0.02 * (axes[1].get_ylim()[1] - axes[1].get_ylim()[0])
                            
                            # Add the annotation with percentage
                            axes[1].annotate(f'{percentage:.1f}%', xy=(pos, max_y + y_buffer), 
                                        xytext=(0, 5), textcoords='offset points',
                                        ha='center', va='bottom', fontsize=9, fontweight='bold')

                # Set x-ticks and labels to be the same for both plots
                axes[1].set_xticks(range(len(all_bandwidths)))
                axes[1].set_xticklabels([str(bw) for bw in all_bandwidths], rotation=45, ha='right')
                axes[1].set_ylim(0, 1900)
                # Adjust layout
                plt.tight_layout()

                # Save the figure
                plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/bw_dl_xput_distribution_2023.pdf')
                plt.close()

        # bw - ul 
        if 0:
            ul_bw_overall_distribution = {}
            ul_bw_xput_distribution = {}

            for tech in main_tech_xput_ul_bandwidth_dict_1.keys():
                ul_bw_overall_distribution[tech] = {}
                ul_bw_xput_distribution[tech] = {}
                temp = {}
                for bw in main_tech_xput_ul_bandwidth_dict_1[tech].keys():
                    if bw > 0:
                        temp[bw] = len(main_tech_xput_ul_bandwidth_dict_1[tech][bw])
                        ul_bw_xput_distribution[tech][int(bw)] = main_tech_xput_ul_bandwidth_dict_1[tech][bw].copy()

                temp = {k: (v / sum(temp.values())) * 100 for k, v in temp.items()}
                ul_bw_overall_distribution[tech] = temp.copy()

            # plot ul 
            if 1:
                # Get all unique bandwidth values across both NSA and SA
                all_bandwidths = set()
                for category in ul_bw_xput_distribution.values():
                    all_bandwidths.update(category.keys())
                all_bandwidths = sorted(all_bandwidths)

                # Create a mapping from bandwidth to x-position
                bw_to_pos = {bw: i for i, bw in enumerate(all_bandwidths)}

                # Create figure and subplots (stacked vertically)
                fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, sharey=True)

                # Plot NSA data (top subplot)
                nsa_data = []
                nsa_positions = []
                nsa_labels = []
                nsa_sample_sizes = []

                # Calculate total NSA samples for percentage calculation
                total_nsa_samples = 0
                for bw in all_bandwidths:
                    if bw in ul_bw_xput_distribution['NSA']:
                        total_nsa_samples += len(ul_bw_xput_distribution['NSA'][bw])

                for bw in all_bandwidths:
                    if bw in ul_bw_xput_distribution['NSA']:
                        nsa_data.append(ul_bw_xput_distribution['NSA'][bw])
                        nsa_positions.append(bw_to_pos[bw])
                        nsa_labels.append(str(bw))
                        # Calculate percentage instead of absolute count
                        sample_count = len(ul_bw_xput_distribution['NSA'][bw])
                        percentage = (sample_count / total_nsa_samples) * 100
                        nsa_sample_sizes.append(percentage)

                if nsa_data:  # Check if there's data to plot
                    # Store the boxplot result to access flier positions
                    nsa_boxplot = axes[0].boxplot(nsa_data, positions=nsa_positions, patch_artist=True,
                                boxprops=dict(facecolor='lightblue'))
                    axes[0].set_ylabel('NSA\nThroughput')
                    axes[0].grid(True, linestyle='--', alpha=0.7)
                    
                    # Annotate percentages above each box
                    for i, (pos, percentage) in enumerate(zip(nsa_positions, nsa_sample_sizes)):
                        if nsa_data[i]:  # Ensure there's data to plot
                            data_array = np.array(nsa_data[i])
                            
                            # Find the max value, including outliers
                            max_y = np.max(data_array)
                            
                            # Check for outliers/fliers
                            fliers = nsa_boxplot['fliers'][i].get_ydata()
                            if len(fliers) > 0:
                                max_y = max(max_y, np.max(fliers))
                            
                            # Add buffer above the highest point (outlier or whisker)
                            y_buffer = 0.02 * (axes[0].get_ylim()[1] - axes[0].get_ylim()[0])
                            
                            # Add the annotation with percentage format
                            axes[0].annotate(f'{percentage:.1f}%', xy=(pos, max_y + y_buffer), 
                                        xytext=(0, 5), textcoords='offset points',
                                        ha='center', va='bottom', fontsize=9, fontweight='bold')

                # Plot SA data (bottom subplot)
                sa_data = []
                sa_positions = []
                sa_labels = []
                sa_sample_sizes = []

                # Calculate total SA samples for percentage calculation
                total_sa_samples = 0
                for bw in all_bandwidths:
                    if bw in ul_bw_xput_distribution['SA']:
                        total_sa_samples += len(ul_bw_xput_distribution['SA'][bw])

                for bw in all_bandwidths:
                    if bw in ul_bw_xput_distribution['SA']:
                        sa_data.append(ul_bw_xput_distribution['SA'][bw])
                        sa_positions.append(bw_to_pos[bw])
                        sa_labels.append(str(bw))
                        # Calculate percentage instead of absolute count
                        sample_count = len(ul_bw_xput_distribution['SA'][bw])
                        percentage = (sample_count / total_sa_samples) * 100
                        sa_sample_sizes.append(percentage)

                if sa_data:  # Check if there's data to plot
                    # Store the boxplot result to access flier positions
                    sa_boxplot = axes[1].boxplot(sa_data, positions=sa_positions, patch_artist=True,
                                boxprops=dict(facecolor='lightblue'))
                    axes[1].set_ylabel('SA\nThroughput')
                    axes[1].set_xlabel('Sum Bandwidth (MHz)')
                    axes[1].grid(True, linestyle='--', alpha=0.7)
                    
                    # Annotate percentages above each box
                    for i, (pos, percentage) in enumerate(zip(sa_positions, sa_sample_sizes)):
                        if sa_data[i]:  # Ensure there's data to plot
                            data_array = np.array(sa_data[i])
                            
                            # Find the max value, including outliers
                            max_y = np.max(data_array)
                            
                            # Check for outliers/fliers
                            fliers = sa_boxplot['fliers'][i].get_ydata()
                            if len(fliers) > 0:
                                max_y = max(max_y, np.max(fliers))
                            
                            # Add buffer above the highest point (outlier or whisker)
                            y_buffer = 0.02 * (axes[1].get_ylim()[1] - axes[1].get_ylim()[0])
                            
                            # Add the annotation with percentage format
                            axes[1].annotate(f'{percentage:.1f}%', xy=(pos, max_y + y_buffer), 
                                        xytext=(0, 5), textcoords='offset points',
                                        ha='center', va='bottom', fontsize=9, fontweight='bold')

                # Set x-ticks and labels to be the same for both plots
                axes[1].set_xticks(range(len(all_bandwidths)))
                axes[1].set_xticklabels([str(bw) for bw in all_bandwidths], rotation=45, ha='right')
                axes[1].set_ylim(0, 250)
                # Adjust layout
                plt.tight_layout()

                # Save the figure
                plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/bw_ul_xput_distribution_2023.pdf')
                plt.close()
        
        # bw + ca - 3D 
        if 0:
            # plot 
            # Use default settings inside this block
            with mpl.rc_context(rc={}):  # empty rc will reset to default
                # Optionally reinitialize defaults manually
                mpl.rcdefaults()
                # DL
                # main_tech_xput_dl_ca_bandwidth_dict
                dl_ca_bw_xput_dict = {}
                for tech in main_tech_xput_dl_ca_bandwidth_dict_1.keys():
                    dl_ca_bw_xput_dict[tech] = []
                    dl_ca_list = []
                    dl_bw_list = []
                    dl_xput_list = []
                    for ca_bw in main_tech_xput_dl_ca_bandwidth_dict_1[tech].keys():
                        if ca_bw[2] > 0:
                            ca = ca_bw[0]
                            band = ca_bw[1]
                            bw = ca_bw[2]
                            if ca != len(band.split(":")) - 1:
                                continue
                            dl_xput_list.extend(main_tech_xput_dl_ca_bandwidth_dict_1[tech][ca_bw])
                            dl_ca_list.extend([ca] * len(main_tech_xput_dl_ca_bandwidth_dict_1[tech][ca_bw]))
                            dl_bw_list.extend([bw] * len(main_tech_xput_dl_ca_bandwidth_dict_1[tech][ca_bw]))
                    dl_ca_bw_xput_dict[tech] = pd.DataFrame({'CA': dl_ca_list, 'BW': dl_bw_list, 'Xput': dl_xput_list})

                    df = dl_ca_bw_xput_dict[tech]
                    # Plot
                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111, projection='3d')

                    # Color mapping based on throughput
                    sc = ax.scatter(df['BW'], df['CA'], df['Xput'], 
                                    c=df['Xput'], cmap='viridis', s=50)

                    # Axis labels
                    ax.set_xlabel('Bandwidth (BW)')
                    ax.set_ylabel('Carrier Aggregation (CA)')
                    ax.set_yticks([1, 2, 3, 4])
                    ax.set_zlabel('Throughput (Xput)')
                    ax.set_zlim(0, 1600)

                    # Add color bar to show the throughput scale
                    cb = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
                    cb.set_label('Throughput')


                    # Show plot
                    plt.tight_layout()
                    plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/dl_ca_bw_xput_%s_2023.png' %tech, dpi=300)
                    plt.close()


                # ul
                # main_tech_xput_ul_ca_bandwidth_dict
                ul_ca_bw_xput_dict = {}
                for tech in main_tech_xput_ul_ca_bandwidth_dict_1.keys():
                    ul_ca_bw_xput_dict[tech] = []
                    ul_ca_list = []
                    ul_bw_list = []
                    ul_xput_list = []
                    for ca_bw in main_tech_xput_ul_ca_bandwidth_dict_1[tech].keys():
                        if ca_bw[2] > 0:
                            ca = ca_bw[0]
                            band = ca_bw[1]
                            bw = ca_bw[2]
                            if ca != len(band.split(":")) - 1:
                                continue
                            ul_xput_list.extend(main_tech_xput_ul_ca_bandwidth_dict_1[tech][ca_bw])
                            ul_ca_list.extend([ca] * len(main_tech_xput_ul_ca_bandwidth_dict_1[tech][ca_bw]))
                            ul_bw_list.extend([bw] * len(main_tech_xput_ul_ca_bandwidth_dict_1[tech][ca_bw]))
                    ul_ca_bw_xput_dict[tech] = pd.DataFrame({'CA': ul_ca_list, 'BW': ul_bw_list, 'Xput': ul_xput_list})

                    df = ul_ca_bw_xput_dict[tech]
                    # Plot
                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111, projection='3d')

                    # Color mapping based on throughput
                    sc = ax.scatter(df['BW'], df['CA'], df['Xput'], 
                                    c=df['Xput'], cmap='viridis', s=50)

                    # Axis labels
                    ax.set_xlabel('Bandwidth (BW)')
                    ax.set_ylabel('Carrier Aggregation (CA)')
                    ax.set_yticks([1, 2])
                    ax.set_zlabel('Throughput (Xput)')
                    ax.set_zlim(0, 200)

                    # Add color bar to show the throughput scale
                    cb = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
                    cb.set_label('Throughput')


                    # Show plot
                    plt.tight_layout()
                    plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ul_ca_bw_xput_%s_2023.png' %tech, dpi=300)
                    plt.close()

        # bw + ca + xput - 2D
        if 0:
            # plot 
            # Define markers for different CA values
            ca_markers = {1: 'o', 2: 's', 3: '^', 4: 'D', 5: 'v', 6: '<', 7: '>', 8: 'p'}
            ca_colors = {1: 'blue', 2: 'red', 3: 'green', 4: 'orange', 5: 'purple', 6: 'brown', 7: 'pink', 8: 'gray'}
            
            # Define marker sizes - largest for 1-CA, smallest for highest CA
            def get_marker_size(ca_value, max_ca):
                # Start with base size 120 for 1-CA, decrease by 15 for each higher CA
                ca_marker_size_dict = {1: 120, 2: 90, 3: 50, 4: 20}
                # return max(120 - (ca_value - 1) * 15, 30)  # Minimum size of 30
                return ca_marker_size_dict[ca_value]
            
            # DL
            dl_ca_bw_xput_dict = {}
            for tech in main_tech_xput_dl_ca_bandwidth_dict_1.keys():
                dl_ca_bw_xput_dict[tech] = []
                dl_ca_list = []
                dl_bw_list = []
                dl_xput_list = []
                for ca_bw in main_tech_xput_dl_ca_bandwidth_dict_1[tech].keys():
                    if ca_bw[2] > 0:
                        ca = ca_bw[0]
                        band = ca_bw[1]
                        bw = ca_bw[2]
                        if ca != len(band.split(":")) - 1:
                            continue
                        dl_xput_list.extend(main_tech_xput_dl_ca_bandwidth_dict_1[tech][ca_bw])
                        dl_ca_list.extend([ca] * len(main_tech_xput_dl_ca_bandwidth_dict_1[tech][ca_bw]))
                        dl_bw_list.extend([bw] * len(main_tech_xput_dl_ca_bandwidth_dict_1[tech][ca_bw]))
                dl_ca_bw_xput_dict[tech] = pd.DataFrame({'CA': dl_ca_list, 'BW': dl_bw_list, 'Xput': dl_xput_list})

                df = dl_ca_bw_xput_dict[tech]
                
                # Create 2D scatter plot
                fig, ax = plt.subplots(figsize=(6, 3.5))
                
                # Get max CA value for size calculation
                max_ca = df['CA'].max()
                
                # Plot each CA value with different markers and sizes
                # Plot in ascending order of CA so higher CA (smaller markers) are plotted on top
                for ca_value in sorted(df['CA'].unique()):
                    ca_data = df[df['CA'] == ca_value]
                    marker_size = get_marker_size(ca_value, max_ca)
                    
                    ax.scatter(ca_data['BW'], ca_data['Xput'], 
                            marker=ca_markers.get(ca_value, 'o'), 
                            color=ca_colors.get(ca_value, 'black'),
                            s=marker_size, alpha=0.7, edgecolor='black', linewidth=0.5,
                            label=f'{ca_value} CA')
                
                # Axis labels and formatting
                ax.set_xlabel('Bandwidth (MHz)', fontsize=18)
                if tech == 'NSA':
                    ax.set_ylabel('Throughput (Mbps)', fontsize=18)
                # ax.set_title(f'Downlink: Bandwidth vs Throughput by CA - {tech}', fontsize=14)
                ax.grid(True, alpha=0.3)
                # ax.legend(loc='best')
                
                # Set reasonable limits
                ax.set_ylim(0, 1750)
                ax.set_xlim(0, 240)
                
                plt.tight_layout()
                plt.savefig(f'/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/dl_ca_bw_xput_{tech}_2023_2d.png', dpi=300)
                # plt.savefig(f'/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/dl_ca_bw_xput_{tech}_2023_2d.pdf')
                plt.close()

            # UL
            ul_ca_bw_xput_dict = {}
            for tech in main_tech_xput_ul_ca_bandwidth_dict_1.keys():
                ul_ca_bw_xput_dict[tech] = []
                ul_ca_list = []
                ul_bw_list = []
                ul_xput_list = []
                for ca_bw in main_tech_xput_ul_ca_bandwidth_dict_1[tech].keys():
                    if ca_bw[2] > 0:
                        ca = ca_bw[0]
                        band = ca_bw[1]
                        bw = ca_bw[2]
                        if ca != len(band.split(":")) - 1:
                            continue
                        ul_xput_list.extend(main_tech_xput_ul_ca_bandwidth_dict_1[tech][ca_bw])
                        ul_ca_list.extend([ca] * len(main_tech_xput_ul_ca_bandwidth_dict_1[tech][ca_bw]))
                        ul_bw_list.extend([bw] * len(main_tech_xput_ul_ca_bandwidth_dict_1[tech][ca_bw]))
                ul_ca_bw_xput_dict[tech] = pd.DataFrame({'CA': ul_ca_list, 'BW': ul_bw_list, 'Xput': ul_xput_list})

                df = ul_ca_bw_xput_dict[tech]
                
                # Create 2D scatter plot
                fig, ax = plt.subplots(figsize=(8, 4))
                
                # Get max CA value for size calculation
                max_ca = df['CA'].max()
                
                # Plot each CA value with different markers and sizes
                # Plot in ascending order of CA so higher CA (smaller markers) are plotted on top
                for ca_value in sorted(df['CA'].unique()):
                    ca_data = df[df['CA'] == ca_value]
                    marker_size = get_marker_size(ca_value, max_ca)
                    
                    ax.scatter(ca_data['BW'], ca_data['Xput'], 
                            marker=ca_markers.get(ca_value, 'o'), 
                            color=ca_colors.get(ca_value, 'black'),
                            s=marker_size, alpha=0.7, edgecolor='black', linewidth=0.5,
                            label=f'{ca_value} CA')
                
                # Axis labels and formatting
                ax.set_xlabel('Bandwidth (MHz)', fontsize=12)
                ax.set_ylabel('Throughput (Mbps)', fontsize=12)
                # ax.set_title(f'Uplink: Bandwidth vs Throughput by CA - {tech}', fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best')
                
                # Set reasonable limits
                ax.set_ylim(0, 250)
                ax.set_xlim(0, 130)
                
                plt.tight_layout()
                plt.savefig(f'/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ul_ca_bw_xput_{tech}_2023_2d.png', dpi=300)
                plt.close()
    
        # bw + ca - 2D
        if 1:
            # plot 
            # Define markers for different CA values
            ca_markers = {1: 'o', 2: 's', 3: '^', 4: 'D', 5: 'v', 6: '<', 7: '>', 8: 'p'}
            ca_colors = {1: 'blue', 2: 'red', 3: 'green', 4: 'orange', 5: 'purple', 6: 'brown', 7: 'pink', 8: 'gray'}
            
            # Define marker sizes - same size for all CA values since we're not emphasizing throughput
            marker_size = 60
            
            # DL
            dl_ca_bw_dict = {}
            for tech in main_tech_xput_dl_ca_bandwidth_dict_1.keys():
                dl_ca_bw_dict[tech] = []
                dl_ca_list = []
                dl_bw_list = []
                for ca_bw in main_tech_xput_dl_ca_bandwidth_dict_1[tech].keys():
                    if ca_bw[2] > 0:
                        ca = ca_bw[0]
                        band = ca_bw[1]
                        bw = ca_bw[2]
                        if ca != len(band.split(":")) - 1:
                            continue
                        # Add one entry per unique CA-BW combination (no throughput data needed)
                        num_samples = len(main_tech_xput_dl_ca_bandwidth_dict_1[tech][ca_bw])
                        dl_ca_list.extend([ca] * num_samples)
                        dl_bw_list.extend([bw] * num_samples)
                dl_ca_bw_dict[tech] = pd.DataFrame({'CA': dl_ca_list, 'BW': dl_bw_list})

                df = dl_ca_bw_dict[tech]
                
                # Create 2D scatter plot
                fig, ax = plt.subplots(figsize=(8, 5))
                
                # Plot each CA value with different markers and colors
                for ca_value in sorted(df['CA'].unique()):
                    ca_data = df[df['CA'] == ca_value]
                    
                    ax.scatter(ca_data['BW'], ca_data['CA'], 
                            marker=ca_markers.get(ca_value, 'o'), 
                            color=ca_colors.get(ca_value, 'black'),
                            s=marker_size, alpha=0.7, edgecolor='black', linewidth=0.5,
                            label=f'{ca_value} CA')
                
                # Axis labels and formatting
                ax.set_xlabel('Bandwidth (MHz)', fontsize=22)
                ax.set_ylabel('Carrier Aggregation (CA)', fontsize=22)
                # ax.set_title(f'Downlink: CA vs Bandwidth - {tech}', fontsize=16)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best', fontsize=16)
                
                # Set reasonable limits
                ax.set_ylim(0.5, df['CA'].max() + 0.5)  # Give some padding around CA values
                ax.set_xlim(0, df['BW'].max() + 10)     # Give some padding for bandwidth
                
                # Set y-axis to show integer CA values
                ax.set_yticks(sorted(df['CA'].unique()))
                
                plt.tight_layout()
                plt.savefig(f'/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/dl_ca_vs_bw_{tech}_2023.png', dpi=300)
                plt.savefig(f'/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/dl_ca_vs_bw_{tech}_2023.pdf')
                plt.close()

            # UL
            ul_ca_bw_dict = {}
            for tech in main_tech_xput_ul_ca_bandwidth_dict_1.keys():
                ul_ca_bw_dict[tech] = []
                ul_ca_list = []
                ul_bw_list = []
                for ca_bw in main_tech_xput_ul_ca_bandwidth_dict_1[tech].keys():
                    if ca_bw[2] > 0:
                        ca = ca_bw[0]
                        band = ca_bw[1]
                        bw = ca_bw[2]
                        if ca != len(band.split(":")) - 1:
                            continue
                        # Add one entry per unique CA-BW combination (no throughput data needed)
                        num_samples = len(main_tech_xput_ul_ca_bandwidth_dict_1[tech][ca_bw])
                        ul_ca_list.extend([ca] * num_samples)
                        ul_bw_list.extend([bw] * num_samples)
                ul_ca_bw_dict[tech] = pd.DataFrame({'CA': ul_ca_list, 'BW': ul_bw_list})

                df = ul_ca_bw_dict[tech]
                
                # Create 2D scatter plot
                fig, ax = plt.subplots(figsize=(8, 5))
                
                # Plot each CA value with different markers and colors
                for ca_value in sorted(df['CA'].unique()):
                    ca_data = df[df['CA'] == ca_value]
                    
                    ax.scatter(ca_data['BW'], ca_data['CA'], 
                            marker=ca_markers.get(ca_value, 'o'), 
                            color=ca_colors.get(ca_value, 'black'),
                            s=marker_size, alpha=0.7, edgecolor='black', linewidth=0.5,
                            label=f'{ca_value} CA')
                
                # Axis labels and formatting
                ax.set_xlabel('Bandwidth (MHz)', fontsize=22)
                ax.set_ylabel('Carrier Aggregation (CA)', fontsize=22)
                # ax.set_title(f'Uplink: CA vs Bandwidth - {tech}', fontsize=16)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best', fontsize=16)
                
                # Set reasonable limits
                ax.set_ylim(0.5, df['CA'].max() + 0.5)  # Give some padding around CA values
                ax.set_xlim(0, df['BW'].max() + 10)     # Give some padding for bandwidth
                
                # Set y-axis to show integer CA values
                ax.set_yticks(sorted(df['CA'].unique()))
                
                plt.tight_layout()
                plt.savefig(f'/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ul_ca_vs_bw_{tech}_2023.png', dpi=300)
                plt.savefig(f'/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ul_ca_vs_bw_{tech}_2023.pdf')
                plt.close()

    # bw ca combined 
    if 1:
        # Define markers for different years
        year_markers = {2023: 'o', 2024: 'o'}
        year_colors = {2023: 'blue', 2024: 'red'}
        
        # Define markers for different CA values (as secondary differentiation)
        ca_markers = {1: 'o', 2: 's', 3: '^', 4: 'D', 5: 'v', 6: '<', 7: '>', 8: 'p'}
        ca_colors = {1: 'blue', 2: 'red', 3: 'green', 4: 'orange', 5: 'purple', 6: 'brown', 7: 'pink', 8: 'gray'}
        
        marker_size = 300
        
        # Combine 2023 and 2024 data for each tech and direction
        combined_data = {
            'NSA': {'DL': [], 'UL': []},
            'SA': {'DL': [], 'UL': []}
        }
        
        # Process DL data
        for tech in ['NSA', 'SA']:
            # 2023 DL data
            if tech in main_tech_xput_dl_ca_bandwidth_dict_1:
                for ca_bw in main_tech_xput_dl_ca_bandwidth_dict_1[tech].keys():
                    if ca_bw[2] > 0:
                        ca = ca_bw[0]
                        band = ca_bw[1]
                        bw = ca_bw[2]
                        if ca != len(band.split(":")) - 1:
                            continue
                        num_samples = len(main_tech_xput_dl_ca_bandwidth_dict_1[tech][ca_bw])
                        for _ in range(num_samples):
                            combined_data[tech]['DL'].append({'CA': ca, 'BW': bw, 'Year': 2023})
            
            # 2024 DL data
            if tech in main_tech_xput_dl_ca_bandwidth_dict_3:
                for ca_bw in main_tech_xput_dl_ca_bandwidth_dict_3[tech].keys():
                    if ca_bw[2] > 0:
                        ca = ca_bw[0]
                        band = ca_bw[1]
                        bw = ca_bw[2]
                        if ca != len(band.split(":")) - 1:
                            continue
                        num_samples = len(main_tech_xput_dl_ca_bandwidth_dict_3[tech][ca_bw])
                        for _ in range(num_samples):
                            combined_data[tech]['DL'].append({'CA': ca, 'BW': bw, 'Year': 2024})
        
        # Process UL data
        for tech in ['NSA', 'SA']:
            # 2023 UL data
            if tech in main_tech_xput_ul_ca_bandwidth_dict_1:
                for ca_bw in main_tech_xput_ul_ca_bandwidth_dict_1[tech].keys():
                    if ca_bw[2] > 0:
                        ca = ca_bw[0]
                        band = ca_bw[1]
                        bw = ca_bw[2]
                        if ca != len(band.split(":")) - 1:
                            continue
                        num_samples = len(main_tech_xput_ul_ca_bandwidth_dict_1[tech][ca_bw])
                        for _ in range(num_samples):
                            combined_data[tech]['UL'].append({'CA': ca, 'BW': bw, 'Year': 2023})
            
            # 2024 UL data
            if tech in main_tech_xput_ul_ca_bandwidth_dict_3:
                for ca_bw in main_tech_xput_ul_ca_bandwidth_dict_3[tech].keys():
                    if ca_bw[2] > 0:
                        ca = ca_bw[0]
                        band = ca_bw[1]
                        bw = ca_bw[2]
                        if ca != len(band.split(":")) - 1:
                            continue
                        num_samples = len(main_tech_xput_ul_ca_bandwidth_dict_3[tech][ca_bw])
                        for _ in range(num_samples):
                            combined_data[tech]['UL'].append({'CA': ca, 'BW': bw, 'Year': 2024})
        
        # Create separate plots for each tech-direction combination
        plot_configs = [
            ('NSA', 'DL', 'NSA Downlink'),
            ('SA', 'DL', 'SA Downlink'),
            ('NSA', 'UL', 'NSA Uplink'),
            ('SA', 'UL', 'SA Uplink')
        ]
        
        for tech, direction, title in plot_configs:
            if not combined_data[tech][direction]:
                continue
                
            df = pd.DataFrame(combined_data[tech][direction])
            
            # Create scatter plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot data points by year and CA
            for year in [2023, 2024]:
                year_data = df[df['Year'] == year]
                if year_data.empty:
                    continue
                    
                for ca_value in sorted(year_data['CA'].unique()):
                    ca_data = year_data[year_data['CA'] == ca_value]
                    
                    ax.scatter(list(set(ca_data['BW'])), list(set(ca_data['CA'])) * len(list(set(ca_data['BW']))), 
                            marker=year_markers[year], 
                            color=ca_colors.get(ca_value, 'black'),
                            s=marker_size, alpha=1, edgecolor='black', linewidth=7,
                            label=f'{ca_value} CA ({year})' if len(df['Year'].unique()) > 1 else f'{ca_value} CA')
            
            # Axis labels and formatting
            ax.set_xlabel('Bandwidth (MHz)', fontsize=34)
            # ax.set_ylabel('Carrier\nAggregation (CA)', fontsize=34)
            ax.set_ylabel('CA', fontsize=34)
            # ax.set_title(f'{title}', fontsize=18)
            ax.grid(True, alpha=0.3)
            
            # Create custom legend
            legend_elements = []
            
            # Add CA legends
            for ca_value in sorted(df['CA'].unique()):
                legend_elements.append(plt.scatter([], [], marker='o', 
                                                color=ca_colors.get(ca_value, 'black'),
                                                s=marker_size, alpha=1, edgecolor='black',
                                                label=f'{ca_value} CA', linewidth=7))
            
            # # Add year legends if both years are present
            # if len(df['Year'].unique()) > 1:
            #     for year in sorted(df['Year'].unique()):
            #         legend_elements.append(plt.scatter([], [], marker=year_markers[year], 
            #                                          color='gray', s=marker_size, alpha=0.7, 
            #                                          edgecolor='black', label=f'{year}'))
            
            ax.legend(handles=legend_elements, loc='best', fontsize=30)
            
            # Set reasonable limits
            ax.set_ylim(0.5, df['CA'].max() + 0.5)
            ax.set_xlim(0, df['BW'].max() + 10)
            
            # Set y-axis to show integer CA values
            ax.set_yticks(sorted(df['CA'].unique()))
            ax.tick_params(axis='both', labelsize=25)
            
            plt.tight_layout()
            plt.savefig(f'/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/{direction.lower()}_ca_vs_bw_{tech}_combined.png', dpi=300)
            plt.savefig(f'/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/{direction.lower()}_ca_vs_bw_{tech}_combined.pdf')
            plt.close()
            a = 1


    # ping
    if 0:
        color_dict = {'NSA' : 'salmon', 'SA' : 'slategrey', 'LTE' : 'green', 'LTE (DC)' : 'lime'}

        fig, ax = plt.subplots(figsize=(5, 4))
        for tech in main_xput_tech_ping_dict_1.keys():
            if tech == 'Skip' or 'gsm' in tech.lower() or 'lte' in tech.lower() or 'service' in tech.lower():
                continue
            data = main_xput_tech_ping_dict_1[tech].copy()
            sorted_data = np.sort(data)
            ax.plot(sorted_data, np.linspace(0, 1, sorted_data.size), label=tech + " (2023)", color=color_dict[tech], ls='-')

            data = main_xput_tech_ping_dict_3[tech].copy()
            sorted_data = np.sort(data)
            ax.plot(sorted_data, np.linspace(0, 1, sorted_data.size), label=tech + " (2024)", color=color_dict[tech], ls='--')

        ax.set_ylabel('CDF')
        ax.set_xlabel('RTT (ms)')
        ax.legend(fontsize=13)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.set_xlim(xmin=0, xmax=200)
        plt.tight_layout()
        plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ping_sa_nsa_xput_overall.pdf')
        plt.close()

        fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        band_color_dict = {'n41' : 'red', 'n71' : 'black', 'n25' : 'royalblue', 'n90' : 'green'}
        for tech in main_tech_ping_band_dict_1.keys():
            if tech == 'NSA':
                ls = '-'
            else:
                ls = '--'
            sum_samples = [len(main_tech_ping_band_dict_1[tech][band]) for band in main_tech_ping_band_dict_1[tech]]
            sum_samples = sum(sum_samples)
            for band in main_tech_ping_band_dict_1[tech].keys():
                if tech == 'NSA' and band == 'n90':
                    continue
                sorted_data = np.sort(main_tech_ping_band_dict_1[tech][band])
                ax[0].plot(sorted_data, np.linspace(0, 1, sorted_data.size), color=band_color_dict[band], ls=ls, label="%s %s (%s" %(tech, band, str(round((len(sorted_data) / sum_samples) * 100))) + "%)")
                # ax[0].plot(sorted_data, np.linspace(0, 1, sorted_data.size), color=band_color_dict[band], ls=ls, label="%s (%s)" %(tech, band))
        
        ax[0].legend(loc='best', fontsize=12)
        ax[0].set_ylabel("CDF")
        ax[0].set_xlabel("RTT (ms)")
        ax[0].set_xlim(0, 200)

        for tech in main_tech_ping_band_dict_3.keys():
            if tech == 'NSA':
                ls = '-'
            else:
                ls = '--'
            sum_samples = [len(main_tech_ping_band_dict_3[tech][band]) for band in main_tech_ping_band_dict_3[tech]]
            sum_samples = sum(sum_samples)
            for band in main_tech_ping_band_dict_3[tech].keys():
                sorted_data = np.sort(main_tech_ping_band_dict_3[tech][band])
                if tech == 'SA' and band == 'n25':
                    ax[1].plot(sorted_data, np.linspace(0, 1, sorted_data.size), color=band_color_dict[band], ls=ls, label="%s %s - (%s" %(tech, band, "16")+ "%)")
                else:
                    ax[1].plot(sorted_data, np.linspace(0, 1, sorted_data.size), color=band_color_dict[band], ls=ls, label="%s %s - (%s" %(tech, band, str(round((len(sorted_data) / sum_samples) * 100))) + "%)")
                # ax[1].plot(sorted_data, np.linspace(0, 1, sorted_data.size), color=band_color_dict[band], ls=ls, label="%s (%s)" %(tech, band))
        
        ax[1].legend(loc='best', fontsize=12)
        ax[1].set_xlabel("RTT (ms)")
        ax[1].set_xlim(0, 200)

        ax[0].set_title("2023", fontweight='bold', fontsize=15)
        ax[1].set_title("2024", fontweight='bold', fontsize=15)
        ax[0].grid(True)
        ax[1].grid(True)
        ax[0].set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ping_sa_nsa_band_yearwise.pdf')
        plt.close()


        fig, ax = plt.subplots(figsize=(10, 4), sharey=True)
        band_color_dict = {'n41' : 'red', 'n71' : 'black', 'n25' : 'royalblue', 'n90' : 'green'}
        for tech in main_tech_ping_band_dict_1.keys():
            if tech == 'NSA':
                ls = '-'
            else:
                ls = '--'
            band_list = list(main_tech_ping_band_dict_1[tech].keys())
            band_list.extend(list(main_tech_ping_band_dict_3[tech].keys()))
            band_list = list(set(band_list))
            for band in band_list:
                data = []
                if band in main_tech_ping_band_dict_1[tech].keys():
                    data.extend(main_tech_ping_band_dict_1[tech][band])
                if band in main_tech_ping_band_dict_3[tech].keys():
                    data.extend(main_tech_ping_band_dict_3[tech][band])   

                sorted_data = np.sort(data)
                # ax.plot(sorted_data, np.linspace(0, 1, sorted_data.size), color=band_color_dict[band], ls=ls, label="%s (%s) %s" %(tech, band, str(len(sorted_data))))
                ax.plot(sorted_data, np.linspace(0, 1, sorted_data.size), color=band_color_dict[band], ls=ls, label="%s (%s) %s" %(tech, band, str(len(sorted_data))))
        
        ax.legend(loc='best', fontsize=16)
        ax.set_ylabel("CDF")
        ax.set_xlabel("RTT (ms)")
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 1)
        ax.grid(True)
        plt.tight_layout()
        plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ping_sa_nsa_band.pdf')
        plt.close()

        fig, ax = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
        city_color_dict = {'big-city' : 'salmon', 'not-big-city': 'black', 'unclassified' : 'slategrey'}
        for tech in main_tech_ping_city_info_dict_1.keys():
            if tech == 'NSA':
                ls = '-'
            else:
                ls = '--'
            for city in main_tech_ping_city_info_dict_1[tech].keys():
                sorted_data = np.sort(main_tech_ping_city_info_dict_1[tech][city])
                ax[0].plot(sorted_data, np.linspace(0, 1, sorted_data.size), color=city_color_dict[city], ls=ls, label="%s (%s) %s" %(tech, city, str(len(sorted_data))))
        
        ax[0].legend(loc='best')
        ax[0].set_ylabel("CDF")
        ax[0].set_xlabel("RTT (ms)")
        ax[0].set_xlim(0, 150)

        for tech in main_tech_ping_city_info_dict_3.keys():
            if tech == 'NSA':
                ls = '-'
            else:
                ls = '--'
            for city in main_tech_ping_city_info_dict_3[tech].keys():
                sorted_data = np.sort(main_tech_ping_city_info_dict_3[tech][city])
                ax[1].plot(sorted_data, np.linspace(0, 1, sorted_data.size), color=city_color_dict[city], ls=ls, label="%s (%s) %s" %(tech, city, str(len(sorted_data))))
        
        ax[1].legend(loc='best')
        ax[1].set_xlabel("RTT (ms)")
        ax[1].set_xlim(0, 150)

        ax[0].set_title("2023", fontweight='bold', fontsize=15)
        ax[1].set_title("2024", fontweight='bold', fontsize=15)
        ax[0].set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ping_sa_nsa_city_yearwise.pdf')
        plt.close()

    # power
    if 0:
        fig, ax = plt.subplots(1, 4, figsize=(13, 6), sharey=True)
        col_dict ={'5G (NSA)' : 'salmon', '5G (SA)' : 'slategrey'}
        for tech in main_sa_nsa_tx_power_dict_1.keys():
            tx_power = main_sa_nsa_tx_power_dict_1[tech].copy()
            tx_power = [float(i) for i in tx_power if not pd.isnull(i) and type(i) != str]
            tx_power = np.sort(tx_power)

            tx_power_control = main_sa_nsa_tx_power_control_dict_1[tech].copy()
            tx_power_control = [float(i) for i in tx_power_control if not pd.isnull(i) and type(i) != str]
            tx_power_control = np.sort(tx_power_control)
            
            rx_power = main_sa_nsa_rx_power_dict_1[tech].copy()
            rx_power = [float(i) for i in rx_power if not pd.isnull(i)]
            rx_power = np.sort(rx_power)

            pathloss = main_sa_nsa_pathloss_dict_1[tech].copy()
            pathloss = [float(i) for i in pathloss if not pd.isnull(i)]
            pathloss = np.sort(pathloss)

            ax[0].plot(tx_power, np.linspace(0, 1, tx_power.size), label=tech                , color=col_dict[tech])
            ax[1].plot(tx_power_control, np.linspace(0, 1, tx_power_control.size), label=tech, color=col_dict[tech])
            ax[2].plot(rx_power, np.linspace(0, 1, rx_power.size), label=tech                , color=col_dict[tech])
            ax[3].plot(pathloss, np.linspace(0, 1, pathloss.size), label=tech                , color=col_dict[tech])
        
        ax[0].set_ylabel("CDF")
        ax[0].set_xlabel("PUSCH Power (dBm)")
        ax[1].set_xlabel("PUCCH Power (dBm)")
        ax[2].set_xlabel("RSRP (dBm)")
        ax[3].set_xlabel("Pathloss (dB)")
        ax[0].set_ylim(0, 1)
        ax[0].legend(loc='best')
        ax[1].legend(loc='best')
        ax[2].legend(loc='best')
        ax[3].legend(loc='best')
        plt.tight_layout()
        #plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/tx_rx_power_all_2023.pdf')
        plt.close()

        fig, ax = plt.subplots(1, 4, figsize=(13, 6), sharey=True)
        for tech in main_sa_nsa_tx_power_dict_3.keys():
            tx_power = main_sa_nsa_tx_power_dict_3[tech].copy()
            tx_power = [float(i) for i in tx_power if not pd.isnull(i) and type(i) != str]
            tx_power = np.sort(tx_power)

            tx_power_control = main_sa_nsa_tx_power_control_dict_3[tech].copy()
            tx_power_control = [float(i) for i in tx_power_control if not pd.isnull(i) and type(i) != str]
            tx_power_control = np.sort(tx_power_control)
            
            rx_power = main_sa_nsa_rx_power_dict_3[tech].copy()
            rx_power = [float(i) for i in rx_power if not pd.isnull(i)]
            rx_power = np.sort(rx_power)

            pathloss = main_sa_nsa_pathloss_dict_3[tech].copy()
            pathloss = [float(i) for i in pathloss if not pd.isnull(i)]
            pathloss = np.sort(pathloss)

            ax[0].plot(tx_power, np.linspace(0, 1, tx_power.size), label=tech                , color=col_dict[tech])
            ax[1].plot(tx_power_control, np.linspace(0, 1, tx_power_control.size), label=tech, color=col_dict[tech])
            ax[2].plot(rx_power, np.linspace(0, 1, rx_power.size), label=tech                , color=col_dict[tech])
            ax[3].plot(pathloss, np.linspace(0, 1, pathloss.size), label=tech                , color=col_dict[tech])
        
        ax[0].set_ylabel("CDF")
        ax[0].set_xlabel("PUSCH Power (dBm)")
        ax[1].set_xlabel("PUCCH Power (dBm)")
        ax[2].set_xlabel("RSRP (dBm)")
        ax[3].set_xlabel("Pathloss (dB)")
        ax[0].set_ylim(0, 1)
        ax[0].legend(loc='best')
        ax[1].legend(loc='best')
        ax[2].legend(loc='best')
        ax[3].legend(loc='best')
        plt.tight_layout()
        #plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/tx_rx_power_all_2024.pdf')
        plt.close()

        fig, ax = plt.subplots(1, 4, figsize=(17, 6), sharey=True)
        for tech in main_tech_xput_dl_tx_power_dict_1.keys():
            if 'NSA' in tech:
                ls = '-'
            else:
                ls = '--'
            tx_power_dl = main_tech_xput_dl_tx_power_dict_1[tech].copy()
            tx_power_dl = [i for i in tx_power_dl if not pd.isnull(i)]
            tx_power_dl = np.sort(tx_power_dl)

            tx_power_control_dl = main_tech_xput_dl_tx_power_control_dict_1[tech].copy()
            tx_power_control_dl = [i for i in tx_power_control_dl if not pd.isnull(i)]
            tx_power_control_dl = np.sort(tx_power_control_dl)

            rx_power_dl = main_tech_xput_dl_rx_power_dict_1[tech].copy()
            rx_power_dl = [i for i in rx_power_dl if not pd.isnull(i)]
            rx_power_dl = np.sort(rx_power_dl)

            pathloss_dl = main_tech_xput_dl_pathloss_dict_1[tech].copy()
            pathloss_dl = [i for i in pathloss_dl if not pd.isnull(i)]
            pathloss_dl = np.sort(pathloss_dl)

            tx_power_ul = main_tech_xput_ul_tx_power_dict_1[tech].copy()
            tx_power_ul = [i for i in tx_power_ul if not pd.isnull(i)]
            tx_power_ul = np.sort(tx_power_ul)

            tx_power_control_ul = main_tech_xput_ul_tx_power_control_dict_1[tech].copy()
            tx_power_control_ul = [i for i in tx_power_control_ul if not pd.isnull(i)]
            tx_power_control_ul = np.sort(tx_power_control_ul)

            rx_power_ul = main_tech_xput_ul_rx_power_dict_1[tech].copy()
            rx_power_ul = [i for i in rx_power_ul if not pd.isnull(i)]
            rx_power_ul = np.sort(rx_power_ul)

            pathloss_ul = main_tech_xput_ul_pathloss_dict_1[tech].copy()
            pathloss_ul = [i for i in pathloss_ul if not pd.isnull(i)]
            pathloss_ul = np.sort(pathloss_ul)

            tx_power_ping = main_ping_tx_power_dict_1[tech].copy()
            tx_power_ping = [i for i in tx_power_ping if not pd.isnull(i)]
            tx_power_ping = np.sort(tx_power_ping)

            tx_power_control_ping = main_ping_tx_power_control_dict_1[tech].copy()
            tx_power_control_ping = [i for i in tx_power_control_ping if not pd.isnull(i)]
            tx_power_control_ping = np.sort(tx_power_control_ping)

            rx_power_ping = main_tech_ping_rsrp_dict_1[tech].copy()
            rx_power_ping = [i for i in rx_power_ping if not pd.isnull(i)]
            rx_power_ping = np.sort(rx_power_ping)

            pathloss_ping = main_tech_ping_pathloss_dict_1[tech].copy()
            pathloss_ping = [i for i in pathloss_ping if not pd.isnull(i)]
            pathloss_ping = np.sort(pathloss_ping)

            ax[0].plot(tx_power_dl, np.linspace(0, 1, tx_power_dl.size), label=tech + " (DL)", color='red', ls=ls)
            ax[1].plot(tx_power_control_dl, np.linspace(0, 1, tx_power_control_dl.size), label=tech + " (DL)", color='red', ls=ls)
            ax[2].plot(rx_power_dl, np.linspace(0, 1, rx_power_dl.size), label=tech + " (DL)", color='red', ls=ls)
            ax[3].plot(pathloss_dl, np.linspace(0, 1, pathloss_dl.size), label=tech + " (DL)", color='red', ls=ls)
            ax[0].plot(tx_power_ul, np.linspace(0, 1, tx_power_ul.size), label=tech + "(UL)",  color='black',     ls=ls)
            ax[1].plot(tx_power_control_ul, np.linspace(0, 1, tx_power_control_ul.size), label=tech + " (UL)", color='black',     ls=ls)
            ax[2].plot(rx_power_ul, np.linspace(0, 1, rx_power_ul.size), label=tech + " (UL)", color='black',     ls=ls)
            ax[3].plot(pathloss_ul, np.linspace(0, 1, pathloss_ul.size), label=tech + " (UL)", color='black',     ls=ls)
            ax[0].plot(tx_power_ping, np.linspace(0, 1, tx_power_ping.size), label=tech + "(RTT)",  color='lime',     ls=ls)
            ax[1].plot(tx_power_control_ping, np.linspace(0, 1, tx_power_control_ping.size), label=tech + " (RTT)", color='lime',     ls=ls)
            ax[2].plot(rx_power_ping, np.linspace(0, 1, rx_power_ping.size), label=tech + " (RTT)", color='lime',     ls=ls)
            ax[3].plot(pathloss_ping, np.linspace(0, 1, pathloss_ping.size), label=tech + " (RTT)", color='lime',     ls=ls)

        ax[0].set_ylabel("CDF")
        ax[0].set_xlabel("PUSCH Power (dBm)")
        ax[1].set_xlabel("PUCCH Power (dBm)")
        ax[2].set_xlabel("RSRP (dBm)")
        ax[3].set_xlabel("Pathloss (dB)")
        ax[0].set_ylim(0, 1)
        ax[0].legend(loc='best')
        plt.tight_layout()
        #plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/tx_rx_power_break_2023.pdf')
        plt.close()
        a = 1


        fig, ax = plt.subplots(1, 4, figsize=(17, 6), sharey=True)
        for tech in main_tech_xput_dl_tx_power_dict_3.keys():
            if 'NSA' in tech:
                ls = '-'
            else:
                ls = '--'
            tx_power_dl = main_tech_xput_dl_tx_power_dict_3[tech].copy()
            tx_power_dl = [i for i in tx_power_dl if not pd.isnull(i)]
            tx_power_dl = np.sort(tx_power_dl)

            tx_power_control_dl = main_tech_xput_dl_tx_power_control_dict_3[tech].copy()
            tx_power_control_dl = [i for i in tx_power_control_dl if not pd.isnull(i)]
            tx_power_control_dl = np.sort(tx_power_control_dl)

            rx_power_dl = main_tech_xput_dl_rx_power_dict_3[tech].copy()
            rx_power_dl = [i for i in rx_power_dl if not pd.isnull(i)]
            rx_power_dl = np.sort(rx_power_dl)

            pathloss_dl = main_tech_xput_dl_pathloss_dict_3[tech].copy()
            pathloss_dl = [i for i in pathloss_dl if not pd.isnull(i)]
            pathloss_dl = np.sort(pathloss_dl)

            tx_power_ul = main_tech_xput_ul_tx_power_dict_3[tech].copy()
            tx_power_ul = [i for i in tx_power_ul if not pd.isnull(i)]
            tx_power_ul = np.sort(tx_power_ul)

            tx_power_control_ul = main_tech_xput_ul_tx_power_control_dict_3[tech].copy()
            tx_power_control_ul = [i for i in tx_power_control_ul if not pd.isnull(i)]
            tx_power_control_ul = np.sort(tx_power_control_ul)

            rx_power_ul = main_tech_xput_ul_rx_power_dict_3[tech].copy()
            rx_power_ul = [i for i in rx_power_ul if not pd.isnull(i)]
            rx_power_ul = np.sort(rx_power_ul)

            pathloss_ul = main_tech_xput_ul_pathloss_dict_3[tech].copy()
            pathloss_ul = [i for i in pathloss_ul if not pd.isnull(i)]
            pathloss_ul = np.sort(pathloss_ul)

            tx_power_ping = main_ping_tx_power_dict_3[tech].copy()
            tx_power_ping = [i for i in tx_power_ping if not pd.isnull(i)]
            tx_power_ping = np.sort(tx_power_ping)

            tx_power_control_ping = main_ping_tx_power_control_dict_3[tech].copy()
            tx_power_control_ping = [i for i in tx_power_control_ping if not pd.isnull(i)]
            tx_power_control_ping = np.sort(tx_power_control_ping)

            rx_power_ping = main_tech_ping_rsrp_dict_3[tech].copy()
            rx_power_ping = [i for i in rx_power_ping if not pd.isnull(i)]
            rx_power_ping = np.sort(rx_power_ping)

            pathloss_ping = main_tech_ping_pathloss_dict_3[tech].copy()
            pathloss_ping = [i for i in pathloss_ping if not pd.isnull(i)]
            pathloss_ping = np.sort(pathloss_ping)


            ax[0].plot(tx_power_dl, np.linspace(0, 1, tx_power_dl.size), label=tech + " (DL)", color='red', ls=ls)
            ax[1].plot(tx_power_control_dl, np.linspace(0, 1, tx_power_control_dl.size), label=tech + " (DL)", color='red', ls=ls)
            ax[2].plot(rx_power_dl, np.linspace(0, 1, rx_power_dl.size), label=tech + " (DL)", color='red', ls=ls)
            ax[3].plot(pathloss_dl, np.linspace(0, 1, pathloss_dl.size), label=tech + " (DL)", color='red', ls=ls)
            ax[0].plot(tx_power_ul, np.linspace(0, 1, tx_power_ul.size), label=tech + "(UL)",  color='black',     ls=ls)
            ax[1].plot(tx_power_control_ul, np.linspace(0, 1, tx_power_control_ul.size), label=tech + " (UL)", color='black',     ls=ls)
            ax[2].plot(rx_power_ul, np.linspace(0, 1, rx_power_ul.size), label=tech + " (UL)", color='black',     ls=ls)
            ax[3].plot(pathloss_ul, np.linspace(0, 1, pathloss_ul.size), label=tech + " (UL)", color='black',     ls=ls)
            ax[0].plot(tx_power_ping, np.linspace(0, 1, tx_power_ping.size), label=tech + "(RTT)",  color='lime',     ls=ls)
            ax[1].plot(tx_power_control_ping, np.linspace(0, 1, tx_power_control_ping.size), label=tech + " (RTT)", color='lime',     ls=ls)
            ax[2].plot(rx_power_ping, np.linspace(0, 1, rx_power_ping.size), label=tech + " (RTT)", color='lime',     ls=ls)
            ax[3].plot(pathloss_ping, np.linspace(0, 1, pathloss_ping.size), label=tech + " (RTT)", color='lime',     ls=ls)

        ax[0].set_ylabel("CDF")
        ax[0].set_xlabel("PUSCH Power (dBm)")
        ax[1].set_xlabel("PUCCH Power (dBm)")
        ax[2].set_xlabel("RSRP (dBm)")
        ax[3].set_xlabel("Pathloss (dB)")
        ax[0].set_ylim(0, 1)
        ax[0].legend(loc='best')
        plt.tight_layout()
        #plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/tx_rx_power_break_2024.pdf')
        plt.close()
    else:
        fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
        col_dict ={'5G (NSA)' : 'salmon', '5G (SA)' : 'slategrey'}
        for tech in main_sa_nsa_tx_power_dict_1.keys():
            tx_power = main_sa_nsa_tx_power_dict_1[tech].copy()
            tx_power = [float(i) for i in tx_power if not pd.isnull(i) and type(i) != str]
            tx_power = np.sort(tx_power)

            tx_power_control = main_sa_nsa_tx_power_control_dict_1[tech].copy()
            tx_power_control = [float(i) for i in tx_power_control if not pd.isnull(i) and type(i) != str]
            tx_power_control = np.sort(tx_power_control)
            
            rx_power = main_sa_nsa_rx_power_dict_1[tech].copy()
            rx_power = [float(i) for i in rx_power if not pd.isnull(i)]
            rx_power = np.sort(rx_power)

            pathloss = main_sa_nsa_pathloss_dict_1[tech].copy()
            pathloss = [float(i) for i in pathloss if not pd.isnull(i)]
            pathloss = np.sort(pathloss)

            ax[0].plot(tx_power, np.linspace(0, 1, tx_power.size), label=tech + " (2023)"               , color=col_dict[tech] )
            ax[1].plot(tx_power_control, np.linspace(0, 1, tx_power_control.size), label=tech + " (2023)", color=col_dict[tech])
            ax[2].plot(rx_power, np.linspace(0, 1, rx_power.size), label=tech        + " (2023)"          , color=col_dict[tech])
            # ax[3].plot(pathloss, np.linspace(0, 1, pathloss.size), label=tech                , color=col_dict[tech])

        for tech in main_sa_nsa_tx_power_dict_3.keys():
            tx_power = main_sa_nsa_tx_power_dict_3[tech].copy()
            tx_power = [float(i) for i in tx_power if not pd.isnull(i) and type(i) != str]
            tx_power = np.sort(tx_power)

            tx_power_control = main_sa_nsa_tx_power_control_dict_3[tech].copy()
            tx_power_control = [float(i) for i in tx_power_control if not pd.isnull(i) and type(i) != str]
            tx_power_control = np.sort(tx_power_control)
            
            rx_power = main_sa_nsa_rx_power_dict_3[tech].copy()
            rx_power = [float(i) for i in rx_power if not pd.isnull(i)]
            rx_power = np.sort(rx_power)

            pathloss = main_sa_nsa_pathloss_dict_3[tech].copy()
            pathloss = [float(i) for i in pathloss if not pd.isnull(i)]
            pathloss = np.sort(pathloss)

            ax[0].plot(tx_power, np.linspace(0, 1, tx_power.size), label=tech + " (2024)"               , color=col_dict[tech] , ls='--')
            ax[1].plot(tx_power_control, np.linspace(0, 1, tx_power_control.size), label=tech + " (2024)", color=col_dict[tech], ls='--')
            ax[2].plot(rx_power, np.linspace(0, 1, rx_power.size), label=tech   + " (2024)"                , color=col_dict[tech], ls='--')
            # ax[3].plot(pathloss, np.linspace(0, 1, pathloss.size), label=tech                , color=col_dict[tech])


        ax[0].set_ylabel("CDF")
        ax[0].set_xlabel("PUSCH Power (dBm)", fontsize=15)
        ax[1].set_xlabel("PUCCH Power (dBm)", fontsize=15)
        ax[2].set_xlabel("RSRP (dBm)")
        # ax[3].set_xlabel("Pathloss (dB)")
        ax[0].set_ylim(0, 1)
        ax[0].legend(loc='best', fontsize=11)
        # ax[1].legend(loc='best')
        # ax[2].legend(loc='best')
        # ax[3].legend(loc='best')
        ax[0].set_xlim(-40, 28)
        ax[1].set_xlim(-40, 28)
        ax[2].set_xlim(-110, -60)
        ax[0].grid(True)
        ax[1].grid(True)
        ax[2].grid(True)
        plt.tight_layout()
        #plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/tx_power_all.pdf')
        plt.close()



        fig, ax = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
        for tech in main_tech_xput_dl_tx_power_dict_1.keys():
            if 'NSA' in tech:
                ls = '-'
            else:
                ls = '--'
            tx_power_dl = main_tech_xput_dl_tx_power_dict_1[tech].copy()
            tx_power_dl = [i for i in tx_power_dl if not pd.isnull(i)]
            tx_power_dl = np.sort(tx_power_dl)

            tx_power_control_dl = main_tech_xput_dl_tx_power_control_dict_1[tech].copy()
            tx_power_control_dl = [i for i in tx_power_control_dl if not pd.isnull(i)]
            tx_power_control_dl = np.sort(tx_power_control_dl)

            rx_power_dl = main_tech_xput_dl_rx_power_dict_1[tech].copy()
            rx_power_dl = [i for i in rx_power_dl if not pd.isnull(i)]
            rx_power_dl = np.sort(rx_power_dl)

            pathloss_dl = main_tech_xput_dl_pathloss_dict_1[tech].copy()
            pathloss_dl = [i for i in pathloss_dl if not pd.isnull(i)]
            pathloss_dl = np.sort(pathloss_dl)

            tx_power_ul = main_tech_xput_ul_tx_power_dict_1[tech].copy()
            tx_power_ul = [i for i in tx_power_ul if not pd.isnull(i)]
            tx_power_ul = np.sort(tx_power_ul)

            tx_power_control_ul = main_tech_xput_ul_tx_power_control_dict_1[tech].copy()
            tx_power_control_ul = [i for i in tx_power_control_ul if not pd.isnull(i)]
            tx_power_control_ul = np.sort(tx_power_control_ul)

            rx_power_ul = main_tech_xput_ul_rx_power_dict_1[tech].copy()
            rx_power_ul = [i for i in rx_power_ul if not pd.isnull(i)]
            rx_power_ul = np.sort(rx_power_ul)

            pathloss_ul = main_tech_xput_ul_pathloss_dict_1[tech].copy()
            pathloss_ul = [i for i in pathloss_ul if not pd.isnull(i)]
            pathloss_ul = np.sort(pathloss_ul)

            tx_power_ping = main_ping_tx_power_dict_1[tech].copy()
            tx_power_ping = [i for i in tx_power_ping if not pd.isnull(i)]
            tx_power_ping = np.sort(tx_power_ping)

            tx_power_control_ping = main_ping_tx_power_control_dict_1[tech].copy()
            tx_power_control_ping = [i for i in tx_power_control_ping if not pd.isnull(i)]
            tx_power_control_ping = np.sort(tx_power_control_ping)

            rx_power_ping = main_tech_ping_rsrp_dict_1[tech].copy()
            rx_power_ping = [i for i in rx_power_ping if not pd.isnull(i)]
            rx_power_ping = np.sort(rx_power_ping)

            pathloss_ping = main_tech_ping_pathloss_dict_1[tech].copy()
            pathloss_ping = [i for i in pathloss_ping if not pd.isnull(i)]
            pathloss_ping = np.sort(pathloss_ping)

            ax[0].plot(tx_power_dl, np.linspace(0, 1, tx_power_dl.size), label=tech + " (DL)", color='salmon', ls=ls)
            ax[1].plot(tx_power_control_dl, np.linspace(0, 1, tx_power_control_dl.size), label=tech + " (DL)", color='salmon', ls=ls)
            ax[2].plot(rx_power_dl, np.linspace(0, 1, rx_power_dl.size), label=tech + " (DL)", color='salmon', ls=ls)
            # ax[3].plot(pathloss_dl, np.linspace(0, 1, pathloss_dl.size), label=tech + " (DL)", color='red', ls=ls)
            ax[0].plot(tx_power_ul, np.linspace(0, 1, tx_power_ul.size), label=tech + " (UL)",  color='black',     ls=ls)
            ax[1].plot(tx_power_control_ul, np.linspace(0, 1, tx_power_control_ul.size), label=tech + " (UL)", color='black',     ls=ls)
            ax[2].plot(rx_power_ul, np.linspace(0, 1, rx_power_ul.size), label=tech + " (UL)", color='black',     ls=ls)
            # ax[3].plot(pathloss_ul, np.linspace(0, 1, pathloss_ul.size), label=tech + " (UL)", color='black',     ls=ls)
            ax[0].plot(tx_power_ping, np.linspace(0, 1, tx_power_ping.size), label=tech + " (RTT)",  color='gray',     ls=ls)
            ax[1].plot(tx_power_control_ping, np.linspace(0, 1, tx_power_control_ping.size), label=tech + " (RTT)", color='gray',     ls=ls)
            ax[2].plot(rx_power_ping, np.linspace(0, 1, rx_power_ping.size), label=tech + " (RTT)", color='gray',     ls=ls)
            # ax[3].plot(pathloss_ping, np.linspace(0, 1, pathloss_ping.size), label=tech + " (RTT)", color='lime',     ls=ls)

        ax[0].set_ylabel("CDF")
        ax[0].set_xlabel("PUSCH Power (dBm)")
        ax[1].set_xlabel("PUCCH Power (dBm)")
        ax[2].set_xlabel("RSRP (dBm)")
        # ax[3].set_xlabel("Pathloss (dB)")
        ax[0].set_ylim(0, 1)
        ax[0].grid(True)
        ax[1].grid(True)
        ax[2].grid(True)
        ax[0].set_xlim(-40, 28)
        ax[1].set_xlim(-40, 28)
        ax[2].set_xlim(-110, -60)
        plt.tight_layout()
        #plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/tx_power_break_2023.pdf')
        plt.close()
        a = 1


        fig, ax = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
        for tech in main_tech_xput_dl_tx_power_dict_3.keys():
            if 'NSA' in tech:
                ls = '-'
            else:
                ls = '--'
            tx_power_dl = main_tech_xput_dl_tx_power_dict_3[tech].copy()
            tx_power_dl = [i for i in tx_power_dl if not pd.isnull(i)]
            tx_power_dl = np.sort(tx_power_dl)

            tx_power_control_dl = main_tech_xput_dl_tx_power_control_dict_3[tech].copy()
            tx_power_control_dl = [i for i in tx_power_control_dl if not pd.isnull(i)]
            tx_power_control_dl = np.sort(tx_power_control_dl)

            rx_power_dl = main_tech_xput_dl_rx_power_dict_3[tech].copy()
            rx_power_dl = [i for i in rx_power_dl if not pd.isnull(i)]
            rx_power_dl = np.sort(rx_power_dl)

            pathloss_dl = main_tech_xput_dl_pathloss_dict_3[tech].copy()
            pathloss_dl = [i for i in pathloss_dl if not pd.isnull(i)]
            pathloss_dl = np.sort(pathloss_dl)

            tx_power_ul = main_tech_xput_ul_tx_power_dict_3[tech].copy()
            tx_power_ul = [i for i in tx_power_ul if not pd.isnull(i)]
            tx_power_ul = np.sort(tx_power_ul)

            tx_power_control_ul = main_tech_xput_ul_tx_power_control_dict_3[tech].copy()
            tx_power_control_ul = [i for i in tx_power_control_ul if not pd.isnull(i)]
            tx_power_control_ul = np.sort(tx_power_control_ul)

            rx_power_ul = main_tech_xput_ul_rx_power_dict_3[tech].copy()
            rx_power_ul = [i for i in rx_power_ul if not pd.isnull(i)]
            rx_power_ul = np.sort(rx_power_ul)

            pathloss_ul = main_tech_xput_ul_pathloss_dict_3[tech].copy()
            pathloss_ul = [i for i in pathloss_ul if not pd.isnull(i)]
            pathloss_ul = np.sort(pathloss_ul)

            tx_power_ping = main_ping_tx_power_dict_3[tech].copy()
            tx_power_ping = [i for i in tx_power_ping if not pd.isnull(i)]
            tx_power_ping = np.sort(tx_power_ping)

            tx_power_control_ping = main_ping_tx_power_control_dict_3[tech].copy()
            tx_power_control_ping = [i for i in tx_power_control_ping if not pd.isnull(i)]
            tx_power_control_ping = np.sort(tx_power_control_ping)

            rx_power_ping = main_tech_ping_rsrp_dict_3[tech].copy()
            rx_power_ping = [i for i in rx_power_ping if not pd.isnull(i)]
            rx_power_ping = np.sort(rx_power_ping)

            pathloss_ping = main_tech_ping_pathloss_dict_3[tech].copy()
            pathloss_ping = [i for i in pathloss_ping if not pd.isnull(i)]
            pathloss_ping = np.sort(pathloss_ping)


            ax[0].plot(tx_power_dl, np.linspace(0, 1, tx_power_dl.size), label=tech + " (DL)", color='salmon', ls=ls)
            ax[1].plot(tx_power_control_dl, np.linspace(0, 1, tx_power_control_dl.size), label=tech + " (DL)", color='salmon', ls=ls)
            ax[2].plot(rx_power_dl, np.linspace(0, 1, rx_power_dl.size), label=tech + " (DL)", color='salmon', ls=ls)
            # ax[3].plot(pathloss_dl, np.linspace(0, 1, pathloss_dl.size), label=tech + " (DL)", color='red', ls=ls)
            ax[0].plot(tx_power_ul, np.linspace(0, 1, tx_power_ul.size), label=tech + " (UL)",  color='black',     ls=ls)
            ax[1].plot(tx_power_control_ul, np.linspace(0, 1, tx_power_control_ul.size), label=tech + " (UL)", color='black',     ls=ls)
            ax[2].plot(rx_power_ul, np.linspace(0, 1, rx_power_ul.size), label=tech + " (UL)", color='black',     ls=ls)
            # ax[3].plot(pathloss_ul, np.linspace(0, 1, pathloss_ul.size), label=tech + " (UL)", color='black',     ls=ls)
            ax[0].plot(tx_power_ping, np.linspace(0, 1, tx_power_ping.size), label=tech + "(RTT)",  color='gray',     ls=ls)
            ax[1].plot(tx_power_control_ping, np.linspace(0, 1, tx_power_control_ping.size), label=tech + " (RTT)", color='gray',     ls=ls)
            ax[2].plot(rx_power_ping, np.linspace(0, 1, rx_power_ping.size), label=tech + " (RTT)", color='gray',     ls=ls)
            # ax[3].plot(pathloss_ping, np.linspace(0, 1, pathloss_ping.size), label=tech + " (RTT)", color='lime',     ls=ls)

        ax[0].set_ylabel("CDF")
        ax[0].set_xlabel("PUSCH Power (dBm)")
        ax[1].set_xlabel("PUCCH Power (dBm)")
        ax[2].set_xlabel("RSRP (dBm)")
        # ax[3].set_xlabel("Pathloss (dB)")
        ax[0].set_ylim(0, 1)
        ax[0].grid(True)
        ax[0].legend(loc='best', fontsize=13)
        ax[1].grid(True)
        ax[2].grid(True)
        ax[0].set_xlim(-40, 28)
        ax[1].set_xlim(-40, 28)
        ax[2].set_xlim(-110, -60)
        # ax[0].legend(loc='best')
        plt.tight_layout()
        #plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/tx_power_break_2024.pdf')
        plt.close()

        fig, ax = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
        col_dict ={'5G (NSA)' : 'salmon', '5G (SA)' : 'slategrey'}
        for tech in main_sa_nsa_tx_power_dict_1.keys():
            tx_power = main_sa_nsa_tx_power_dict_1[tech].copy()
            tx_power = [float(i) for i in tx_power if not pd.isnull(i) and type(i) != str]

            tx_power_control = main_sa_nsa_tx_power_control_dict_1[tech].copy()
            tx_power_control = [float(i) for i in tx_power_control if not pd.isnull(i) and type(i) != str]
            
            rx_power = main_sa_nsa_rx_power_dict_1[tech].copy()
            rx_power = [float(i) for i in rx_power if not pd.isnull(i)]

            pathloss = main_sa_nsa_pathloss_dict_1[tech].copy()
            pathloss = [float(i) for i in pathloss if not pd.isnull(i)]
            

            tx_power.extend(main_sa_nsa_tx_power_dict_3[tech])
            tx_power = [float(i) for i in tx_power if not pd.isnull(i) and type(i) != str]
            tx_power = np.sort(tx_power)

            tx_power_control.extend(main_sa_nsa_tx_power_control_dict_3[tech])
            tx_power_control = [float(i) for i in tx_power_control if not pd.isnull(i) and type(i) != str]
            tx_power_control = np.sort(tx_power_control)
            
            rx_power.extend(main_sa_nsa_rx_power_dict_3[tech])
            rx_power = [float(i) for i in rx_power if not pd.isnull(i)]
            rx_power = np.sort(rx_power)

            pathloss.extend(main_sa_nsa_pathloss_dict_3[tech])
            pathloss = [float(i) for i in pathloss if not pd.isnull(i)]
            pathloss = np.sort(pathloss)

            ax[0].plot(tx_power, np.linspace(0, 1, tx_power.size), label=tech          , color=col_dict[tech] , ls='-')
            ax[1].plot(tx_power_control, np.linspace(0, 1, tx_power_control.size), label=tech, color=col_dict[tech], ls='-')
            # ax[2].plot(rx_power, np.linspace(0, 1, rx_power.size), label=tech   , color=col_dict[tech], ls='-')
            # ax[3].plot(pathloss, np.linspace(0, 1, pathloss.size), label=tech                , color=col_dict[tech])


        ax[0].set_ylabel("CDF")
        ax[0].set_xlabel("PUSCH Power (dBm)", fontsize=15)
        ax[1].set_xlabel("PUCCH Power (dBm)", fontsize=15)
        # ax[2].set_xlabel("RSRP (dBm)", fontsize=15)
        # ax[3].set_xlabel("Pathloss (dB)", fontsize=15)
        ax[0].set_ylim(0, 1)
        ax[0].legend(loc='best', fontsize=13)
        # ax[1].legend(loc='best')
        # ax[2].legend(loc='best')
        # ax[3].legend(loc='best')
        ax[0].set_xlim(-10, 28)
        ax[1].set_xlim(-40, 28)
        # ax[2].set_xlim(-110, -60)
        # ax[3].set_xlim(80, 150)
        ax[0].grid(True)
        ax[1].grid(True)
        # ax[2].grid(True)
        # ax[3].grid(True)
        plt.tight_layout()
        #plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/tx_power_mixed.pdf')
        plt.close()

        fig, ax = plt.subplots(figsize=(3, 3))
        col_dict ={'5G (NSA)' : 'salmon', '5G (SA)' : 'slategrey'}
        for tech in main_sa_nsa_tx_power_dict_1.keys():
            pathloss = main_sa_nsa_pathloss_dict_1[tech].copy()
            pathloss = [float(i) for i in pathloss if not pd.isnull(i)]
            pathloss.extend(main_sa_nsa_pathloss_dict_3[tech])
            pathloss = [float(i) for i in pathloss if not pd.isnull(i)]
            pathloss = np.sort(pathloss)
            ax.plot(pathloss, np.linspace(0, 1, pathloss.size), label=tech, color=col_dict[tech], ls='-')

        ax.set_ylabel("CDF")
        ax.set_xlabel("Pathloss (dB)", fontsize=15)
        ax.set_ylim(0, 1)
        ax.set_xlim(90, 140)
        ax.grid(True)
        plt.tight_layout()
        #plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/pathloss_mixed.pdf')
        plt.close()

        fig, ax = plt.subplots(figsize=(4, 3))
        for tech in main_tech_xput_dl_tx_power_dict_1.keys():
            if 'NSA' in tech:
                ls = '-'
            else:
                ls = '--'
            tx_power_dl = main_tech_xput_dl_tx_power_dict_1[tech].copy()
            tx_power_dl = [i for i in tx_power_dl if not pd.isnull(i)]

            tx_power_control_dl = main_tech_xput_dl_tx_power_control_dict_1[tech].copy()
            tx_power_control_dl = [i for i in tx_power_control_dl if not pd.isnull(i)]

            rx_power_dl = main_tech_xput_dl_rx_power_dict_1[tech].copy()
            rx_power_dl = [i for i in rx_power_dl if not pd.isnull(i)]


            tx_power_ul = main_tech_xput_ul_tx_power_dict_1[tech].copy()
            tx_power_ul = [i for i in tx_power_ul if not pd.isnull(i)]

            tx_power_control_ul = main_tech_xput_ul_tx_power_control_dict_1[tech].copy()
            tx_power_control_ul = [i for i in tx_power_control_ul if not pd.isnull(i)]

            rx_power_ul = main_tech_xput_ul_rx_power_dict_1[tech].copy()
            rx_power_ul = [i for i in rx_power_ul if not pd.isnull(i)]

            tx_power_ping = main_ping_tx_power_dict_1[tech].copy()
            tx_power_ping = [i for i in tx_power_ping if not pd.isnull(i)]

            tx_power_control_ping = main_ping_tx_power_control_dict_1[tech].copy()
            tx_power_control_ping = [i for i in tx_power_control_ping if not pd.isnull(i)]

            rx_power_ping = main_tech_ping_rsrp_dict_1[tech].copy()
            rx_power_ping = [i for i in rx_power_ping if not pd.isnull(i)]


            tx_power_dl.extend(main_tech_xput_dl_tx_power_dict_3[tech])
            tx_power_dl = [i for i in tx_power_dl if not pd.isnull(i)]
            tx_power_dl = np.sort(tx_power_dl)

            tx_power_control_dl.extend(main_tech_xput_dl_tx_power_control_dict_3[tech])
            tx_power_control_dl = [i for i in tx_power_control_dl if not pd.isnull(i)]
            tx_power_control_dl = np.sort(tx_power_control_dl)

            rx_power_dl.extend(main_tech_xput_dl_rx_power_dict_3[tech])
            rx_power_dl = [i for i in rx_power_dl if not pd.isnull(i)]
            rx_power_dl = np.sort(rx_power_dl)

            tx_power_ul.extend(main_tech_xput_ul_tx_power_dict_3[tech])
            tx_power_ul = [i for i in tx_power_ul if not pd.isnull(i)]
            tx_power_ul = np.sort(tx_power_ul)

            tx_power_control_ul.extend(main_tech_xput_ul_tx_power_control_dict_3[tech])
            tx_power_control_ul = [i for i in tx_power_control_ul if not pd.isnull(i)]
            tx_power_control_ul = np.sort(tx_power_control_ul)

            rx_power_ul.extend(main_tech_xput_ul_rx_power_dict_3[tech])
            rx_power_ul = [i for i in rx_power_ul if not pd.isnull(i)]
            rx_power_ul = np.sort(rx_power_ul)

            tx_power_ping.extend(main_ping_tx_power_dict_3[tech])
            tx_power_ping = [i for i in tx_power_ping if not pd.isnull(i)]
            tx_power_ping = np.sort(tx_power_ping)

            tx_power_control_ping.extend(main_ping_tx_power_control_dict_3[tech])
            tx_power_control_ping = [i for i in tx_power_control_ping if not pd.isnull(i)]
            tx_power_control_ping = np.sort(tx_power_control_ping)

            rx_power_ping.extend(main_tech_ping_rsrp_dict_3[tech])
            rx_power_ping = [i for i in rx_power_ping if not pd.isnull(i)]
            rx_power_ping = np.sort(rx_power_ping)

            pathloss_dl = main_tech_xput_dl_pathloss_dict_1[tech].copy()
            pathloss_dl.extend(main_tech_xput_dl_pathloss_dict_3[tech])
            pathloss_dl = [i for i in pathloss_dl if not pd.isnull(i)]
            pathloss_dl = np.sort(pathloss_dl)

            pathloss_ul = main_tech_xput_ul_pathloss_dict_1[tech].copy()
            pathloss_ul.extend(main_tech_xput_ul_pathloss_dict_3[tech])
            pathloss_ul = [i for i in pathloss_ul if not pd.isnull(i)]
            pathloss_ul = np.sort(pathloss_ul)

            pathloss_ping = main_tech_ping_pathloss_dict_1[tech].copy()
            pathloss_ping.extend(main_tech_ping_pathloss_dict_3[tech])
            pathloss_ping = [i for i in pathloss_ping if not pd.isnull(i)]
            pathloss_ping = np.sort(pathloss_ping)


            ax.plot(tx_power_dl, np.linspace(0, 1, tx_power_dl.size), label=tech + " (DL)", color='salmon', ls=ls)
            # ax[1].plot(tx_power_control_dl, np.linspace(0, 1, tx_power_control_dl.size), label=tech + " (DL)", color='salmon', ls=ls)
            # ax[2].plot(rx_power_dl, np.linspace(0, 1, rx_power_dl.size), label=tech + " (DL)", color='salmon', ls=ls)
            # ax[3].plot(pathloss_dl, np.linspace(0, 1, pathloss_dl.size), label=tech + " (DL)", color='salmon', ls=ls)
            ax.plot(tx_power_ul, np.linspace(0, 1, tx_power_ul.size), label=tech + " (UL)",  color='black',     ls=ls)
            # ax[1].plot(tx_power_control_ul, np.linspace(0, 1, tx_power_control_ul.size), label=tech + " (UL)", color='black',     ls=ls)
            # ax[2].plot(rx_power_ul, np.linspace(0, 1, rx_power_ul.size), label=tech + " (UL)", color='black',     ls=ls)
            # ax[3].plot(pathloss_ul, np.linspace(0, 1, pathloss_ul.size), label=tech + " (UL)", color='black',     ls=ls)
            ax.plot(tx_power_ping, np.linspace(0, 1, tx_power_ping.size), label=tech + "(RTT)",  color='gray',     ls=ls)
            # ax[1].plot(tx_power_control_ping, np.linspace(0, 1, tx_power_control_ping.size), label=tech + " (RTT)", color='gray',     ls=ls)
            # ax[2].plot(rx_power_ping, np.linspace(0, 1, rx_power_ping.size), label=tech + " (RTT)", color='gray',     ls=ls)
            # ax[3].plot(pathloss_ping, np.linspace(0, 1, pathloss_ping.size), label=tech + " (RTT)", color='gray',     ls=ls)

        ax.set_ylabel("CDF")
        ax.set_xlabel("PUSCH Power (dBm)")
        # ax[1].set_xlabel("PUCCH Power (dBm)")
        # ax[2].set_xlabel("RSRP (dBm)")
        # ax[3].set_xlabel("Pathloss (dB)")
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(loc='best', fontsize=13)
        # ax[1].grid(True)
        # ax[2].grid(True)
        # ax[3].grid(True)
        ax.set_xlim(-15, 28)
        # ax[1].set_xlim(-40, 28)
        # ax[2].set_xlim(-110, -60)
        # ax[3].set_xlim(80, 150)
        # ax[0].legend(loc='best')
        plt.tight_layout()
        #plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/tx_power_break_mixed.pdf')
        plt.close()
    

    # ho 2023 vs 2024 overall 
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(np.sort(nsa_nr_nr_duration_list), np.linspace(0, 1, np.sort(nsa_nr_nr_duration_list).size), label="NSA (2023)", color='salmon')
    ax.plot(np.sort(nsa_nr_nr_duration_list_3), np.linspace(0, 1, np.sort(nsa_nr_nr_duration_list_3).size), label="NSA (2024)", color='salmon', ls='--')
    ax.plot(np.sort(sa_duration_list), np.linspace(0, 1, np.sort(sa_duration_list).size), label="SA (2023)", color='slategrey')
    ax.plot(np.sort(sa_duration_list_3), np.linspace(0, 1, np.sort(sa_duration_list_3).size), label="SA (2024)", color='slategrey', ls='--')
    
    ax.set_ylabel("CDF")
    ax.set_xlabel("Handover duration (s)")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 0.1)
    ax.grid(True)
    ax.legend(loc='best', fontsize=14)
    plt.tight_layout()
    plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ho_duration_overall.pdf')
    plt.close()

    fig, ax = plt.subplots(figsize=(3, 4))
    ax.plot(np.sort(nsa_nr_nr_time_diff_after_lte_ho), np.linspace(0, 1, np.sort(nsa_nr_nr_time_diff_after_lte_ho).size), label="2023", color='black')
    ax.plot(np.sort(nsa_nr_nr_time_diff_after_lte_ho_3), np.linspace(0, 1, np.sort(nsa_nr_nr_time_diff_after_lte_ho_3).size), label="2024", color='black', ls='--')
    
    ax.set_ylabel("CDF")
    ax.set_xlabel("Duration (s)")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 60)
    ax.axvline(3, ls='--', color='red', lw=1)
    # Annotate the vertical line with an arrow and text
    ax.annotate('3 sec', xy=(3, 0.7), xytext=(15, 0.9),  # Annotation text and arrow position
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
                fontsize=14, color='black')
    ax.grid(True)
    ax.legend(loc='best', fontsize=14)
    plt.tight_layout()
    plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/nsa_nr_nr_time_diff_after_lte_ho.pdf')
    plt.close()

    fig, ax = plt.subplots(figsize=(4, 5))

    # Data for boxplot
    data = [nsa_nr_nr_time_diff_after_lte_ho, nsa_nr_nr_time_diff_after_lte_ho_3]

    # Boxplot
    ax.boxplot(data, patch_artist=True, widths=0.5,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red', linewidth=2),
            whiskerprops=dict(color='blue'),
            capprops=dict(color='blue'))

    # Set labels and title
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['2023', '2024'])
    ax.set_ylim(0, 200)
    ax.set_ylabel("Duration (s)")

    # Save the plot
    plt.tight_layout()
    #plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/nsa_nr_nr_time_diff_after_lte_ho_box.pdf')
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True, sharex=True)
    ax[0].plot(np.sort(nsa_nr_nr_duration_intra_list), np.linspace(0, 1, np.sort(nsa_nr_nr_duration_intra_list).size), label="NSA (2023)", color='salmon')
    ax[0].plot(np.sort(nsa_nr_nr_duration_intra_list_3), np.linspace(0, 1, np.sort(nsa_nr_nr_duration_intra_list_3).size), label="NSA (2024)", color='salmon', ls='--')
    ax[0].plot(np.sort(sa_duration_list_intra_freq), np.linspace(0, 1, np.sort(sa_duration_list_intra_freq).size), label="SA (2023)", color='slategrey')
    ax[0].plot(np.sort(sa_duration_list_intra_freq_3), np.linspace(0, 1, np.sort(sa_duration_list_intra_freq_3).size), label="SA (2024)", color='slategrey', ls='--')
    ax[0].set_title("Intra-frequency", fontsize=20, fontweight='bold')

    ax[1].plot(np.sort(nsa_nr_nr_duration_inter_list), np.linspace(0, 1, np.sort(nsa_nr_nr_duration_inter_list).size), label="NSA (2023)", color='salmon')
    ax[1].plot(np.sort(nsa_nr_nr_duration_inter_list_3), np.linspace(0, 1, np.sort(nsa_nr_nr_duration_inter_list_3).size), label="NSA (2024)", color='salmon', ls='--')
    ax[1].plot(np.sort(sa_duration_list_inter_freq), np.linspace(0, 1, np.sort(sa_duration_list_inter_freq).size), label="SA (2023)", color='slategrey')
    ax[1].plot(np.sort(sa_duration_list_inter_freq_3), np.linspace(0, 1, np.sort(sa_duration_list_inter_freq_3).size), label="SA (2024)", color='slategrey', ls='--')
    ax[1].set_title("Inter-frequency", fontsize=20, fontweight='bold')

    ax[0].set_ylabel("CDF", fontsize=20)
    # ax[0].set_xlabel("Handover duration (s)")
    # ax[1].set_xlabel("Handover duration (s)")
    fig.text(0.5, -0.03, "Handover duration (s)", ha='center', fontsize=21)
    ax[0].set_ylim(0, 1)
    ax[0].set_xlim(0, 0.1)
    ax[0].legend(loc='best', fontsize=14) 
    # ax[1].legend(loc='best')
    ax[0].grid(True)
    ax[1].grid(True)
    plt.tight_layout()
    plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/ho_duration_break.pdf')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    label_map_dict = {'intra_gnb:intra_freq' : "Intra-gNB + Intra-frequency", 'intra_gnb:inter_freq' : "Intra-gNB + Inter-frequency", 'inter_gnb:intra_freq' : "Inter-gNB + Intra-frequency", 'inter_gnb:inter_freq' : "Inter-gNB + Inter-frequency"}
    color_map_dict = {'intra_gnb:intra_freq' : "red", 'intra_gnb:inter_freq' : "green", 'inter_gnb:intra_freq' : "skyblue", 'inter_gnb:inter_freq' : "black"}

    for ho_type in label_map_dict.keys():
        ax.plot(np.sort(sa_inter_intra_dict[ho_type]), np.linspace(0, 1, np.sort(sa_inter_intra_dict[ho_type]).size), label=label_map_dict[ho_type] + " (2023)", color=color_map_dict[ho_type])
        ax.plot(np.sort(sa_inter_intra_dict_3[ho_type]), np.linspace(0, 1, np.sort(sa_inter_intra_dict_3[ho_type]).size), label=label_map_dict[ho_type] + " (2024)", color=color_map_dict[ho_type], ls='--')

    ax.set_ylabel("CDF")
    ax.set_xlabel("Handover duration (s)")
    ax.legend(loc='upper left', fontsize=12, ncol=2)
    ax.grid(True)
    ax.set_ylim(0, 1.5)
    ax.set_xlim(0, 0.1)
    plt.tight_layout()
    plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/sa_ho_duration_break.pdf')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    label_map_dict = {'intra_gnb:intra_freq' : "Intra-gNB + Intra-frequency", 'inter_gnb:intra_freq' : "Inter-gNB + Intra-frequency", 'inter_gnb:inter_freq' : "Inter-gNB + Inter-frequency"}
    color_map_dict = {'intra_gnb:intra_freq' : "red", 'intra_gnb:inter_freq' : "green", 'inter_gnb:intra_freq' : "skyblue", 'inter_gnb:inter_freq' : "black"}

    for ho_type in label_map_dict.keys():
        ax.plot(np.sort(nsa_inter_intra_dict[ho_type]), np.linspace(0, 1, np.sort(nsa_inter_intra_dict[ho_type]).size), label=label_map_dict[ho_type] + " (2023)", color=color_map_dict[ho_type])
        ax.plot(np.sort(nsa_inter_intra_dict_3[ho_type]), np.linspace(0, 1, np.sort(nsa_inter_intra_dict_3[ho_type]).size), label=label_map_dict[ho_type] + " (2024)", color=color_map_dict[ho_type], ls='--')

    ax.set_ylabel("CDF")
    ax.set_xlabel("Handover duration (s)")
    ax.legend(loc='upper left', fontsize=12, ncol=2)
    ax.grid(True)
    ax.set_ylim(0, 1.5)
    ax.set_xlim(0, 0.1)
    plt.tight_layout()
    plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/nsa_ho_duration_break.pdf')
    plt.close()


    # tx power 
    fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    col_dict ={'5G (NSA)' : 'salmon', '5G (SA)' : 'slategrey'}
    for tech in main_sa_nsa_tx_power_dict_1.keys():
        tx_power = main_sa_nsa_tx_power_dict_1[tech].copy()
        tx_power = [float(i) for i in tx_power if not pd.isnull(i) and type(i) != str]
        tx_power = np.sort(tx_power)

        tx_power_control = main_sa_nsa_tx_power_control_dict_1[tech].copy()
        tx_power_control = [float(i) for i in tx_power_control if not pd.isnull(i) and type(i) != str]
        tx_power_control = np.sort(tx_power_control)
        
        rx_power = main_sa_nsa_rx_power_dict_1[tech].copy()
        rx_power = [float(i) for i in rx_power if not pd.isnull(i)]
        rx_power = np.sort(rx_power)

        pathloss = main_sa_nsa_pathloss_dict_1[tech].copy()
        pathloss = [float(i) for i in pathloss if not pd.isnull(i)]
        pathloss = np.sort(pathloss)

        ax[0].plot(tx_power, np.linspace(0, 1, tx_power.size), label=tech + " (2023)"               , color=col_dict[tech] )
        ax[1].plot(tx_power_control, np.linspace(0, 1, tx_power_control.size), label=tech + " (2023)", color=col_dict[tech])
        ax[2].plot(rx_power, np.linspace(0, 1, rx_power.size), label=tech        + " (2023)"          , color=col_dict[tech])
        # ax[3].plot(pathloss, np.linspace(0, 1, pathloss.size), label=tech                , color=col_dict[tech])

    for tech in main_sa_nsa_tx_power_dict_3.keys():
        tx_power = main_sa_nsa_tx_power_dict_3[tech].copy()
        tx_power = [float(i) for i in tx_power if not pd.isnull(i) and type(i) != str]
        tx_power = np.sort(tx_power)

        tx_power_control = main_sa_nsa_tx_power_control_dict_3[tech].copy()
        tx_power_control = [float(i) for i in tx_power_control if not pd.isnull(i) and type(i) != str]
        tx_power_control = np.sort(tx_power_control)
        
        rx_power = main_sa_nsa_rx_power_dict_3[tech].copy()
        rx_power = [float(i) for i in rx_power if not pd.isnull(i)]
        rx_power = np.sort(rx_power)

        pathloss = main_sa_nsa_pathloss_dict_3[tech].copy()
        pathloss = [float(i) for i in pathloss if not pd.isnull(i)]
        pathloss = np.sort(pathloss)

        ax[0].plot(tx_power, np.linspace(0, 1, tx_power.size), label=tech + " (2024)"               , color=col_dict[tech] , ls='--')
        ax[1].plot(tx_power_control, np.linspace(0, 1, tx_power_control.size), label=tech + " (2024)", color=col_dict[tech], ls='--')
        ax[2].plot(rx_power, np.linspace(0, 1, rx_power.size), label=tech   + " (2024)"                , color=col_dict[tech], ls='--')
        # ax[3].plot(pathloss, np.linspace(0, 1, pathloss.size), label=tech                , color=col_dict[tech])


    ax[0].set_ylabel("CDF")
    ax[0].set_xlabel("PUSCH Power (dBm)", fontsize=15)
    ax[1].set_xlabel("PUCCH Power (dBm)", fontsize=15)
    ax[2].set_xlabel("RSRP (dBm)")
    # ax[3].set_xlabel("Pathloss (dB)")
    ax[0].set_ylim(0, 1)
    ax[0].legend(loc='best', fontsize=11)
    # ax[1].legend(loc='best')
    # ax[2].legend(loc='best')
    # ax[3].legend(loc='best')
    ax[0].set_xlim(-40, 28)
    ax[1].set_xlim(-40, 28)
    ax[2].set_xlim(-110, -60)
    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)
    plt.tight_layout()
    plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/tx_power_all.pdf')
    plt.close()



    fig, ax = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    for tech in main_tech_xput_dl_tx_power_dict_1.keys():
        if 'NSA' in tech:
            ls = '-'
        else:
            ls = '--'
        tx_power_dl = main_tech_xput_dl_tx_power_dict_1[tech].copy()
        tx_power_dl = [i for i in tx_power_dl if not pd.isnull(i)]
        tx_power_dl = np.sort(tx_power_dl)

        tx_power_control_dl = main_tech_xput_dl_tx_power_control_dict_1[tech].copy()
        tx_power_control_dl = [i for i in tx_power_control_dl if not pd.isnull(i)]
        tx_power_control_dl = np.sort(tx_power_control_dl)

        rx_power_dl = main_tech_xput_dl_rx_power_dict_1[tech].copy()
        rx_power_dl = [i for i in rx_power_dl if not pd.isnull(i)]
        rx_power_dl = np.sort(rx_power_dl)

        pathloss_dl = main_tech_xput_dl_pathloss_dict_1[tech].copy()
        pathloss_dl = [i for i in pathloss_dl if not pd.isnull(i)]
        pathloss_dl = np.sort(pathloss_dl)

        tx_power_ul = main_tech_xput_ul_tx_power_dict_1[tech].copy()
        tx_power_ul = [i for i in tx_power_ul if not pd.isnull(i)]
        tx_power_ul = np.sort(tx_power_ul)

        tx_power_control_ul = main_tech_xput_ul_tx_power_control_dict_1[tech].copy()
        tx_power_control_ul = [i for i in tx_power_control_ul if not pd.isnull(i)]
        tx_power_control_ul = np.sort(tx_power_control_ul)

        rx_power_ul = main_tech_xput_ul_rx_power_dict_1[tech].copy()
        rx_power_ul = [i for i in rx_power_ul if not pd.isnull(i)]
        rx_power_ul = np.sort(rx_power_ul)

        pathloss_ul = main_tech_xput_ul_pathloss_dict_1[tech].copy()
        pathloss_ul = [i for i in pathloss_ul if not pd.isnull(i)]
        pathloss_ul = np.sort(pathloss_ul)

        tx_power_ping = main_ping_tx_power_dict_1[tech].copy()
        tx_power_ping = [i for i in tx_power_ping if not pd.isnull(i)]
        tx_power_ping = np.sort(tx_power_ping)

        tx_power_control_ping = main_ping_tx_power_control_dict_1[tech].copy()
        tx_power_control_ping = [i for i in tx_power_control_ping if not pd.isnull(i)]
        tx_power_control_ping = np.sort(tx_power_control_ping)

        rx_power_ping = main_tech_ping_rsrp_dict_1[tech].copy()
        rx_power_ping = [i for i in rx_power_ping if not pd.isnull(i)]
        rx_power_ping = np.sort(rx_power_ping)

        pathloss_ping = main_tech_ping_pathloss_dict_1[tech].copy()
        pathloss_ping = [i for i in pathloss_ping if not pd.isnull(i)]
        pathloss_ping = np.sort(pathloss_ping)

        ax[0].plot(tx_power_dl, np.linspace(0, 1, tx_power_dl.size), label=tech + " (DL)", color='salmon', ls=ls)
        ax[1].plot(tx_power_control_dl, np.linspace(0, 1, tx_power_control_dl.size), label=tech + " (DL)", color='salmon', ls=ls)
        ax[2].plot(rx_power_dl, np.linspace(0, 1, rx_power_dl.size), label=tech + " (DL)", color='salmon', ls=ls)
        # ax[3].plot(pathloss_dl, np.linspace(0, 1, pathloss_dl.size), label=tech + " (DL)", color='red', ls=ls)
        ax[0].plot(tx_power_ul, np.linspace(0, 1, tx_power_ul.size), label=tech + " (UL)",  color='black',     ls=ls)
        ax[1].plot(tx_power_control_ul, np.linspace(0, 1, tx_power_control_ul.size), label=tech + " (UL)", color='black',     ls=ls)
        ax[2].plot(rx_power_ul, np.linspace(0, 1, rx_power_ul.size), label=tech + " (UL)", color='black',     ls=ls)
        # ax[3].plot(pathloss_ul, np.linspace(0, 1, pathloss_ul.size), label=tech + " (UL)", color='black',     ls=ls)
        ax[0].plot(tx_power_ping, np.linspace(0, 1, tx_power_ping.size), label=tech + " (RTT)",  color='gray',     ls=ls)
        ax[1].plot(tx_power_control_ping, np.linspace(0, 1, tx_power_control_ping.size), label=tech + " (RTT)", color='gray',     ls=ls)
        ax[2].plot(rx_power_ping, np.linspace(0, 1, rx_power_ping.size), label=tech + " (RTT)", color='gray',     ls=ls)
        # ax[3].plot(pathloss_ping, np.linspace(0, 1, pathloss_ping.size), label=tech + " (RTT)", color='lime',     ls=ls)

    ax[0].set_ylabel("CDF")
    ax[0].set_xlabel("PUSCH Power (dBm)")
    ax[1].set_xlabel("PUCCH Power (dBm)")
    ax[2].set_xlabel("RSRP (dBm)")
    # ax[3].set_xlabel("Pathloss (dB)")
    ax[0].set_ylim(0, 1)
    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)
    ax[0].set_xlim(-40, 28)
    ax[1].set_xlim(-40, 28)
    ax[2].set_xlim(-110, -60)
    plt.tight_layout()
    plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/tx_power_break_2023.pdf')
    plt.close()
    a = 1


    fig, ax = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    for tech in main_tech_xput_dl_tx_power_dict_3.keys():
        if 'NSA' in tech:
            ls = '-'
        else:
            ls = '--'
        tx_power_dl = main_tech_xput_dl_tx_power_dict_3[tech].copy()
        tx_power_dl = [i for i in tx_power_dl if not pd.isnull(i)]
        tx_power_dl = np.sort(tx_power_dl)

        tx_power_control_dl = main_tech_xput_dl_tx_power_control_dict_3[tech].copy()
        tx_power_control_dl = [i for i in tx_power_control_dl if not pd.isnull(i)]
        tx_power_control_dl = np.sort(tx_power_control_dl)

        rx_power_dl = main_tech_xput_dl_rx_power_dict_3[tech].copy()
        rx_power_dl = [i for i in rx_power_dl if not pd.isnull(i)]
        rx_power_dl = np.sort(rx_power_dl)

        pathloss_dl = main_tech_xput_dl_pathloss_dict_3[tech].copy()
        pathloss_dl = [i for i in pathloss_dl if not pd.isnull(i)]
        pathloss_dl = np.sort(pathloss_dl)

        tx_power_ul = main_tech_xput_ul_tx_power_dict_3[tech].copy()
        tx_power_ul = [i for i in tx_power_ul if not pd.isnull(i)]
        tx_power_ul = np.sort(tx_power_ul)

        tx_power_control_ul = main_tech_xput_ul_tx_power_control_dict_3[tech].copy()
        tx_power_control_ul = [i for i in tx_power_control_ul if not pd.isnull(i)]
        tx_power_control_ul = np.sort(tx_power_control_ul)

        rx_power_ul = main_tech_xput_ul_rx_power_dict_3[tech].copy()
        rx_power_ul = [i for i in rx_power_ul if not pd.isnull(i)]
        rx_power_ul = np.sort(rx_power_ul)

        pathloss_ul = main_tech_xput_ul_pathloss_dict_3[tech].copy()
        pathloss_ul = [i for i in pathloss_ul if not pd.isnull(i)]
        pathloss_ul = np.sort(pathloss_ul)

        tx_power_ping = main_ping_tx_power_dict_3[tech].copy()
        tx_power_ping = [i for i in tx_power_ping if not pd.isnull(i)]
        tx_power_ping = np.sort(tx_power_ping)

        tx_power_control_ping = main_ping_tx_power_control_dict_3[tech].copy()
        tx_power_control_ping = [i for i in tx_power_control_ping if not pd.isnull(i)]
        tx_power_control_ping = np.sort(tx_power_control_ping)

        rx_power_ping = main_tech_ping_rsrp_dict_3[tech].copy()
        rx_power_ping = [i for i in rx_power_ping if not pd.isnull(i)]
        rx_power_ping = np.sort(rx_power_ping)

        pathloss_ping = main_tech_ping_pathloss_dict_3[tech].copy()
        pathloss_ping = [i for i in pathloss_ping if not pd.isnull(i)]
        pathloss_ping = np.sort(pathloss_ping)


        ax[0].plot(tx_power_dl, np.linspace(0, 1, tx_power_dl.size), label=tech + " (DL)", color='salmon', ls=ls)
        ax[1].plot(tx_power_control_dl, np.linspace(0, 1, tx_power_control_dl.size), label=tech + " (DL)", color='salmon', ls=ls)
        ax[2].plot(rx_power_dl, np.linspace(0, 1, rx_power_dl.size), label=tech + " (DL)", color='salmon', ls=ls)
        # ax[3].plot(pathloss_dl, np.linspace(0, 1, pathloss_dl.size), label=tech + " (DL)", color='red', ls=ls)
        ax[0].plot(tx_power_ul, np.linspace(0, 1, tx_power_ul.size), label=tech + " (UL)",  color='black',     ls=ls)
        ax[1].plot(tx_power_control_ul, np.linspace(0, 1, tx_power_control_ul.size), label=tech + " (UL)", color='black',     ls=ls)
        ax[2].plot(rx_power_ul, np.linspace(0, 1, rx_power_ul.size), label=tech + " (UL)", color='black',     ls=ls)
        # ax[3].plot(pathloss_ul, np.linspace(0, 1, pathloss_ul.size), label=tech + " (UL)", color='black',     ls=ls)
        ax[0].plot(tx_power_ping, np.linspace(0, 1, tx_power_ping.size), label=tech + "(RTT)",  color='gray',     ls=ls)
        ax[1].plot(tx_power_control_ping, np.linspace(0, 1, tx_power_control_ping.size), label=tech + " (RTT)", color='gray',     ls=ls)
        ax[2].plot(rx_power_ping, np.linspace(0, 1, rx_power_ping.size), label=tech + " (RTT)", color='gray',     ls=ls)
        # ax[3].plot(pathloss_ping, np.linspace(0, 1, pathloss_ping.size), label=tech + " (RTT)", color='lime',     ls=ls)

    ax[0].set_ylabel("CDF")
    ax[0].set_xlabel("PUSCH Power (dBm)")
    ax[1].set_xlabel("PUCCH Power (dBm)")
    ax[2].set_xlabel("RSRP (dBm)")
    # ax[3].set_xlabel("Pathloss (dB)")
    ax[0].set_ylim(0, 1)
    ax[0].grid(True)
    ax[0].legend(loc='best', fontsize=13)
    ax[1].grid(True)
    ax[2].grid(True)
    ax[0].set_xlim(-40, 28)
    ax[1].set_xlim(-40, 28)
    ax[2].set_xlim(-110, -60)
    # ax[0].legend(loc='best')
    plt.tight_layout()
    plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/tx_power_break_2024.pdf')
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
    col_dict ={'5G (NSA)' : 'salmon', '5G (SA)' : 'slategrey'}
    for tech in main_sa_nsa_tx_power_dict_1.keys():
        tx_power = main_sa_nsa_tx_power_dict_1[tech].copy()
        tx_power = [float(i) for i in tx_power if not pd.isnull(i) and type(i) != str]

        tx_power_control = main_sa_nsa_tx_power_control_dict_1[tech].copy()
        tx_power_control = [float(i) for i in tx_power_control if not pd.isnull(i) and type(i) != str]
        
        rx_power = main_sa_nsa_rx_power_dict_1[tech].copy()
        rx_power = [float(i) for i in rx_power if not pd.isnull(i)]

        pathloss = main_sa_nsa_pathloss_dict_1[tech].copy()
        pathloss = [float(i) for i in pathloss if not pd.isnull(i)]
        

        tx_power.extend(main_sa_nsa_tx_power_dict_3[tech])
        tx_power = [float(i) for i in tx_power if not pd.isnull(i) and type(i) != str]
        tx_power = np.sort(tx_power)

        tx_power_control.extend(main_sa_nsa_tx_power_control_dict_3[tech])
        tx_power_control = [float(i) for i in tx_power_control if not pd.isnull(i) and type(i) != str]
        tx_power_control = np.sort(tx_power_control)
        rx_power.extend(main_sa_nsa_rx_power_dict_3[tech])
        rx_power = [float(i) for i in rx_power if not pd.isnull(i)]
        rx_power = np.sort(rx_power)

        pathloss.extend(main_sa_nsa_pathloss_dict_3[tech])
        pathloss = [float(i) for i in pathloss if not pd.isnull(i)]
        pathloss = np.sort(pathloss)
        print("%s --> %s" %(tech, str(np.median(pathloss))))

        ax[0].plot(tx_power, np.linspace(0, 1, tx_power.size), label=tech          , color=col_dict[tech] , ls='-')
        ax[1].plot(tx_power_control, np.linspace(0, 1, tx_power_control.size), label=tech, color=col_dict[tech], ls='-')
        # ax[2].plot(rx_power, np.linspace(0, 1, rx_power.size), label=tech   , color=col_dict[tech], ls='-')
        # ax[3].plot(pathloss, np.linspace(0, 1, pathloss.size), label=tech                , color=col_dict[tech])


    ax[0].set_ylabel("CDF")
    ax[0].set_xlabel("PUSCH Power (dBm)", fontsize=15)
    ax[1].set_xlabel("PUCCH Power (dBm)", fontsize=15)
    # ax[2].set_xlabel("RSRP (dBm)", fontsize=15)
    # ax[3].set_xlabel("Pathloss (dB)", fontsize=15)
    ax[0].set_ylim(0, 1)
    ax[0].legend(loc='best', fontsize=13)
    # ax[1].legend(loc='best')
    # ax[2].legend(loc='best')
    # ax[3].legend(loc='best')
    ax[0].set_xlim(-10, 28)
    ax[1].set_xlim(-40, 28)
    # ax[2].set_xlim(-110, -60)
    # ax[3].set_xlim(80, 150)
    ax[0].grid(True)
    ax[1].grid(True)
    # ax[2].grid(True)
    # ax[3].grid(True)
    plt.tight_layout()
    plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/tx_power_mixed.pdf')
    plt.close()

    fig, ax = plt.subplots(figsize=(3, 3))
    col_dict ={'5G (NSA)' : 'salmon', '5G (SA)' : 'slategrey'}
    for tech in main_sa_nsa_tx_power_dict_1.keys():
        pathloss = main_sa_nsa_pathloss_dict_1[tech].copy()
        pathloss = [float(i) for i in pathloss if not pd.isnull(i)]
        pathloss.extend(main_sa_nsa_pathloss_dict_3[tech])
        pathloss = [float(i) for i in pathloss if not pd.isnull(i)]
        pathloss = np.sort(pathloss)
        ax.plot(pathloss, np.linspace(0, 1, pathloss.size), label=tech, color=col_dict[tech], ls='-')

    ax.set_ylabel("CDF")
    ax.set_xlabel("Pathloss (dB)", fontsize=15)
    ax.set_ylim(0, 1)
    ax.set_xlim(80, 140)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/pathloss_mixed.pdf')
    plt.close()

    fig, ax = plt.subplots(figsize=(4, 3))
    for tech in main_tech_xput_dl_tx_power_dict_1.keys():
        if 'NSA' in tech:
            ls = '-'
        else:
            ls = '--'
        tx_power_dl = main_tech_xput_dl_tx_power_dict_1[tech].copy()
        tx_power_dl = [i for i in tx_power_dl if not pd.isnull(i)]

        tx_power_control_dl = main_tech_xput_dl_tx_power_control_dict_1[tech].copy()
        tx_power_control_dl = [i for i in tx_power_control_dl if not pd.isnull(i)]

        rx_power_dl = main_tech_xput_dl_rx_power_dict_1[tech].copy()
        rx_power_dl = [i for i in rx_power_dl if not pd.isnull(i)]


        tx_power_ul = main_tech_xput_ul_tx_power_dict_1[tech].copy()
        tx_power_ul = [i for i in tx_power_ul if not pd.isnull(i)]

        tx_power_control_ul = main_tech_xput_ul_tx_power_control_dict_1[tech].copy()
        tx_power_control_ul = [i for i in tx_power_control_ul if not pd.isnull(i)]

        rx_power_ul = main_tech_xput_ul_rx_power_dict_1[tech].copy()
        rx_power_ul = [i for i in rx_power_ul if not pd.isnull(i)]

        tx_power_ping = main_ping_tx_power_dict_1[tech].copy()
        tx_power_ping = [i for i in tx_power_ping if not pd.isnull(i)]

        tx_power_control_ping = main_ping_tx_power_control_dict_1[tech].copy()
        tx_power_control_ping = [i for i in tx_power_control_ping if not pd.isnull(i)]

        rx_power_ping = main_tech_ping_rsrp_dict_1[tech].copy()
        rx_power_ping = [i for i in rx_power_ping if not pd.isnull(i)]


        tx_power_dl.extend(main_tech_xput_dl_tx_power_dict_3[tech])
        tx_power_dl = [i for i in tx_power_dl if not pd.isnull(i)]
        tx_power_dl = np.sort(tx_power_dl)

        tx_power_control_dl.extend(main_tech_xput_dl_tx_power_control_dict_3[tech])
        tx_power_control_dl = [i for i in tx_power_control_dl if not pd.isnull(i)]
        tx_power_control_dl = np.sort(tx_power_control_dl)

        rx_power_dl.extend(main_tech_xput_dl_rx_power_dict_3[tech])
        rx_power_dl = [i for i in rx_power_dl if not pd.isnull(i)]
        rx_power_dl = np.sort(rx_power_dl)

        tx_power_ul.extend(main_tech_xput_ul_tx_power_dict_3[tech])
        tx_power_ul = [i for i in tx_power_ul if not pd.isnull(i)]
        tx_power_ul = np.sort(tx_power_ul)

        tx_power_control_ul.extend(main_tech_xput_ul_tx_power_control_dict_3[tech])
        tx_power_control_ul = [i for i in tx_power_control_ul if not pd.isnull(i)]
        tx_power_control_ul = np.sort(tx_power_control_ul)

        rx_power_ul.extend(main_tech_xput_ul_rx_power_dict_3[tech])
        rx_power_ul = [i for i in rx_power_ul if not pd.isnull(i)]
        rx_power_ul = np.sort(rx_power_ul)

        tx_power_ping.extend(main_ping_tx_power_dict_3[tech])
        tx_power_ping = [i for i in tx_power_ping if not pd.isnull(i)]
        tx_power_ping = np.sort(tx_power_ping)

        tx_power_control_ping.extend(main_ping_tx_power_control_dict_3[tech])
        tx_power_control_ping = [i for i in tx_power_control_ping if not pd.isnull(i)]
        tx_power_control_ping = np.sort(tx_power_control_ping)

        rx_power_ping.extend(main_tech_ping_rsrp_dict_3[tech])
        rx_power_ping = [i for i in rx_power_ping if not pd.isnull(i)]
        rx_power_ping = np.sort(rx_power_ping)

        pathloss_dl = main_tech_xput_dl_pathloss_dict_1[tech].copy()
        pathloss_dl.extend(main_tech_xput_dl_pathloss_dict_3[tech])
        pathloss_dl = [i for i in pathloss_dl if not pd.isnull(i)]
        pathloss_dl = np.sort(pathloss_dl)

        pathloss_ul = main_tech_xput_ul_pathloss_dict_1[tech].copy()
        pathloss_ul.extend(main_tech_xput_ul_pathloss_dict_3[tech])
        pathloss_ul = [i for i in pathloss_ul if not pd.isnull(i)]
        pathloss_ul = np.sort(pathloss_ul)

        pathloss_ping = main_tech_ping_pathloss_dict_1[tech].copy()
        pathloss_ping.extend(main_tech_ping_pathloss_dict_3[tech])
        pathloss_ping = [i for i in pathloss_ping if not pd.isnull(i)]
        pathloss_ping = np.sort(pathloss_ping)


        ax.plot(tx_power_dl, np.linspace(0, 1, tx_power_dl.size), label=tech + " (DL)", color='salmon', ls=ls)
        # ax[1].plot(tx_power_control_dl, np.linspace(0, 1, tx_power_control_dl.size), label=tech + " (DL)", color='salmon', ls=ls)
        # ax[2].plot(rx_power_dl, np.linspace(0, 1, rx_power_dl.size), label=tech + " (DL)", color='salmon', ls=ls)
        # ax[3].plot(pathloss_dl, np.linspace(0, 1, pathloss_dl.size), label=tech + " (DL)", color='salmon', ls=ls)
        ax.plot(tx_power_ul, np.linspace(0, 1, tx_power_ul.size), label=tech + " (UL)",  color='black',     ls=ls)
        # ax[1].plot(tx_power_control_ul, np.linspace(0, 1, tx_power_control_ul.size), label=tech + " (UL)", color='black',     ls=ls)
        # ax[2].plot(rx_power_ul, np.linspace(0, 1, rx_power_ul.size), label=tech + " (UL)", color='black',     ls=ls)
        # ax[3].plot(pathloss_ul, np.linspace(0, 1, pathloss_ul.size), label=tech + " (UL)", color='black',     ls=ls)
        ax.plot(tx_power_ping, np.linspace(0, 1, tx_power_ping.size), label=tech + "(RTT)",  color='gray',     ls=ls)
        # ax[1].plot(tx_power_control_ping, np.linspace(0, 1, tx_power_control_ping.size), label=tech + " (RTT)", color='gray',     ls=ls)
        # ax[2].plot(rx_power_ping, np.linspace(0, 1, rx_power_ping.size), label=tech + " (RTT)", color='gray',     ls=ls)
        # ax[3].plot(pathloss_ping, np.linspace(0, 1, pathloss_ping.size), label=tech + " (RTT)", color='gray',     ls=ls)

    ax.set_ylabel("CDF")
    ax.set_xlabel("PUSCH Power (dBm)")
    # ax[1].set_xlabel("PUCCH Power (dBm)")
    # ax[2].set_xlabel("RSRP (dBm)")
    # ax[3].set_xlabel("Pathloss (dB)")
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(loc='best', fontsize=13)
    # ax[1].grid(True)
    # ax[2].grid(True)
    # ax[3].grid(True)
    ax.set_xlim(-15, 28)
    # ax[1].set_xlim(-40, 28)
    # ax[2].set_xlim(-110, -60)
    # ax[3].set_xlim(80, 150)
    # ax[0].legend(loc='best')
    plt.tight_layout()
    plt.savefig('/home/moinakgh/csv_ho/nsa_sa_analysis_perf/plots/yearwise/tx_power_break_mixed.pdf')
    plt.close()
