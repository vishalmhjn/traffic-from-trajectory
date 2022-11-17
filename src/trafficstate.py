from generic import Get_Centerline
import json
import numpy as np

def get_lane_configuration(meta):
	'''Obtain lane configuration'''
	if len(Get_Centerline(meta)[0][0])==3:
		lane_config = [2, 3, 4, 6, 7, 8]
	elif len(Get_Centerline(meta)[0][0])==2:
		lane_config = [2, 3, 5, 6]
	return lane_config

def get_lane_traffic_density(i, df, lane_config, return_dict):
	'''extract traffic density lane-wise'''
	list_density = []
	for frame in df['frame'].unique():
		df_frame = df[df['frame']==frame]
		if i >= len(lane_config)/2:
			df_frame_dl = df_frame[(df_frame['xVelocity']>0) & (df_frame['laneId']==lane_config[i])]
			n_vehicles_dl_frame = len(df_frame_dl)
			list_density.append(n_vehicles_dl_frame)
		else:
			df_frame_ul = df_frame[(df_frame['xVelocity']<0) & (df_frame['laneId']==lane_config[i])]
			n_vehicles_ul_frame = len(df_frame_ul)
			list_density.append(n_vehicles_ul_frame)
	return_dict[lane_config[i]] = list_density
	return return_dict #list_density


def get_lane_flowrate(j, df, lane_config, detector_location, delta_time, return_dict):
	'''Extract lane density at the detector location on the observed section'''
	list_flowrate = []
	frame_seq = df['frame'].unique()[::delta_time]
	df_lane = df[df['laneId']==lane_config[j]]
	for i in range(0, len(frame_seq)-1):
		list_vehicle_initial = []
		list_vehicle_final = []
		for frame in range(frame_seq[i], frame_seq[i+1]):
			if j >= len(lane_config)/2:
				df_temp_frame = df_lane[(df_lane['frame']==frame) & (df_lane['xVelocity']>0)]
				list_vehicle_initial.extend(list(df_temp_frame[df_temp_frame['x']<detector_location]['id'].unique()))
				list_vehicle_final.extend(list(df_temp_frame[df_temp_frame['x']>detector_location]['id'].unique()))           
			else:
				df_temp_frame = df_lane[(df_lane['frame']==frame) & (df_lane['xVelocity']<0)]
				list_vehicle_initial.extend(list(df_temp_frame[df_temp_frame['x']>detector_location]['id'].unique()))
				list_vehicle_final.extend(list(df_temp_frame[df_temp_frame['x']<detector_location]['id'].unique()))
			
		list_veh_initial_uniq = list(set(list_vehicle_initial))
		list_veh_final_uniq = list(set(list_vehicle_final))

		list_flowrate.append(len(list(set(list_veh_initial_uniq).intersection(list_veh_final_uniq))))
	return_dict[lane_config[j]] = list_flowrate
	print(return_dict)
	return return_dict #list_flowrate

def get_lane_velocity(j, df, lane_config, delta_time, return_dict):
	'''Extract lane density at the detector location on the observed section'''
	list_velocity = []
	frame_seq = df['frame'].unique()[::delta_time]
	df_lane = df[df['laneId']==lane_config[j]]
	for i in range(0, len(frame_seq)-1):
		if j >= len(lane_config)/2:
			df_temp_frame = df_lane[(df_lane['frame']>frame_seq[i]) & (df_lane['frame']<frame_seq[i+1])]
			
			list_velocity.append(3.6*np.mean(list(df_temp_frame['xVelocity'])))          
		else:
			df_temp_frame = df_lane[(df_lane['frame']>frame_seq[i]) & (df_lane['frame']<frame_seq[i+1])]
			list_velocity.append(-3.6*np.mean(list(df_temp_frame['xVelocity'])))
	return_dict[lane_config[j]] = list_velocity
	return return_dict #list_flowrate

def get_lane_truck_cars(j, df, lane_config, delta_time, return_dict1, return_dict2, truck_width = 8):
	'''Extract lane density at the detector location on the observed section'''
	list_trucks = []
	list_cars = []
	frame_seq = df['frame'].unique()[::delta_time]
	df_lane = df[df['laneId']==lane_config[j]]
	for i in range(0, len(frame_seq)-1):
		df_temp_frame = df_lane[(df_lane['frame']>frame_seq[i]) & (df_lane['frame']<frame_seq[i+1])]
		total_vehicles = df_temp_frame['id'].unique()
		trucks = len(df_temp_frame[df_temp_frame['width']>truck_width]['id'].unique())
		cars = len(df_temp_frame['id'].unique()) - trucks
		list_cars.append(cars) 
		list_trucks.append(trucks)
	return_dict1[lane_config[j]] = list_cars
	return_dict2[lane_config[j]] = list_trucks
	return return_dict1, return_dict2

def write_file(file, dictionary, tag):
	'''Write the dictionary in a json format'''
	filename = '../data_processed/macro_variables/'+file+tag+'.json'
	with open(filename, 'w') as fp:
		json.dump(dictionary, fp)
		
