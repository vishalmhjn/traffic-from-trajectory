#extract following and lane changing vehicle trajectories
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import typing
from typing import List

def Get_Filenames(Length):
	filelist = []
	for i in range(1, Length+1):
		if np.floor(i/10) == 0:
			filelist = np.append(filelist, '0'+str(i)+'_tracks.csv')
		else:
			filelist = np.append(filelist, str(i)+'_tracks.csv')
	return filelist
	
def Get_Filenumbers(Length):
	filelist = []
	for i in range(1, Length+1):
		if np.floor(i/10) == 0:
			filelist = np.append(filelist, '0'+str(i))
		else:
			filelist = np.append(filelist, str(i))
	return filelist
	
	

def read_data(tracknumber):
	'''Function to real the data from track files'''
	recording_meta = pd.read_csv("../data/highd/"+tracknumber+"_recordingMeta.csv")
	tracks = pd.read_csv("../data/highd/"+tracknumber+"_tracks.csv")
	tracks_meta = pd.read_csv("../data/highd/"+tracknumber+"_tracksMeta.csv")
	Image_path = "../data/highd/"+tracknumber+"_highway.jpg"
	return recording_meta, tracks, Image_path

def trajectory_plot(id: List[int], image, track, width)-> None:
	fig, ax = plt.subplots(figsize=(100,10))
	for i in id:
		temp = track[track['id']==i]
		scatter = ax.scatter(temp['x']/(4*0.10106), temp['y']/(4*0.10106), linewidth = width, c='r')
	plt.imshow(plt.imread(image), alpha=0.5)


def Calculate_center(df):
	'''The x and y in tracks file are the upper left corner of the bounding boxes. In order to locate the centre
	of the vehicles, following function is used'''
	df['x_center'] = df.apply(lambda x: (x['x']+x['width']/2),axis=1)
	df['y_center'] = df.apply(lambda x: (x['y']+x['height']/2),axis=1)
	return df
		



def Extract_SSM(df):
	'''This function iterates through each frame in data. At each frame instance, it then iterates through each vehicle
	present in the scene to identify if a leader-follower pair exists. Then for each leader follower pair, it calculates
	the SSMs metrics viz. distance headway (dhw), time headway (thw) and time to collision (ttc). For the rest (no leader-
	follower pair), it assigns a dummy value equal to 0. This function is made to extract SSMs, For now it also checks
	the values of SSMs provided in highD. The value of dhw and thw match closely over all range. But the values of ttc
	match within the normal range. The higher values of ttc show a deviation due to division error (when velocities
	are almost equal) but that should not affect out study, as such high values of ttc are irrelevant.'''
	df['manual_dhw'] = 0
	df['manual_thw'] = 0
	df['manual_ttc'] = 0
	
	for frame in df['frame'].unique():
		temp = df[df['frame'] == frame]
		for vehicle in temp[temp['frame'] == frame]['id']:
			vehicle_index = temp[(temp['frame'] == frame) & (temp['id'] == vehicle)].index.values.astype(int)[0]
			if  temp.at[vehicle_index, 'precedingId'] != 0:
				precedingvehicle_index = temp[(temp['id'] == temp.at[vehicle_index, 'precedingId'])].index.values.astype(int)[0]
				if df.at[vehicle_index, 'xVelocity'] < 0:
					df.at[vehicle_index, 'manual_dhw'] = abs(df.at[precedingvehicle_index, 'x'] - df.at[vehicle_index, 'x']) - df.at[precedingvehicle_index, 'width']
				else:
					df.at[vehicle_index, 'manual_dhw'] = abs(df.at[vehicle_index, 'x'] - df.at[precedingvehicle_index, 'x']) - df.at[vehicle_index, 'width'] 
				df.at[vehicle_index, 'manual_thw'] = df.at[vehicle_index, 'manual_dhw']/abs(df.at[vehicle_index, 'xVelocity'])
				df.loc[vehicle_index, 'manual_ttc'] = df.at[vehicle_index, 'manual_dhw']/(abs(df.at[vehicle_index, 'xVelocity']) - abs(df.at[precedingvehicle_index, 'xVelocity']))
				#print('Done:'+str(frame))
		if frame > 300:
			break
	return df


def Extract_Velnei(df):
	''' Following function computes the average of the Velocities of the neighbours of the ego car (maximum possible 8)
	at each frame intstance. It is very inefficient due to number of if conidtionals. Will try to make it more efficient hopefully.'''
	df['V_nei'] = 0
	for frame in df['frame'].unique():
		temp = df[df['frame'] == frame]
		for vehicle in temp[temp['frame'] == frame]['id']:
			v_nei = 0
			vehicle_index = temp[(temp['frame'] == frame) & (temp['id'] == vehicle)].index.values.astype(int)[0]
			count_neighbours = np.count_nonzero(tracks.at[vehicle_index, 'precedingId':'rightFollowingId'])
			if  temp.loc[vehicle_index, 'precedingId'] != 0:
				neighbour_index = temp[(temp['id'] == temp.at[vehicle_index, 'precedingId'])].index.values.astype(int)[0]
				v_nei += abs(df.at[neighbour_index, 'xVelocity'])
			if  temp.at[vehicle_index, 'followingId'] != 0:
				neighbour_index = temp[(temp['id'] == temp.at[vehicle_index, 'followingId'])].index.values.astype(int)[0]
				v_nei += abs(df.at[neighbour_index, 'xVelocity'])
			if  temp.at[vehicle_index, 'leftPrecedingId'] != 0:
				neighbour_index  = temp[(temp['id'] == temp.at[vehicle_index, 'leftPrecedingId'])].index.values.astype(int)[0]
				v_nei += abs(df.at[neighbour_index, 'xVelocity'])
			if  temp.at[vehicle_index, 'leftAlongsideId'] != 0:
				neighbour_index = temp[(temp['id'] == temp.at[vehicle_index, 'leftAlongsideId'])].index.values.astype(int)[0]
				v_nei += abs(df.at[neighbour_index, 'xVelocity'])
			if  temp.at[vehicle_index, 'leftFollowingId'] != 0:
				neighbour_index  = temp[(temp['id'] == temp.at[vehicle_index, 'leftFollowingId'])].index.values.astype(int)[0]
				v_nei += abs(df.at[neighbour_index, 'xVelocity'])
			if  temp.at[vehicle_index, 'rightPrecedingId'] != 0:
				neighbour_index  = temp[(temp['id'] == temp.at[vehicle_index, 'rightPrecedingId'])].index.values.astype(int)[0]
				v_nei += abs(df.at[neighbour_index, 'xVelocity'])
			if  temp.at[vehicle_index, 'rightAlongsideId'] != 0:
				neighbour_index = temp[(temp['id'] == temp.at[vehicle_index, 'rightAlongsideId'])].index.values.astype(int)[0]
				v_nei += abs(df.at[neighbour_index, 'xVelocity'])
			if  temp.at[vehicle_index, 'rightFollowingId'] != 0:
				neighbour_index = temp[(temp['id'] == temp.at[vehicle_index, 'rightFollowingId'])].index.values.astype(int)[0]
				v_nei += abs(df.at[neighbour_index, 'xVelocity'])
			v_nei = v_nei/count_neighbours
			df.at[vehicle_index, 'V_nei'] = v_nei
		#print('Done:'+str(frame))
		if frame > 300:
			break
	return df


def Extract_VelAvg(df):
	'''This function computes the moving average of the veoclity of the car'''
	df['V_avg'] = 0
	for vehicle in df['id'].unique():
		temp = df[df['id'] == vehicle]
		frame_counter = 1
		v_avg = 0
		for frame in temp['frame']:
			frame_index = temp[(temp['frame'] == frame)].index.values.astype(int)[0]
			v_avg = v_avg*(frame_counter-1) + df.loc[frame_index, 'xVelocity']
			v_avg /= frame_counter
			frame_counter += 1
			df.at[frame_index, 'V_avg'] = v_avg
		print(vehicle)
		#print('Done:'+str(frame))
	return df

def Detect_lanechange(df):
	'''This function takes the raw data and creates and identifier at the point where lane change 
	happened in a new column'''
	df['lane_change'] = -1
	for vehicle in df['id'].unique():
		temp = df[(df['id'] == vehicle)]
		frame_initial = list(temp['frame'])[0]
		vehicle_index = temp[(temp['frame'] == frame_initial)].index.values.astype(int)[0]
		old_laneid = temp.loc[vehicle_index, 'laneId']
		for frame in temp['frame']:
			vehicle_index = temp[(temp['frame'] == frame)].index.values.astype(int)[0]
			curr_lane_id = temp.loc[vehicle_index, 'laneId']
			if curr_lane_id != old_laneid:
				df.loc[vehicle_index, 'lane_change'] = 1
			old_laneid = curr_lane_id
	return df


def get_LaneChangers(df):
	'''This function takes the raw data and creates two data frames: one each for lane changers
	and non lane changers'''
	lane_changers = []
	df_laneChangers = pd.DataFrame()
	for vehicle in df['id'].unique():
		temp = df[df['id']==vehicle]
		if len(set(temp['laneId']))>1:
			lane_changers.append(vehicle)
	df_laneChangers = df[df['id'].isin(lane_changers)]
	df_nonlaneChangers = df[~df['id'].isin(lane_changers)]
	return df_laneChangers, df_nonlaneChangers            

def Get_Centerline(df_meta):
	'''Function to extract the position of the center lines of the lanes from the meta file of the track 
	recording'''
	marks_upper = df_meta['upperLaneMarkings']
	marks_lower = df_meta['lowerLaneMarkings']
	l_upper = str(marks_upper[0]).split(';')
	lane_marks_west = [float(l_upper[i]) for i in range(0, len(l_upper))]
	l_lower = str(marks_lower[0]).split(';')
	lane_marks_east = [float(l_lower[i]) for i in range(0, len(l_lower))]
	center_line_west = []
	center_line_east = []
	for i in range(0, len(lane_marks_west)-1):
		center_line_west = np.append(center_line_west,(lane_marks_west[i] +0.5*(lane_marks_west[i+1] - lane_marks_west[i])))
	for i in range(0, len(lane_marks_east)-1):
		center_line_east = np.append(center_line_east,(lane_marks_east[i] +0.5*(lane_marks_east[i+1] - lane_marks_east[i])))
	return list([list(center_line_west), list(center_line_east)]), list([lane_marks_west, lane_marks_east])


def LaneDrift_metric(df, df_meta):
	''' Function to estimate the lane drift. The drift is calculated by measuring the lateral displacement from 
   the center line of the respective lane of trajectory
   Function to estimate the lane drift. The drift is calculated by measuring the lateral displacement from 
   the center line of the respective lane of trajectory'''

	param_k = 3 # time for which vehicle should stay in a lane to be eligible for a lane change
	#lane_id = list(map(int, list(set(df['laneId']))))
	center_line, lane_position = Get_Centerline(df_meta)
	df = Calculate_center(df)
	df['lane_drift'] = 0
	for k in range(0, len(center_line)):
		for i in range(0, len(center_line[k])):
			df['lane_drift'] = df.apply(lambda x: (x['y_center']-center_line[k][i]) if ((x['y_center'] > lane_position[k][i]) & (x['y_center'] < lane_position[k][i+1] ))
									else x['lane_drift'],axis=1)
	return df


def Estimate_drift_gradient(df):
	'''This function is used to calculated the derivative of lane drift'''
	df['derivative_drift'] = 0
	col = []
	for vehicle in df['id'].unique():
		temp = df[df['id']==vehicle]
		temp['derivative_drift'] = np.gradient(temp['lane_drift'])
		col = np.append(col, temp['derivative_drift'])
	df['derivative_drift'] = list(col)
	return df

def Extract_Surroundings(df, meta, surroundinglist, paramlist):
	'''Following function computes the average of the Velocities of the neighbours of the ego car (maximum possible 8)
	at each frame intstance. It is very inefficient due to number of if conidtionals. Will try to make it more
	efficient hopefully.
	Things to do
	x, y's, v and Acc wrt to the ego vehicle lane: Done
	centre the a and ys (NOT top left corner !!!): Done
	Handle categorical variables: Done for while x, v = 100, rest = 0
	Add lane presence: Done
	Add if lanes to the right/ left is present or not: Done
	Vehicle type: not available
	'''
	complete = []
	lane_markings = Get_Centerline(meta)[1]
	surrounding = surroundinglist
	vehicles = list(df['id'].unique())
	for vehicle in vehicles:
		d = initialize_metrics(surrounding, paramlist)
		df_ego = df[df['id']==vehicle]
		for frame in df_ego['frame'].unique():
			temp_frame = df[df['frame'] == frame]
			temp_ego = temp_frame[temp_frame['id']==vehicle]
			
			vehicle_index = temp_ego.index.values.astype(int)[0]
			for member in surrounding:
				if member == 'ego':
					for param in paramlist[3:-1]: #ignore presence, x and y for ego vehicle
						d[member][param].append(temp_frame.at[vehicle_index, param])
					d[member][paramlist[-1]].append(CheckLane(temp_frame.at[vehicle_index, paramlist[-1]], 
															  lane_markings))
				else:
					if  temp_frame.at[vehicle_index, member] != 0:
						neighbour_index = temp_frame[(temp_frame['id'] == \
													  temp_frame.at[vehicle_index, member])].index.values.astype(int)[0]
						d[member][paramlist[0]].append(1)
						for param in paramlist[1:-2]:
							param_value = temp_frame.at[neighbour_index, param] - temp_frame.at[vehicle_index, param]
							d[member][param].append(param_value)
					else:
						d[member][paramlist[0]].append(0)
						for param in paramlist[1:-2]:
							if param in(['x_center', 'y_center', 'xVelocity', 'yVelocity']):
								d[member][param].append(100)
							else:
								d[member][param].append(0)
		temp_vehicle = pd.DataFrame()
		for member in surrounding:
			if member == 'ego':
				for param in paramlist[3:]:
					temp_vehicle[str(member+'_'+param)] = d[member][param]
			else:
				for param in paramlist[:-2]:
					temp_vehicle[str(member+'_'+param)] = d[member][param]
		complete.append(temp_vehicle)
		sys.stdout.write("\rWorking on vehicle_df %d" % vehicle)
		sys.stdout.flush()
		if vehicle >5:
			break
	super_x = pd.concat(complete, axis=0)
	print('done')
	return super_x

def CheckLane(y, lane_markings):
	'''Function to check the availability of lane of left, right or both
	Accordingly this function assigns -1 or 1 if lane is not available to right or left respectively
	else it assigns a value of 0 if a lane is available on either side of the ego lane'''
	if y == 2 or y == len(lane_markings[1])*2:
		return 1
	elif y == len(lane_markings[1]) or y==len(lane_markings[1])+2:
		return -1
	else:
		return 0

def initialize_metrics(surroundinglist, parameterlist):
	d = {}
	for member in surroundinglist:
		d[member]={}            
		if member == 'ego':
			for parameter in parameterlist[3:]: #ignore presence, x and y for ego vehicle
				d[member][parameter] = []
		else:
			for parameter in parameterlist[:-2]:
				d[member][parameter] = []
	return d

def parallelize_extract_mttc(track, frame):
	'''parallelized function for same task'''
	df = track[track['frame']==frame]
	df['mttc'] = np.nan
	temp = df
	for vehicle in temp[temp['frame'] == frame]['id']:
		vehicle_index = temp[(temp['frame'] == frame) & (temp['id'] == vehicle)].index.values.astype(int)[0]
		if  temp.at[vehicle_index, 'precedingId'] != 0:
			precedingvehicle_index = temp[(temp['id'] == temp.at[vehicle_index, 'precedingId'])].index.values.astype(int)[0]
			if df.at[vehicle_index, 'xVelocity'] < 0:
				d_gap = abs(df.at[precedingvehicle_index, 'x'] - df.at[vehicle_index, 'x']) - df.at[precedingvehicle_index, 'width']
				v_sub = -df.at[vehicle_index, 'xVelocity']
				v_pre = -df.at[precedingvehicle_index, 'xVelocity']
				a_sub = -df.at[vehicle_index, 'xAcceleration']
				a_pre = -df.at[precedingvehicle_index, 'xAcceleration']
			else:
				d_gap = abs(df.at[vehicle_index, 'x'] - df.at[precedingvehicle_index, 'x']) - df.at[vehicle_index, 'width'] 
				v_sub = df.at[vehicle_index, 'xVelocity']
				v_pre = df.at[precedingvehicle_index, 'xVelocity']
				a_sub = df.at[vehicle_index, 'xAcceleration']
				a_pre = df.at[precedingvehicle_index, 'xAcceleration']



			df.at[vehicle_index, 'mttc'] = mttc_formula(v_sub, v_pre, a_sub, a_pre, d_gap)
	return df

def mttc_formula(v_sub, v_pre, a_sub, a_pre, d_gap):
	'''simple function to derive modified time to collision'''
	if a_pre!=a_sub:
		t_1 = (-(v_sub-v_pre) + np.sqrt((v_sub-v_pre)**2 + 2*(a_sub-a_pre)*d_gap))/(a_sub-a_pre)
		t_2 = (-(v_sub-v_pre) - np.sqrt((v_sub-v_pre)**2 + 2*(a_sub-a_pre)*d_gap))/(a_sub-a_pre)
		if t_1*t_2>0:
			mttc = min(t_1, t_2)
		else:
			mttc = max(t_1, t_2)
	else:
		if v_sub!=v_pre:
			mttc = d_gap/(v_sub-v_pre)
		else:
			mttc = 99999
	return np.round(mttc, 3)