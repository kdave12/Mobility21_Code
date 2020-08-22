import math

def interpolation(lat1, lon1, lat2, lon2):
	avg_lat =  (lat1 + lat2)/2
	avg_lon = (lon1 + lon2)/2
	return avg_lat, avg_lon


def produce_points(lat1, lon1, lat2, lon2):

	if (abs(lon2 - lon1) < 0.001):
		return (lat1, lon1)
	else:
		new_lat, new_log = interpolation(lat1, lon1, lat2, lon2)
		new_lat = round(new_lat, 6)
		new_log = round(new_log, 6)
		return [produce_points(lat1, lon1, new_lat, new_log), (new_lat, new_log), 
					produce_points(new_lat, new_log, lat2, lon2)]		


points = produce_points(40.444661, -79.943662, 40.444458, -79.948674)
print(points)