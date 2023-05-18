import json
import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon, mapping
from geojson import Feature, FeatureCollection, dump

import os

import time

input_dir = 'input'  # Specify the input directory
output_dir = 'output'  # Specify the output directory

while True:
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    for input_file in input_files:
        input_file_path = os.path.join(input_dir, input_file)

        # Read input file
        with open(input_file_path) as f:
            data = json.load(f)

        # Get the latitude and longitude from the locations
        coords = [(loc[0], loc[1]) for loc in data['locations']]

        # Convert the coordinates to radians
        coords = np.radians(coords)

        # Compute clustering with DBSCAN
        epsilon = 0.01  # epsilon in radians

        min_samples = 2

        db = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(coords)

        # Get the cluster labels and number of clusters
        labels = db.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Create a list of polygons for each cluster

        polygons = []
        for i in range(n_clusters):
            # Get all the coordinates in the cluster

            cluster_coords = coords[labels == i]

            # Compute the center of the cluster

            center = np.mean(cluster_coords, axis=0)

            # Compute the radius of the cluster

            distances = np.linalg.norm(cluster_coords - center, axis=1)

            radius = np.max(distances)

            # Convert the center and radius back to degrees

            center = np.degrees(center)

            radius = np.degrees(radius)

            # Create a rectangle around the center point

            lat, lon = center

            lat_offset = radius

            lon_offset = radius / np.cos(np.radians(lat))

            lat1, lon1 = lat - lat_offset, lon - lon_offset

            lat2, lon2 = lat + lat_offset, lon + lon_offset

            polygon = Polygon([(lon1, lat1), (lon1, lat2), (lon2, lat2), (lon2, lat1)])

            polygons.append(polygon)

        # Merge overlapping polygons

        merged_polygon = polygons[0]

        for polygon in polygons[1:]:
            merged_polygon = merged_polygon.union(polygon)

        # Generate a new AoiID

        event_id = data['id']

        output_files = [f for f in os.listdir(output_dir) if f.startswith(f"{event_id}_AoiID_")]

        new_aoi_id = len(output_files) + 1

        # Convert the merged polygon to GeoJSON

        properties = data.copy()

        properties['AoiID'] = new_aoi_id

        feature = Feature(geometry=mapping(merged_polygon), properties=properties)

        feature_collection = FeatureCollection([feature])

        # Format the output file name

        output_file_name = os.path.join(output_dir, f"{event_id}_AoiID_{new_aoi_id}.geojson")

        with open(output_file_name, 'w') as f:

            dump(feature_collection, f)

        # Delay for 12 hours
        time.sleep(10)