import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from shapely.geometry import Polygon, mapping
from geojson import Feature, FeatureCollection, dump
import os


# Processes all input files in the input directory.
def calculate_aoi(data, input_dir='input', output_dir='output'):
    coords = get_coordinates(data)
    df2 = compute_distances(coords)
    # plot_distances(df2)
    df3 = filter_distances(df2, 0.5)
    # plot_distances(df3)
    eps = compute_eps_values(df3)
    silhouette_list = compute_silhouette_scores(eps, coords)
    max_silhouette_avg, best_eps = find_best_eps(eps, silhouette_list)
    labels, n_clusters = cluster_data(coords, best_eps, min_samples=2)
    polygons = create_polygons(coords, labels, n_clusters)
    merged_polygon = merge_polygons(polygons)
    result = generate_output_file(data, merged_polygon, output_dir)
    return result


# Extracts the coordinates (latitude and longitude) from the data.
def get_coordinates(data):
    coords = [(loc[0], loc[1]) for loc in data['locations']]
    coords = np.radians(coords)
    return coords


# Computes the distances between the coordinates.
def compute_distances(coords):
    df2 = pd.DataFrame(columns=['index', 'distance'])
    for i in range(len(coords) - 1):
        dist = np.linalg.norm(np.array(coords[i]) - np.array(coords[i + 1]))
        df2 = df2._append({'index': str(i), 'distance': dist}, ignore_index=True)
    df2 = df2.sort_values(by=['distance'])
    return df2


# Creates a scatter plot of the distances.
def plot_distances(df):
    plt.scatter(df['index'], df['distance'])
    plt.show()


# Filters the distances based on a maximum distance.
def filter_distances(df, max_distance):
    df_filtered = df[df['distance'] < max_distance]
    df_filtered = df_filtered.sort_values(by=['distance'])
    return df_filtered


# Computes the eps values for DBSCAN based on the filtered distances.
def compute_eps_values(df):
    eps = df['distance'].values.tolist()
    return eps


# Computes the silhouette scores for the different eps values.
def compute_silhouette_scores(eps, coords):
    silhouette_list = []
    for i in eps:
        db = DBSCAN(eps=i, min_samples=2).fit(coords)
        labels = db.labels_

        if all(label == labels[0] for label in labels):
            silhouette_avg = -1
        else:
            silhouette_avg = silhouette_score(coords, labels)

        silhouette_list.append(silhouette_avg)

    return silhouette_list


# Finds the best eps value based on the maximum silhouette score.
def find_best_eps(eps, silhouette_list):
    max_silhouette_avg = max(silhouette_list)
    best_eps = eps[silhouette_list.index(max_silhouette_avg)]
    return max_silhouette_avg, best_eps


# Performs data clustering with DBSCAN.
def cluster_data(coords, epsilon, min_samples):
    db = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(coords)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, n_clusters


# Creates polygons for each cluster.
def create_polygons(coords, labels, n_clusters):
    polygons = []
    for i in range(n_clusters):
        cluster_coords = coords[labels == i]
        center = np.mean(cluster_coords, axis=0)
        distances = np.linalg.norm(cluster_coords - center, axis=1)
        radius = np.max(distances)
        center = np.degrees(center)
        radius = np.degrees(radius)
        lat, lon = center
        lat_offset = radius
        lon_offset = radius / np.cos(np.radians(lat))
        lat1, lon1 = lat - lat_offset, lon - lon_offset
        lat2, lon2 = lat + lat_offset, lon + lon_offset
        polygon = Polygon([(lon1, lat1), (lon1, lat2), (lon2, lat2), (lon2, lat1)])
        polygons.append(polygon)
    return polygons


# Merges overlapping polygons.
def merge_polygons(polygons):
    merged_polygon = polygons[0]
    for polygon in polygons[1:]:
        merged_polygon = merged_polygon.union(polygon)
    return merged_polygon


# Generates the GeoJSON output file.
def generate_output_file(data, merged_polygon, output_dir):
    event_id = data['id']
    output_files = [f for f in os.listdir(output_dir) if f.startswith(f"{event_id}_AoiID_")]
    new_aoi_id = len(output_files) + 1
    properties = data.copy()
    properties['AoiID'] = new_aoi_id
    feature = Feature(geometry=mapping(merged_polygon), properties=properties)
    feature_collection = FeatureCollection([feature])
    output_file_name = os.path.join(output_dir, f"{event_id}_AoiID_{new_aoi_id}.geojson")
    with open(output_file_name, 'w') as f:
        dump(feature_collection, f)
    return feature_collection


# Reads the JSON input file and returns the data.
def read_input_file(input_file_path):
    with open(input_file_path) as f:
        data = json.load(f)
    return data
