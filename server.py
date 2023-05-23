import os
from flask import Flask, send_from_directory, request
import json
import geopandas as gpd
import matplotlib.pyplot as plt
from calculator import calculate_aoi

app = Flask(__name__)


@app.route('/aoi/', methods=['POST'])
def get_input():
    try:
        event_data = json.loads(request.data)
        #    event_data = data['event']
        response = calculate_aoi(event_data)
        return response
    except:
        return "Bad input", 400


@app.route('/files/')
def serve_files():
    file_dir = os.getcwd() + "/output/"
    files = os.listdir(file_dir)
    response = []
    for filename in files:
        with open("./output/" + filename, 'r') as file:
            data = json.load(file)
        data["id"] = filename
        response.append(data)
    return response


@app.route('/files/<path:filename>')
def serve_file(filename):
    plt.rcParams['figure.figsize'] = (20, 10)
    data = gpd.read_file("./output/" + filename)

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    fig, ax = plt.subplots(figsize=(10, 6))
    world.boundary.plot(ax=ax, linewidth=0.8, color='black')
    data.plot(ax=ax, color='blue', alpha=0.5)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

    return send_from_directory("./output/", filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4444)
