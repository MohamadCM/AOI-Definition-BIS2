import os
from flask import Flask, send_from_directory
import json
import geopandas as gpd
import matplotlib.pyplot as plt

app = Flask(__name__)


@app.route('/files/')
def serve_files():
    file_dir = os.getcwd() + "/output/"
    files = os.listdir(file_dir)
    response = f"<P1'>Output files</P1><br>"
    for filename in files:
        response += f"<a href='/files/{filename}'>{filename}</a><br>"
    response += f"<hr />"
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
      app.run(host='0.0.0.0', port=8080)
