import os
from flask import Flask, send_from_directory

app = Flask(__name__)


@app.route('/files/')
def serve_files():
    file_dir = os.getcwd() + "/output/"
    files = os.listdir(file_dir)
    response = ""
    for filename in files:
        response += f"<a href='/files/{filename}'>{filename}</a><br>"
    return response


@app.route('/files/<path:filename>')
def serve_file(filename):
    return send_from_directory(os.getcwd() + "/output/", filename)


if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8080)
