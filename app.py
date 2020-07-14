from Load_Model_Locally import build, initialize

from flask import Flask, request

app = Flask(__name__)


@app.route('/')
def get_res():
    # Getting Input
    url = request.args.get('url')
    # Getting Result
    result = build(url)
    # Returning Result
    return result


if __name__ == '__main__':
    initialize()
    app.run(debug=False)
