"""
Graph Image HTTP Server
"""

import os
import sys

# Set Python search path to the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.common import eprint

from flask import Flask, send_file, render_template
app = Flask(__name__)

@app.route('/graph/<datetime>.<ext>')
def send_png(**kwargs):
    import StringIO
    from gengraph import gen_graph

    datetime = kwargs['datetime']
    ext = kwargs['ext']

    errmsg =  '<title>404 Not Found</title>' \
            + '<h1>Not Found</h1>' \
            + '<p>The requested URL was not found on the server.</p>'

    if ext in ['png', 'gif']:
        file = StringIO.StringIO()
        gen_graph(datetime, file, format=ext.upper())

        try:
            return send_file(file,
                attachment_filename='%s.%s' % (datetime, ext),
                mimetype='image/%s' % ext)
        except:
            return errmsg, 404
    else:
        return errmsg, 404

def graphserv(debug=False):
    from lib.config import BeaconConfigParser
    port = BeaconConfigParser().getint('GraphServ', 'port')
    app.run(host='0.0.0.0', port=port)

def task():
    """
    Entry point for Task Keeper
    """
    graphserv()

def main():
    import argparse
    import re
    import sys

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Graph Image HTTP Server')
    parser.add_argument('-d', '--debug',
        action='store_true',
        default=False,
        help='enable debug')
    args = parser.parse_args()

    graphserv(args.debug)

if __name__ == "__main__":
    main()