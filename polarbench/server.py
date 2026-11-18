from __future__ import annotations

import os
import threading
from pathlib import Path

from flask import Flask, request, send_from_directory
from waitress import serve

from polarbench.api_capture import api as capture_api
from polarbench.api_inspection import api as inspection_api
from polarbench.api_tools import api as tools_api
from polarbench.capture_zip import MAX_UNPACKED
from polarbench.http_support import install_error_responses
from polarbench.store import ProjectStore
from polarbench.workbench import Workbench

ROOT = Path(__file__).resolve().parent.parent


def create_app(data_dir=None):
    app = Flask(__name__, static_folder=str(ROOT / 'static'), static_url_path='/static')
    app.config['MAX_CONTENT_LENGTH'] = MAX_UNPACKED + 256 * 1024
    app.config['MAX_FORM_PARTS'] = 24
    configured = data_dir or os.environ.get('POLARBENCH_DATA_DIR') or str(ROOT / '.local-data')
    app.config['BENCH'] = Workbench(ProjectStore(Path(configured)))
    app.register_blueprint(capture_api)
    app.register_blueprint(inspection_api)
    app.register_blueprint(tools_api)
    install_error_responses(app)
    data_gate = threading.BoundedSemaphore(value=2)

    @app.before_request
    def resource_budget():
        if not request.path.startswith('/api/'):
            return
        expensive_reads = ('/preview', '/evidence', '/source-package')
        if request.method in ('HEAD', 'OPTIONS') or (request.method == 'GET' and not request.path.endswith(expensive_reads)):
            return
        from flask import g
        from polarbench.domain import ConflictError
        if not data_gate.acquire(timeout=12):
            raise ConflictError('The image processor is busy. Wait for the active operation to finish.')
        g.polarbench_acquired = True

    @app.teardown_request
    def release_budget(error):
        from flask import g
        if g.pop('polarbench_acquired', False):
            data_gate.release()

    @app.after_request
    def local_response_policy(result):
        result.headers['X-Content-Type-Options'] = 'nosniff'
        result.headers['Referrer-Policy'] = 'no-referrer'
        if request.path.startswith('/api/') and 'Cache-Control' not in result.headers:
            result.headers['Cache-Control'] = 'no-store'
        return result

    @app.get('/')
    def index():
        return send_from_directory(app.static_folder, 'index.html')

    return app


def main():
    port = int(os.environ.get('PORT', '18943'))
    host = os.environ.get('POLARBENCH_HOST', '127.0.0.1')
    print(f'PolarBench is listening on http://{host}:{port}', flush=True)
    serve(create_app(), host=host, port=port, threads=5, max_request_body_size=MAX_UNPACKED + 256 * 1024)
