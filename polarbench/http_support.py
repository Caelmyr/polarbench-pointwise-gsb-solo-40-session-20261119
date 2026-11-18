from __future__ import annotations

import io
from dataclasses import dataclass

from flask import Response, current_app, jsonify, request, send_file
from werkzeug.exceptions import HTTPException

from .domain import ConflictError, InputError, MissingError, object_value
from .store import public_record


@dataclass(frozen=True)
class CurrentCase:
    bench: object
    record: dict
    options: dict

    def results(self):
        return self.bench.pending(self.record, self.options.get('state'))


def request_object(name='Request', optional=False):
    if not request.data and optional:
        return {}
    return object_value(request.get_json(), name)


def loaded_case(project_id, name='Inspection options'):
    engine = current_app.config['BENCH']
    return CurrentCase(engine, engine.store.get(project_id), request_object(name, optional=True))


def app_bench():
    return current_app.config['BENCH']


def store():
    return app_bench().store


def return_project(record, status=200):
    return jsonify(public_record(record)), status


def send_bytes(content, *, mime, name=None, **headers):
    if name is None:
        result = Response(content, content_type=mime)
    else:
        result = send_file(io.BytesIO(content), mimetype=mime, as_attachment=True, download_name=name, max_age=0)
    result.headers.update(headers)
    result.headers['Cache-Control'] = 'no-store'
    return result


def send_image(image, **headers):
    return send_bytes(image, mime='image/png', **headers)


def send_download(result):
    return send_bytes(result.payload, mime=result.content_type, name=result.name)


def send_csv(payload, name):
    return send_bytes(payload, mime='text/csv; charset=utf-8', name=name)


def install_error_responses(app):
    @app.errorhandler(InputError)
    def input_error(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(MissingError)
    def missing_error(error):
        return jsonify(error=str(error)), 404

    @app.errorhandler(ConflictError)
    def conflict(error):
        return jsonify(error=str(error)), 409

    @app.errorhandler(413)
    def too_large(error):
        return jsonify(error='The request exceeds the 95 MB upload limit.'), 413

    @app.errorhandler(HTTPException)
    def common_error(error):
        if not request.path.startswith('/api/'):
            return error
        return jsonify(error=error.description), error.code
