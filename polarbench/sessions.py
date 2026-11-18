from __future__ import annotations

from copy import deepcopy

from .domain import ConflictError, InputError, MAX_SESSIONS, MissingError, identifier, label, note, now, object_value
from .settings import parse_settings
from .store import fingerprint, new_id, source_identity


def get_session(record, session_id):
    identifier(session_id, 'Saved view')
    for item in record['sessions']:
        if item['id'] == session_id:
            return item
    raise MissingError('Saved view could not be found.')


def _source_snapshot(record):
    return {
        'mask': deepcopy(record['mask']),
        'references': deepcopy(record['references']),
        'regions': deepcopy(record['regions']),
        'profiles': deepcopy(record['profiles']),
        'scale': deepcopy(record.get('scale')),
    }


def add_session(record, raw):
    if len(record['sessions']) >= MAX_SESSIONS:
        raise ConflictError(f'A capture can hold at most {MAX_SESSIONS} saved views.')
    data = object_value(raw, 'Saved view')
    processing, view = parse_settings(data.get('state', {}), record)
    item = {
        'id': new_id(),
        'name': label(data.get('name'), 'Saved view name'),
        'note': note(data.get('note', ''), 'Saved view note'),
        'saved_at': now(),
        'capture_name': record['name'],
        'capture_revision': record['revision'],
        'source_identity': fingerprint(source_identity(record)),
        'state': {'processing': processing.as_dict(), 'view': view.as_dict()},
        '_source_snapshot': _source_snapshot(record),
    }
    record['sessions'].append(item)
    return item


def session_list(record):
    active = fingerprint(source_identity(record))
    return [session_overview(session, active) for session in record['sessions']]


def session_overview(item, active):
    data = {k: deepcopy(v) for k, v in item.items() if k != '_source_snapshot'}
    data['current_sources_match'] = item['source_identity'] == active
    return data


def replay(record, session_id):
    item = get_session(record, session_id)
    snapshots = item.get('_source_snapshot')
    if snapshots is None:
        raise ConflictError('Source snapshot for this saved view is unavailable.')
    replayable = dict(record, **snapshots, defaults=item['state'], name=item['capture_name'])
    if item['source_identity'] != fingerprint(source_identity(replayable)):
        raise ConflictError('Source fingerprint for this saved view no longer matches.')
    return replayable, item['state']


def rename_session(record, session_id, raw):
    item = get_session(record, session_id)
    data = object_value(raw, 'Saved view details')
    item['name'] = label(data.get('name', item['name']), 'Saved view name')
    item['note'] = note(data.get('note', item['note']), 'Saved view note')
    return item


def remove_session(record, session_id):
    record['sessions'].remove(get_session(record, session_id))


def apply_saved_view(record, session_id):
    item = get_session(record, session_id)
    if item['source_identity'] != fingerprint(source_identity(record)):
        raise InputError('This view was saved with different source references. Open its replay instead.')
    record['defaults'] = deepcopy(item['state'])
    return record['defaults']
