from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import yaml
import json
import time

import psycopg2

DB_CONF_FILE = 'tfdragonn_dbconf.yaml'
TABLE = 'tfdragonn_v1_test'
_DB_INIT = False

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

def _get_connection():
    with open(DB_CONF_FILE, 'r') as fp:
        return psycopg2.connect(**yaml.load(fp))


def _get_cursor(connection):
    return connection.cursor()


def _db_init():
    if not os.path.isfile(DB_CONF_FILE):
        raise FileNotFoundError(
            'DB config file {} not found.'.format(DB_CONF_FILE))
    # Check the table actually exists
    with _get_connection() as connection:
        with _get_cursor(connection) as cursor:
            cursor.execute(
                "select exists(select * from information_schema.tables"
                " where table_name='{}')".format(TABLE))
            table_exists = cursor.fetchone()[0]
            if not table_exists:
                raise ConnectionError('Database table {} not found'.format(TABLE))
    global _DB_INIT
    _DB_INIT = True


def get_all_runs():
    with _get_connection() as connection:
        with _get_cursor(connection) as cursor:
            cursor.execute("select * from {};".format(TABLE))
            results = cursor.fetchall()
    return results


def add_run(run_id, data_config_file_path, interval_config_file_path,
            model_config_file_path, log_directory, metadata={}):
    if not _DB_INIT:
        _db_init()
    metadata['time'] = time.strftime("%Y-%m-%d:::%H:%M:%S")
    metadata = json.dumps(metadata)  # json serialize
    assert isinstance(run_id, str)
    with _get_connection() as connection:
        with _get_cursor(connection) as cursor:
            cursor.execute("INSERT INTO {} (run_id, data_config_file_path, interval_config_file_path, model_config_file_path, log_directory, metadata) VALUES (%s, %s, %s, %s, %s, %s)".format(TABLE),
                           (log_directory, data_config_file_path, interval_config_file_path,
                            model_config_file_path, log_directory, metadata))
            connection.commit()
