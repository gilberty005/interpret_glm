import sqlite3
import pickle
import os.path as op
# from diskcache import Cache
import warnings
import io
import pickletools
'''
Copyright 2016-2022 Grant Jenks

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License.
https://github.com/grantjenks/python-diskcache
'''

MODE_NONE = 0
MODE_RAW = 1
MODE_BINARY = 2
MODE_TEXT = 3
MODE_PICKLE = 4


class ReadDisk:
    """Cache key and value serialization for SQLite database and files."""

    def __init__(self, directory, min_file_size=0, pickle_protocol=0):
        """Initialize disk instance.

        :param str directory: directory path
        :param int min_file_size: minimum size for file use
        :param int pickle_protocol: pickle protocol for serialization

        """
        self._directory = directory
        self.min_file_size = min_file_size
        self.pickle_protocol = pickle_protocol
        
    def put(self, key):
        """Convert `key` to fields key and raw for Cache table.

        :param key: key to convert
        :return: (database key, raw boolean) pair

        """
        # pylint: disable=unidiomatic-typecheck
        type_key = type(key)

        if type_key is bytes:
            return sqlite3.Binary(key), True
        elif (
            (type_key is str)
            or (
                type_key is int
                and -9223372036854775808 <= key <= 9223372036854775807
            )
            or (type_key is float)
        ):
            return key, True
        else:
            data = pickle.dumps(key, protocol=self.pickle_protocol)
            result = pickletools.optimize(data)
            return sqlite3.Binary(result), False
            
    def fetch(self, mode, filename, value, read):
        """Convert fields `mode`, `filename`, and `value` from Cache table to
        value.

        :param int mode: value mode raw, binary, text, or pickle
        :param str filename: filename of corresponding value
        :param value: database value
        :param bool read: when True, return an open file handle
        :return: corresponding Python value
        :raises: IOError if the value cannot be read

        """
        # pylint: disable=unidiomatic-typecheck,consider-using-with
        if mode == MODE_RAW:
            return bytes(value) if type(value) is sqlite3.Binary else value
        elif mode == MODE_BINARY:
            if read:
                return open(op.join(self._directory, filename), 'rb')
            else:
                with open(op.join(self._directory, filename), 'rb') as reader:
                    return reader.read()
        elif mode == MODE_TEXT:
            full_path = op.join(self._directory, filename)
            with open(full_path, 'r', encoding='UTF-8') as reader:
                return reader.read()
        elif mode == MODE_PICKLE:
            if value is None:
                with open(op.join(self._directory, filename), 'rb') as reader:
                    return pickle.load(reader)
            else:
                return pickle.load(io.BytesIO(value))

class ReadCache:
    def __init__(self, directory):
        self._disk = ReadDisk(directory)
        self.db = op.join(directory, 'cache.db')
        self.con = None
        
    @property
    def _sql(self):
        return self._con.execute

    @property
    def _con(self):
        if self.con is None:
            self.con = sqlite3.connect(
                f'file:{self.db}?immutable=1', uri=True, #mode=ro must have wal and shm
                isolation_level=None,
                check_same_thread=False,
            )
            self.con.execute('pragma synchronous=off')
            self.con.execute('pragma journal_mode=off')  # WAL can be ro, off pair with immutable
            self.con.execute('pragma query_only=ON')
        return self.con
        
    def get(
        self,
        key,
        default=None,
        read=False,
        expire_time=False,
        tag=False,
        retry=False,
    ):
        """Retrieve value from cache. If `key` is missing, return `default`.

        Raises :exc:`Timeout` error when database timeout occurs and `retry` is
        `False` (default).

        :param key: key for item
        :param default: value to return if key is missing (default None)
        :param bool read: if True, return file handle to value
            (default False)
        :param bool expire_time: if True, return expire_time in tuple
            (default False)
        :param bool tag: if True, return tag in tuple (default False)
        :param bool retry: retry if database timeout occurs (default False)
        :return: value for item or default if key not found
        :raises Timeout: if database timeout occurs

        """
        db_key, raw = self._disk.put(key)
        # update_column = EVICTION_POLICY[self.eviction_policy]['get']
        select = (
            'SELECT rowid, expire_time, tag, mode, filename, value'
            ' FROM Cache WHERE key = ? AND raw = ?'
            # ' AND (expire_time IS NULL OR expire_time > ?)'
        )

        rows = self._sql(select, (db_key, raw)).fetchall()

        if not rows:
            warnings.warn(f"Fail to fetch {key} in {self.db}")
            return default

        ((rowid, _, db_tag, mode, filename, db_value),) = rows

        try:
            value = self._disk.fetch(mode, filename, db_value, read)
        except IOError:
            warnings.warn(f"Cannot find contig{key} in {self.db}: "+
                f"mode {mode} filename {filename} db_value {db_value}")
            return default

        return value

    def __getitem__(self, key):
        return self.get(key)


    def close(self):
        if self.con is not None:
            self.con.close()
    
    def __del__(self):
        if self.con is not None:
            self.con.close()

