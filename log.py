import csv
import os
from datetime import datetime
from pathlib import Path
import datetime

class CheckpointableData:
    def __init__(self):
        self._data = {}

    def __getitem__(self, item):
        return self._data[item]

    def append(self, name, value, verbose=False):
        if name not in self._data:
            self._data[name] = []
        self._data[name].append(value)

class Log:
    def __init__(self, env, path):
        self.file = os.path.join(path, f'{env}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt')

    def write(self, data):
        # print('writing')
        with open(self.file, 'a') as file:
            file.write(data + '\n')
    
class TabularLog:
    def __init__(self, dir, filename):
        self._dir = Path(dir)
        assert self._dir.is_dir()
        self._filename = filename
        self._column_names = None
        self._file = open(self.path, mode=('a' if self.path.exists() else 'w'), newline='')
        self._writer = csv.writer(self._file)

    @property
    def path(self):
        return self._dir/self._filename

    def row(self, row, flush=True):
        if self._column_names is None:
            self._column_names = list(row.keys())
            self._writer.writerow(self._column_names)
        self._writer.writerow([row[col] for col in self._column_names])
        if flush:
            self._file.flush()