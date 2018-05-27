import numpy as np
import os
import csv

class DataLogger(object):
    def __init__(self, parent_dir, max_to_keep=100, file_name='worker_data.csv'):
        self.path = parent_dir + '/' + file_name
        self.max_to_keep = max_to_keep
        self._image_dir = parent_dir + '/' + 'expert_images/'
        self._image_index = 0
        self.buffer_size = 0
        self._data = []

    def store_data(self, worker_id, ep_num, reward):        
        ep_summary = {'worker_id':worker_id, 'ep_number':ep_num, 'reward':reward}
        self._data.append(ep_summary)
        self.buffer_size +=1
        if self.buffer_size > self.max_to_keep:
            self._write_to_csv()
            self._reset_buffer()

    def _reset_buffer(self):
        self.buffer_size = 0
        self._data = []

    def _write_to_new_csv(self):
        with open(self.path, 'w') as csv_file:
            field_names = self._data[0].keys()
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
            writer.writeheader()
            for obs in self._data:
                writer.writerow(obs)

    def _append_to_csv(self):
        with open(self.path, 'a') as csv_file:
            field_names = self._data[0].keys()
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
            for obs in self._data:
                writer.writerow(obs)

    def _write_to_csv(self):
        if os.path.exists(self.path):
            self._append_to_csv()
        else:
            self._write_to_new_csv()