# -*- coding: utf-8 -*-

import sys
import os
import common_io
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class OdpsTableDataset(torch.utils.data.IterableDataset):
    def __init__(self, odps_tables, slice_id=0, slice_count=1,
                 selected_cols="", excluded_cols="",
                 capacity=512, num_threads=4):
        self.odps_tables = odps_tables.split(',')
        self.slice_id = slice_id
        self.slice_count = slice_count
        self.capacity = capacity
        self.num_threads = num_threads
        self.selected_cols = selected_cols
        self.excluded_cols = excluded_cols

        self.row_counts = []
        self.start_pos = []
        self.end_pos = []
        self.total_count = 0
        self._parse_odps_tables_meta()

    def _parse_odps_tables_meta(self):
        for odps_table_path in self.odps_tables:
            # 注意，这里的num_threads必须为0
            reader = common_io.table.TableReader(odps_table_path,
                                                 slice_id=self.slice_id,
                                                 slice_count=self.slice_count,
                                                 num_threads=0)
            row_count = reader.get_row_count()
            self.row_counts.append(row_count)
            self.total_count = self.total_count + row_count
            self.start_pos.append(reader.start_pos)
            self.end_pos.append(reader.end_pos)
            reader.close()
        print("total row_count:{}".format(self.total_count))

    def __len__(self):
        return self.total_count

    def _get_slice_range(self, row_count, worker_id, num_workers, baseline=0):
        # div-mod split, each slice data number max diff 1
        size = int(row_count / num_workers)
        split_point = row_count % num_workers
        if worker_id < split_point:
            start = worker_id * (size + 1) + baseline
            end = start + (size + 1)
        else:
            start = split_point * (size + 1) + (worker_id - split_point) * size + baseline
            end = start + size
        return start, end

    def _get_table_reader(self,  worker_id, num_workers):
        if len(self.odps_tables) == 0:
            return None
        odps_table_path = self.odps_tables.pop(0)
        table_row_count = self.row_counts.pop(0)
        start_pos = self.start_pos.pop(0)
        table_start, table_end = self._get_slice_range(table_row_count, worker_id, num_workers, start_pos)
        if table_start - table_end == 0:
            print("table slice start==end, return None!!!")
            if len(self.odps_tables) == 0:
                return None  # 全部table遍历结束
            else:
                return -1

        table_path = "{}?start={}&end={}".format(odps_table_path, table_start, table_end)
        return common_io.table.TableReader(table_path,
                                             num_threads=self.num_threads,
                                             selected_cols=self.selected_cols,
                                             excluded_cols=self.excluded_cols,
                                             capacity=self.capacity)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        def table_data_iterator():
            reader = self._get_table_reader(worker_id, num_workers)
            while True:
                try:
                    data = reader.read(num_records=1, allow_smaller_final_batch=True)[0]
                except common_io.exception.OutOfRangeException:
                    reader.close()
                    while True:
                        reader = self._get_table_reader(worker_id, num_workers)
                        if reader == -1:
                            continue
                        else:
                            break
                    if reader is None:
                        break
                yield data

        return table_data_iterator()
