# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import datetime
import json
import logging
import time
from collections import defaultdict, deque
from pathlib import Path

import psutil
import torch

from ..dist_comm import is_distributed_enabled

logger = logging.getLogger(__name__)


class MetricLogger:
    def __init__(self, delimiter: str = "  ", output_file: str | Path | None = None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        if isinstance(output_file, str):
            output_file = Path(output_file)
        self.output_file = output_file

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter!s}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        if not torch.distributed.is_initialized():
            return
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def dump_in_output_file(self, iteration, iter_time, data_time):
        if self.output_file is None:
            return
        dict_to_dump = {"iteration": iteration, "iter_time": iter_time, "data_time": data_time}
        dict_to_dump.update({k: v.median for k, v in self.meters.items()})
        with self.output_file.open("a") as f:
            f.write(json.dumps(dict_to_dump) + "\n")

    def log_every(self, iterable, print_freq, header=None, n_iterations=None, start_iteration=0):
        i = start_iteration
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.6f}")
        data_time = SmoothedValue(fmt="{avg:.6f}")

        if n_iterations is None:
            n_iterations = len(iterable)

        space_fmt = ":" + str(len(str(n_iterations))) + "d"

        log_list = [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}", "cpu mem: {cpu_mem:.3f}GB"]
        if torch.cuda.is_available():
            log_list += ["max mem: {memory:.3f}GB"]

        log_msg = self.delimiter.join(log_list)
        if i < n_iterations:
            for obj in iterable:
                data_time.update(time.time() - end)
                yield obj
                iter_time.update(time.time() - end)
                if i % print_freq == 0 or i == n_iterations - 1:
                    self.synchronize_between_processes()
                    self.dump_in_output_file(iteration=i, iter_time=iter_time.avg, data_time=data_time.avg)
                    eta_seconds = iter_time.global_avg * (n_iterations - i)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    if torch.cuda.is_available():
                        logger.info(log_msg.format(i, n_iterations, eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time), cpu_mem=psutil.Process().memory_info().rss / (1024**3), memory=torch.cuda.max_memory_allocated() / (1024**3)))
                    else:
                        logger.info(log_msg.format(i, n_iterations, eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time), cpu_mem=psutil.Process().memory_info().rss / (1024**3)))
                    # 重置GPU内存峰值统计
                    torch.cuda.reset_peak_memory_stats()
                i += 1
                end = time.time()
                if i >= n_iterations:
                    break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f"{header} Total time: {total_time_str} ({total_time / n_iterations:.6f} s / it)")


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, num=1):
        self.deque.append(value)
        self.count += num
        self.total += value * num

    def synchronize_between_processes(self):
        """
        Distributed synchronization of the metric
        Warning: does not synchronize the deque!
        """
        if not is_distributed_enabled():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()  # returns float("nan") when d is empty

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()  # returns float("nan") when d is empty

    @property
    def global_avg(self):
        if self.count == 0:
            return float("nan")
        return self.total / self.count

    @property
    def max(self):
        if len(self.deque) == 0:
            return float("nan")
        return max(self.deque)

    @property
    def value(self):
        if len(self.deque) == 0:
            return float("nan")
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value)
