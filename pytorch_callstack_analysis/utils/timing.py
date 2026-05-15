"""执行时间测量工具

该模块提供了多种测量代码执行时间的方法：
1. TimingTracker: 全局计时跟踪器，支持多个计时点
2. time_it: 装饰器方式计时
3. measure: 上下文管理器方式计时

使用示例:
    # 方式一：装饰器
    @time_it("function_name")
    def my_function():
        pass
    
    # 方式二：上下文管理器
    timer = get_timer()
    with timer.measure("block_name"):
        # 代码块
        pass
    
    # 方式三：手动控制
    timer = get_timer()
    timer.start("operation")
    # ... 执行操作
    elapsed = timer.stop("operation")
    
    # 打印摘要
    print(timer.summary())
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from contextlib import contextmanager
from functools import wraps


@dataclass
class TimingRecord:
    """计时记录"""
    name: str
    start_time: float
    end_time: Optional[float] = None
    elapsed_time: Optional[float] = None
    
    def __str__(self) -> str:
        if self.elapsed_time is None:
            return f"{self.name}: (in progress)"
        return f"{self.name}: {self.elapsed_time:.4f}s ({self.elapsed_time/60:.2f}min)"


class TimingTracker:
    """全局计时跟踪器
    
    用于跟踪多个代码块的执行时间，并生成摘要报告。
    
    Attributes:
        records: 当前正在进行的计时记录
        completed_records: 已完成的计时记录列表
    
    Example:
        timer = TimingTracker()
        
        # 方式一：手动控制
        timer.start("data_loading")
        load_data()
        elapsed = timer.stop("data_loading")
        
        # 方式二：上下文管理器
        with timer.measure("analysis"):
            analyze()
        
        # 打印摘要
        print(timer.summary())
    """
    
    def __init__(self):
        self.records: Dict[str, TimingRecord] = {}
        self.completed_records: List[TimingRecord] = []
    
    def start(self, name: str) -> None:
        """开始计时
        
        Args:
            name: 计时名称，用于标识这个计时块
        
        Raises:
            ValueError: 如果同名计时已经存在
        """
        if name in self.records:
            raise ValueError(f"Timer '{name}' already exists")
        
        self.records[name] = TimingRecord(
            name=name,
            start_time=time.perf_counter()
        )
    
    def stop(self, name: str) -> float:
        """停止计时
        
        Args:
            name: 计时名称
        
        Returns:
            elapsed_time: 执行时间（秒）
        
        Raises:
            ValueError: 如果计时名称不存在
        """
        if name not in self.records:
            raise ValueError(f"Timer '{name}' not found")
        
        record = self.records[name]
        record.end_time = time.perf_counter()
        record.elapsed_time = record.end_time - record.start_time
        
        self.completed_records.append(record)
        del self.records[name]
        
        return record.elapsed_time
    
    @contextmanager
    def measure(self, name: str, verbose: bool = True):
        """上下文管理器方式计时
        
        Args:
            name: 计时名称
            verbose: 是否立即打印时间
        
        Example:
            with timer.measure("data_processing"):
                process_data()
        """
        self.start(name)
        yield
        elapsed = self.stop(name)
        if verbose:
            print(f"[TIMING] {name}: {elapsed:.4f}s ({elapsed/60:.2f}min)")
    
    def get_elapsed(self, name: str) -> Optional[float]:
        """获取已完成的计时时间
        
        Args:
            name: 计时名称
        
        Returns:
            elapsed_time: 执行时间（秒），如果不存在返回 None
        """
        for record in self.completed_records:
            if record.name == name:
                return record.elapsed_time
        return None
    
    def clear(self) -> None:
        """清除所有计时记录"""
        self.records.clear()
        self.completed_records.clear()
    
    def summary(self, sort_by_time: bool = True) -> str:
        """生成计时摘要
        
        Args:
            sort_by_time: 是否按时间排序
        
        Returns:
            summary_str: 格式化的摘要字符串
        """
        if not self.completed_records:
            return "No timing records"
        
        # 排序
        records = self.completed_records.copy()
        if sort_by_time:
            records.sort(key=lambda r: r.elapsed_time, reverse=True)
        
        # 生成摘要
        lines = [
            "=" * 70,
            "TIMING SUMMARY",
            "=" * 70,
        ]
        
        total_time = 0.0
        for record in records:
            elapsed = record.elapsed_time
            total_time += elapsed
            
            # 格式化输出
            hours = elapsed / 3600
            minutes = elapsed / 60
            seconds = elapsed
            
            if hours >= 1:
                time_str = f"{hours:.2f}h"
            elif minutes >= 1:
                time_str = f"{minutes:.2f}min"
            else:
                time_str = f"{seconds:.2f}s"
            
            lines.append(f"{record.name:50s} {elapsed:10.4f}s ({time_str:>10s})")
        
        lines.extend([
            "-" * 70,
            f"{'TOTAL':50s} {total_time:10.4f}s ({total_time/60:6.2f}min)",
            "=" * 70,
        ])
        
        return "\n".join(lines)
    
    def print_summary(self, sort_by_time: bool = True) -> None:
        """打印计时摘要"""
        print(self.summary(sort_by_time))
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式
        
        Returns:
            timing_dict: {name: elapsed_time} 的字典
        """
        return {
            record.name: record.elapsed_time
            for record in self.completed_records
        }


# 全局计时器实例
_global_timer: Optional[TimingTracker] = None


def get_timer() -> TimingTracker:
    """获取全局计时器
    
    如果全局计时器不存在，会自动创建一个新的。
    
    Returns:
        TimingTracker: 全局计时器实例
    """
    global _global_timer
    if _global_timer is None:
        _global_timer = TimingTracker()
    return _global_timer


def reset_timer() -> None:
    """重置全局计时器
    
    清除所有计时记录，重新开始计时。
    """
    global _global_timer
    _global_timer = TimingTracker()


def time_it(name: str, verbose: bool = True):
    """计时装饰器
    
    用于测量函数执行时间。
    
    Args:
        name: 计时名称
        verbose: 是否打印时间
    
    Example:
        @time_it("data_loading")
        def load_data():
            pass
        
        @time_it("analysis", verbose=False)
        def analyze():
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            timer = get_timer()
            timer.start(name)
            result = func(*args, **kwargs)
            elapsed = timer.stop(name)
            if verbose:
                print(f"[TIMING] {name}: {elapsed:.4f}s ({elapsed/60:.2f}min)")
            return result
        return wrapper
    return decorator


def measure_time(name: str, verbose: bool = True):
    """便捷的上下文管理器
    
    使用全局计时器进行计时。
    
    Args:
        name: 计时名称
        verbose: 是否打印时间
    
    Example:
        with measure_time("data_processing"):
            process_data()
    """
    return get_timer().measure(name, verbose)


# 便捷的快速计时类（不依赖全局计时器）
class QuickTimer:
    """快速计时器
    
    用于简单的计时场景，不依赖全局计时器。
    
    Example:
        with QuickTimer("operation"):
            do_something()
        
        # 或者手动控制
        timer = QuickTimer("operation")
        timer.start()
        do_something()
        elapsed = timer.stop()
    """
    
    def __init__(self, name: str = "unnamed"):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_time: Optional[float] = None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()
    
    def start(self) -> None:
        """开始计时"""
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """停止计时"""
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
        print(f"[TIMING] {self.name}: {self.elapsed_time:.4f}s ({self.elapsed_time/60:.2f}min)")
        return self.elapsed_time
    
    def __str__(self) -> str:
        if self.elapsed_time is None:
            return f"{self.name}: (not completed)"
        return f"{self.name}: {self.elapsed_time:.4f}s"


# 用于 MPI 环境的计时器
class MPITimer:
    """MPI 环境下的计时器
    
    只在 root rank 上打印计时信息。
    
    Example:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        
        timer = MPITimer(comm, name="operation")
        with timer:
            do_something()
    """
    
    def __init__(self, comm, name: str = "unnamed", root: int = 0):
        self.comm = comm
        self.name = name
        self.root = root
        self.rank = comm.Get_rank()
        self.start_time: Optional[float] = None
        self.elapsed_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed_time = time.perf_counter() - self.start_time
        if self.rank == self.root:
            print(f"[TIMING] {self.name}: {self.elapsed_time:.4f}s ({self.elapsed_time/60:.2f}min)")