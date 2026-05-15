"""计时工具使用示例

该文件展示了如何使用 pytorch_callstack_analysis/utils/timing.py 中的计时工具。

运行方式:
    python example_timing_usage.py
"""

import time
from pytorch_callstack_analysis.utils.timing import (
    TimingTracker,
    get_timer,
    reset_timer,
    time_it,
    measure_time,
    QuickTimer,
)


def example_decorator():
    """示例一：使用装饰器计时"""
    print("\n" + "=" * 70)
    print("示例一：装饰器方式计时")
    print("=" * 70)
    
    @time_it("slow_function")
    def slow_function():
        """模拟耗时操作"""
        time.sleep(1.5)
        return "result"
    
    @time_it("fast_function")
    def fast_function():
        """模拟快速操作"""
        time.sleep(0.3)
        return "result"
    
    # 调用函数
    result1 = slow_function()
    result2 = fast_function()
    
    # 打印摘要
    timer = get_timer()
    print("\n计时摘要:")
    print(timer.summary())


def example_context_manager():
    """示例二：使用上下文管理器计时"""
    print("\n" + "=" * 70)
    print("示例二：上下文管理器方式计时")
    print("=" * 70)
    
    # 重置计时器
    reset_timer()
    timer = get_timer()
    
    # 使用上下文管理器
    with timer.measure("data_loading"):
        print("正在加载数据...")
        time.sleep(1.0)
    
    with timer.measure("data_processing"):
        print("正在处理数据...")
        time.sleep(0.5)
    
    with timer.measure("data_saving"):
        print("正在保存数据...")
        time.sleep(0.3)
    
    # 打印摘要
    print("\n计时摘要:")
    timer.print_summary()


def example_manual_control():
    """示例三：手动控制计时"""
    print("\n" + "=" * 70)
    print("示例三：手动控制计时")
    print("=" * 70)
    
    timer = TimingTracker()
    
    # 手动开始和停止
    timer.start("operation_1")
    time.sleep(0.8)
    elapsed1 = timer.stop("operation_1")
    print(f"操作1耗时: {elapsed1:.4f}s")
    
    timer.start("operation_2")
    time.sleep(1.2)
    elapsed2 = timer.stop("operation_2")
    print(f"操作2耗时: {elapsed2:.4f}s")
    
    # 获取已完成的计时
    print(f"\n从记录中获取: operation_1 = {timer.get_elapsed('operation_1'):.4f}s")
    
    # 打印摘要
    print("\n计时摘要:")
    print(timer.summary())


def example_quick_timer():
    """示例四：使用快速计时器"""
    print("\n" + "=" * 70)
    print("示例四：快速计时器（不依赖全局计时器）")
    print("=" * 70)
    
    # 方式一：上下文管理器
    with QuickTimer("quick_operation"):
        time.sleep(0.5)
    
    # 方式二：手动控制
    timer = QuickTimer("manual_operation")
    timer.start()
    time.sleep(0.7)
    elapsed = timer.stop()
    print(f"手动计时结果: {elapsed:.4f}s")


def example_nested_timing():
    """示例五：嵌套计时"""
    print("\n" + "=" * 70)
    print("示例五：嵌套计时")
    print("=" * 70)
    
    reset_timer()
    timer = get_timer()
    
    with timer.measure("total_process"):
        with timer.measure("step_1"):
            time.sleep(0.3)
        
        with timer.measure("step_2"):
            time.sleep(0.5)
        
        with timer.measure("step_3"):
            time.sleep(0.4)
    
    # 打印摘要（按时间排序）
    print("\n计时摘要（按时间排序）:")
    timer.print_summary(sort_by_time=True)


def example_convenience_function():
    """示例六：使用便捷函数"""
    print("\n" + "=" * 70)
    print("示例六：使用便捷函数 measure_time")
    print("=" * 70)
    
    reset_timer()
    
    # 使用便捷的上下文管理器
    with measure_time("convenience_test"):
        time.sleep(0.6)
    
    # 打印摘要
    print("\n计时摘要:")
    get_timer().print_summary()


def example_to_dict():
    """示例七：转换为字典"""
    print("\n" + "=" * 70)
    print("示例七：转换为字典格式")
    print("=" * 70)
    
    reset_timer()
    timer = get_timer()
    
    with timer.measure("task_a"):
        time.sleep(0.3)
    
    with timer.measure("task_b"):
        time.sleep(0.5)
    
    # 转换为字典
    timing_dict = timer.to_dict()
    print("\n字典格式:")
    for name, elapsed in timing_dict.items():
        print(f"  {name}: {elapsed:.4f}s")


def main():
    """运行所有示例"""
    print("=" * 70)
    print("计时工具使用示例")
    print("=" * 70)
    
    # 运行各个示例
    example_decorator()
    example_context_manager()
    example_manual_control()
    example_quick_timer()
    example_nested_timing()
    example_convenience_function()
    example_to_dict()
    
    print("\n" + "=" * 70)
    print("所有示例完成")
    print("=" * 70)


if __name__ == "__main__":
    main()