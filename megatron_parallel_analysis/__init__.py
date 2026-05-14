__all__ = [
    "MegatronTraceRegressionAnalyzer",
    "MegatronTraceRegressionConfig",
    "MegatronTraceRegressionReport",
    "RegressionClassification",
]


def __getattr__(name):
    if name in __all__:
        from megatron_parallel_analysis.trace_regression_analysis import (
            MegatronTraceRegressionAnalyzer,
            MegatronTraceRegressionConfig,
            MegatronTraceRegressionReport,
            RegressionClassification,
        )

        exports = {
            "MegatronTraceRegressionAnalyzer": MegatronTraceRegressionAnalyzer,
            "MegatronTraceRegressionConfig": MegatronTraceRegressionConfig,
            "MegatronTraceRegressionReport": MegatronTraceRegressionReport,
            "RegressionClassification": RegressionClassification,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
