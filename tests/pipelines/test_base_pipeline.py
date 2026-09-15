"""
Tests for the base pipeline.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.pipelines.base_pipeline import BasePipeline


class ConcretePipeline(BasePipeline):
    """Concrete implementation used for testing."""

    @property
    def name(self) -> str:
        """Return the pipeline name."""

        return "test_pipeline"

    def run(
        self,
        *args,
        **kwargs,
    ):
        """Return a test result."""

        return "success"


def test_base_pipeline_cannot_be_instantiated() -> None:
    """BasePipeline must remain abstract."""

    with pytest.raises(TypeError):
        BasePipeline()


@patch(
    "src.pipelines.base_pipeline.get_logger",
)
def test_pipeline_initializes_logger(
    mock_get_logger: MagicMock,
) -> None:
    """Pipeline must initialize its logger."""

    logger = MagicMock()
    mock_get_logger.return_value = logger

    pipeline = ConcretePipeline()

    mock_get_logger.assert_called_once()
    assert pipeline.logger is logger


def test_pipeline_name() -> None:
    """Concrete pipeline must expose its name."""

    pipeline = ConcretePipeline()

    assert pipeline.name == "test_pipeline"


def test_pipeline_run() -> None:
    """Concrete pipeline must implement run."""

    pipeline = ConcretePipeline()

    result = pipeline.run()

    assert result == "success"


def test_log_start_returns_start_time() -> None:
    """_log_start must log the start and return a timestamp."""

    pipeline = ConcretePipeline()
    pipeline.logger = MagicMock()

    start_time = pipeline._log_start()

    assert isinstance(start_time, float)

    pipeline.logger.info.assert_called_once_with(
        "começando o pipeline: test_pipeline"
    )


def test_log_finish_logs_completion() -> None:
    """_log_finish must log pipeline completion."""

    pipeline = ConcretePipeline()
    pipeline.logger = MagicMock()

    pipeline._log_finish(
        start_time=0.0,
    )

    pipeline.logger.success.assert_called_once()

    message = pipeline.logger.success.call_args.args[0]

    assert "pipeline 'test_pipeline' finalizado" in message
    assert "segundos" in message


def test_log_failure_logs_exception() -> None:
    """_log_failure must log the pipeline exception."""

    pipeline = ConcretePipeline()
    pipeline.logger = MagicMock()

    exception = ValueError("test error")

    pipeline._log_failure(exception)

    pipeline.logger.exception.assert_called_once_with(
        exception,
    )


def test_pipeline_accepts_arguments() -> None:
    """Concrete pipelines must be able to receive arguments."""

    class ArgumentPipeline(BasePipeline):
        """Pipeline used to test run arguments."""

        @property
        def name(self) -> str:
            return "argument_pipeline"

        def run(self, *args, **kwargs):
            return args, kwargs

    pipeline = ArgumentPipeline()

    args = pipeline.run(
        "DATA",
        123,
        mode="test",
    )

    assert args == (
        ("DATA", 123),
        {"mode": "test"},
    )