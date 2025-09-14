"""
Enhanced logging configuration for the performance benchmark system.
Provides structured, configurable logging with multiple handlers and formatters.
"""

import logging
import logging.handlers
import sys
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field


@dataclass
class LoggingConfig:
    """Configuration for the logging system."""
    
    # Basic configuration
    level: str = "INFO"
    format_style: str = "detailed"  # "simple", "detailed", "json"
    
    # Console logging
    console_enabled: bool = True
    console_level: str = "INFO"
    console_color: bool = True
    
    # File logging
    file_enabled: bool = True
    file_level: str = "DEBUG"
    log_dir: Optional[str] = None
    max_file_size: int = 10  # MB
    backup_count: int = 5
    
    # Structured logging
    json_enabled: bool = False
    json_level: str = "INFO"
    
    # Performance logging
    performance_enabled: bool = True
    performance_level: str = "INFO"
    
    # Error logging
    error_file_enabled: bool = True
    error_level: str = "ERROR"
    
    # Logger names to configure
    logger_names: List[str] = field(default_factory=lambda: [
        "benchmark_runner", "experiment_config", "dataset_loader", 
        "som_wrapper", "sklearn_wrapper", "performance_metrics",
        "results_analyzer", "main"
    ])
    
    # Suppress external library logging
    suppress_external: bool = True
    external_level: str = "WARNING"


class ColorFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        if hasattr(record, 'use_color') and record.use_color:
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'message']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


class PerformanceFilter(logging.Filter):
    """Filter for performance-related log messages."""
    
    def filter(self, record):
        # Allow performance-related messages
        performance_keywords = ['performance', 'timing', 'memory', 'benchmark', 'metric']
        message = record.getMessage().lower()
        return any(keyword in message for keyword in performance_keywords)


class BenchmarkLogger:
    """
    Enhanced logging system for the benchmark application.
    Provides multiple handlers, formatters, and configuration options.
    """
    
    def __init__(self, config: LoggingConfig):
        """
        Initialize the logging system.
        
        Args:
            config: Logging configuration
        """
        self.config = config
        self.log_dir = Path(config.log_dir) if config.log_dir else Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this session
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup loggers
        self._setup_formatters()
        self._setup_handlers()
        self._configure_loggers()
        
        # Suppress external libraries if requested
        if config.suppress_external:
            self._suppress_external_logging()
    
    def _setup_formatters(self):
        """Setup different formatters."""
        self.formatters = {}
        
        # Simple formatter
        self.formatters['simple'] = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Detailed formatter
        self.formatters['detailed'] = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Colored formatter for console
        self.formatters['colored'] = ColorFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # JSON formatter
        self.formatters['json'] = JsonFormatter()
        
        # Performance formatter
        self.formatters['performance'] = logging.Formatter(
            '%(asctime)s - PERF - %(name)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    def _setup_handlers(self):
        """Setup different handlers."""
        self.handlers = {}
        
        # Console handler
        if self.config.console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, self.config.console_level))
            
            if self.config.console_color:
                formatter = self.formatters['colored']
                # Add color flag to records
                class ColorFilter(logging.Filter):
                    def filter(self, record):
                        record.use_color = True
                        return True
                console_handler.addFilter(ColorFilter())
            else:
                formatter = self.formatters[self.config.format_style]
            
            console_handler.setFormatter(formatter)
            self.handlers['console'] = console_handler
        
        # Main file handler
        if self.config.file_enabled:
            log_file = self.log_dir / f"benchmark_{self.session_timestamp}.log"
            # Ensure log directory exists
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.max_file_size * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            file_handler.setLevel(getattr(logging, self.config.file_level))
            file_handler.setFormatter(self.formatters['detailed'])
            self.handlers['file'] = file_handler
        
        # Error file handler
        if self.config.error_file_enabled:
            error_file = self.log_dir / f"errors_{self.session_timestamp}.log"
            # Ensure log directory exists
            error_file.parent.mkdir(parents=True, exist_ok=True)
            error_handler = logging.FileHandler(error_file)
            error_handler.setLevel(getattr(logging, self.config.error_level))
            error_handler.setFormatter(self.formatters['detailed'])
            self.handlers['error'] = error_handler
        
        # JSON file handler
        if self.config.json_enabled:
            json_file = self.log_dir / f"structured_{self.session_timestamp}.json"
            # Ensure log directory exists
            json_file.parent.mkdir(parents=True, exist_ok=True)
            json_handler = logging.FileHandler(json_file)
            json_handler.setLevel(getattr(logging, self.config.json_level))
            json_handler.setFormatter(self.formatters['json'])
            self.handlers['json'] = json_handler
        
        # Performance file handler
        if self.config.performance_enabled:
            perf_file = self.log_dir / f"performance_{self.session_timestamp}.log"
            # Ensure log directory exists
            perf_file.parent.mkdir(parents=True, exist_ok=True)
            perf_handler = logging.FileHandler(perf_file)
            perf_handler.setLevel(getattr(logging, self.config.performance_level))
            perf_handler.setFormatter(self.formatters['performance'])
            perf_handler.addFilter(PerformanceFilter())
            self.handlers['performance'] = perf_handler
    
    def _configure_loggers(self):
        """Configure loggers with handlers."""
        # Set root logger level
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.level))
        
        # Configure specific loggers
        for logger_name in self.config.logger_names:
            logger = logging.getLogger(logger_name)
            logger.setLevel(getattr(logging, self.config.level))
            
            # Remove existing handlers to avoid duplicates
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # Add our handlers
            for handler in self.handlers.values():
                logger.addHandler(handler)
            
            # Prevent propagation to avoid duplicate messages
            logger.propagate = False
    
    def _suppress_external_logging(self):
        """Suppress logging from external libraries."""
        external_loggers = [
            'matplotlib', 'PIL', 'urllib3', 'requests', 'sklearn',
            'numpy', 'pandas', 'scipy', 'joblib', 'tensorflow',
            'keras', 'torch', 'torchvision'
        ]
        
        for logger_name in external_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(getattr(logging, self.config.external_level))
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a configured logger.
        
        Args:
            name: Logger name
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)
        
        # Ensure logger is configured
        if not logger.handlers:
            logger.setLevel(getattr(logging, self.config.level))
            for handler in self.handlers.values():
                logger.addHandler(handler)
            logger.propagate = False
        
        return logger
    
    def log_experiment_start(self, experiment_name: str, config_dict: Dict[str, Any]):
        """Log experiment start with configuration."""
        logger = self.get_logger("experiment")
        
        logger.info("="*80)
        logger.info(f"EXPERIMENT START: {experiment_name}")
        logger.info("="*80)
        logger.info(f"Session timestamp: {self.session_timestamp}")
        logger.info(f"Log directory: {self.log_dir}")
        
        # Log configuration
        logger.info("Experiment Configuration:")
        for key, value in config_dict.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("="*80)
    
    def log_experiment_end(self, experiment_name: str, duration: float, success: bool):
        """Log experiment end with summary."""
        logger = self.get_logger("experiment")
        
        logger.info("="*80)
        status = "COMPLETED" if success else "FAILED"
        logger.info(f"EXPERIMENT {status}: {experiment_name}")
        logger.info(f"Total duration: {duration:.2f} seconds")
        logger.info(f"Session timestamp: {self.session_timestamp}")
        logger.info("="*80)
    
    def log_performance_metric(self, metric_name: str, value: float, 
                             algorithm: str = None, dataset: str = None, 
                             run_number: int = None, **kwargs):
        """Log performance metric with structured data."""
        logger = self.get_logger("performance")
        
        # Create structured message
        context = []
        if algorithm:
            context.append(f"algorithm={algorithm}")
        if dataset:
            context.append(f"dataset={dataset}")
        if run_number is not None:
            context.append(f"run={run_number}")
        
        context_str = f"[{', '.join(context)}]" if context else ""
        
        message = f"{metric_name}: {value}"
        if context_str:
            message = f"{context_str} {message}"
        
        # Add extra attributes for JSON logging
        extra = {
            'metric_name': metric_name,
            'metric_value': value,
            'algorithm': algorithm,
            'dataset': dataset,
            'run_number': run_number
        }
        extra.update(kwargs)
        
        logger.info(message, extra=extra)
    
    def log_run_progress(self, current: int, total: int, algorithm: str, 
                        dataset: str, run_number: int = None):
        """Log progress of benchmark runs."""
        logger = self.get_logger("progress")
        
        percentage = (current / total) * 100 if total > 0 else 0
        
        if run_number is not None:
            message = f"Progress: {current}/{total} ({percentage:.1f}%) - {algorithm} on {dataset} (run {run_number})"
        else:
            message = f"Progress: {current}/{total} ({percentage:.1f}%) - {algorithm} on {dataset}"
        
        logger.info(message, extra={
            'progress_current': current,
            'progress_total': total,
            'progress_percentage': percentage,
            'algorithm': algorithm,
            'dataset': dataset,
            'run_number': run_number
        })
    
    def close(self):
        """Close all handlers and cleanup."""
        for handler in self.handlers.values():
            handler.close()


def setup_logging(config: Optional[LoggingConfig] = None, 
                 log_dir: Optional[str] = None,
                 level: str = "INFO") -> BenchmarkLogger:
    """
    Setup the logging system with configuration.
    
    Args:
        config: Logging configuration (if None, creates default)
        log_dir: Log directory (overrides config if provided)
        level: Log level (overrides config if provided)
        
    Returns:
        Configured BenchmarkLogger instance
    """
    if config is None:
        config = LoggingConfig()
    
    # Override with provided parameters
    if log_dir is not None:
        config.log_dir = log_dir
    if level != "INFO":
        config.level = level
        config.console_level = level
    
    return BenchmarkLogger(config)


# Predefined logging configurations
class LoggingTemplates:
    """Predefined logging configuration templates."""
    
    @staticmethod
    def development() -> LoggingConfig:
        """Development logging configuration."""
        return LoggingConfig(
            level="DEBUG",
            console_level="DEBUG",
            file_level="DEBUG",
            console_color=True,
            json_enabled=False,
            performance_enabled=True,
            suppress_external=True
        )
    
    @staticmethod
    def production() -> LoggingConfig:
        """Production logging configuration."""
        return LoggingConfig(
            level="INFO",
            console_level="INFO",
            file_level="DEBUG",
            console_color=False,
            json_enabled=True,
            performance_enabled=True,
            suppress_external=True,
            max_file_size=50,
            backup_count=10
        )
    
    @staticmethod
    def performance_focused() -> LoggingConfig:
        """Performance-focused logging configuration."""
        return LoggingConfig(
            level="INFO",
            console_level="WARNING",
            file_level="INFO",
            console_color=True,
            json_enabled=True,
            performance_enabled=True,
            performance_level="DEBUG",
            suppress_external=True
        )
    
    @staticmethod
    def minimal() -> LoggingConfig:
        """Minimal logging configuration."""
        return LoggingConfig(
            level="WARNING",
            console_level="WARNING",
            file_enabled=False,
            json_enabled=False,
            performance_enabled=False,
            error_file_enabled=True,
            suppress_external=True
        )