import os
import logging
from typing import Dict, Any, Optional

from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from dotenv import load_dotenv

load_dotenv()

def setup_telemetry(service_name: str = "fastapi-service", 
                   connection_string: str = None,
                   log_level: int = logging.INFO) -> trace.Tracer:
    """
    Simple telemetry setup for Azure Application Insights with tracing only
    
    Args:
        service_name: Name of the service for tracing
        connection_string: Azure Application Insights connection string
        log_level: Logging level
    
    Returns:
        OpenTelemetry Tracer instance
    """
    
    # Get connection string from environment if not provided
    conn_string = connection_string or os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    
    if not conn_string:
        raise ValueError("APPLICATIONINSIGHTS_CONNECTION_STRING environment variable is required")
    
    # Setup basic logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress Azure SDK logs to reduce noise
    azure_loggers = [
        'azure.core.pipeline.policies.http_logging_policy',
        'azure.monitor.opentelemetry.exporter',
        'azure.core',
        'azure.monitor',
        'azure',
        'opentelemetry'
    ]
    
    for logger_name in azure_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Configure Azure Monitor
    configure_azure_monitor(
        connection_string=conn_string,
        enable_live_metrics=False,
    )
    
    # Instrument logging and requests
    LoggingInstrumentor().instrument()
    RequestsInstrumentor().instrument()
    
    # Get and return tracer
    tracer = trace.get_tracer(service_name, "1.0.0")
    
    logger = logging.getLogger(service_name)
    logger.info("Telemetry setup completed for service: %s", service_name)
    
    return tracer


def instrument_fastapi_app(app):
    """
    Instrument FastAPI application with OpenTelemetry
    
    Args:
        app: FastAPI application instance
    """
    FastAPIInstrumentor.instrument_app(app)


def log_with_custom_dimensions(logger: logging.Logger, level: int, message: str, custom_dims: Dict[str, Any]):
    """
    Log with custom dimensions for Application Insights
    
    Args:
        logger: Logger instance
        level: Logging level
        message: Log message
        custom_dims: Custom dimensions dictionary
    """
    logger.log(level, message, extra={
        "custom_dimensions": custom_dims
    })