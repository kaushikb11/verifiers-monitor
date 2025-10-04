"""
Constants and default values for Verifiers Monitor

This module centralizes all default values to ensure consistency across the codebase.
"""

# =============================================================================
# Storage Constants
# =============================================================================

# Default values for session and entity identification
DEFAULT_SESSION_ID = "unknown"
DEFAULT_MODEL_NAME = "unknown"
DEFAULT_ENV_TYPE = "unknown"
DEFAULT_PARSER_TYPE = "unknown"
DEFAULT_TASK = "default"

# Empty string defaults
DEFAULT_EMPTY_STRING = ""

# =============================================================================
# Dashboard Constants
# =============================================================================

DEFAULT_DASHBOARD_PORT = 8080
DEFAULT_TUNNEL = False

# =============================================================================
# Monitoring Constants
# =============================================================================

# Default limits for data retrieval
DEFAULT_QUERY_LIMIT = 100
DEFAULT_PAGE_SIZE = 100

# Default thresholds
DEFAULT_PERFORMANCE_THRESHOLD = 1.0  # seconds
DEFAULT_MEMORY_THRESHOLD = 100  # MB

# =============================================================================
# Retry Constants
# =============================================================================

DEFAULT_MAX_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BASE_DELAY = 1.0  # seconds
DEFAULT_RETRY_MAX_DELAY = 60.0  # seconds
DEFAULT_BACKOFF_MULTIPLIER = 2.0
