"""Validation utilities for data processing"""

def validate_data_length(data_length: int, min_required: int = 60) -> None:
    """Validate if there's sufficient data for analysis"""
    if data_length < min_required:
        raise ValueError(f"Insufficient data. Required at least {min_required} data points")
