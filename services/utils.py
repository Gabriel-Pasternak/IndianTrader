from typing import List, Dict, Any
import numpy as np

def validate_data_length(data_length: int, min_required: int = 60) -> None:
    """Validate if there's sufficient data for analysis"""
    if data_length < min_required:
        raise ValueError(f"Insufficient data. Required at least {min_required} data points")

def calculate_confidence_score(touches: int, max_score: float = 100.0) -> float:
    """Calculate confidence score based on number of touches"""
    return min(float(touches * 10), max_score)