"""Atomic tool interfaces for mathematical operations."""

from .calculator import (
    evaluate_expression,
    simplify_expression,
    solve_equation,
    differentiate_expression,
    integrate_expression,
    numerical_integration,
    solve_ode,
)
from .plotter import plot_function_2d
from .ocr import OCRProcessingError, OCRUnavailableError, extract_text_from_images

__all__ = [
    "evaluate_expression",
    "simplify_expression",
    "solve_equation",
    "differentiate_expression",
    "integrate_expression",
    "numerical_integration",
    "solve_ode",
    "plot_function_2d",
    "extract_text_from_images",
    "OCRUnavailableError",
    "OCRProcessingError",
]
