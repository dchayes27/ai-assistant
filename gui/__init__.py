"""
GUI module for AI Assistant
Contains Gradio-based user interface components
"""

from .interface import create_interface, launch_interface
from .components import *

__all__ = [
    "create_interface",
    "launch_interface"
]