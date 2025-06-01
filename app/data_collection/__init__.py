"""
Data Collection Module

This module handles survey data collection, validation, storage, and export functionality.
"""

from flask import Blueprint

bp = Blueprint('data_collection', __name__)

from . import routes, models, validators, exporters 