"""
Data Validation Module

Handles validation of survey data and responses.
"""

import re
from typing import Dict, List, Any, Optional


class SurveyValidator:
    """Validates survey data and responses."""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(email_pattern, email))
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """Validate phone number format."""
        phone_pattern = r'^\+?[\d\s\-\(\)]{10,}$'
        return bool(re.match(phone_pattern, phone))
    
    @staticmethod
    def validate_required_fields(data: Dict, required_fields: List[str]) -> List[str]:
        """Check if all required fields are present and not empty."""
        missing_fields = []
        for field in required_fields:
            if field not in data or not data[field] or str(data[field]).strip() == '':
                missing_fields.append(field)
        return missing_fields
    
    @staticmethod
    def validate_field_length(value: str, min_length: int = 0, max_length: int = 1000) -> bool:
        """Validate field length constraints."""
        return min_length <= len(value.strip()) <= max_length
    
    @staticmethod
    def validate_rating_scale(value: Any, min_value: int = 1, max_value: int = 5) -> bool:
        """Validate rating scale values."""
        try:
            rating = int(value)
            return min_value <= rating <= max_value
        except (ValueError, TypeError):
            return False


class ResponseValidator:
    """Validates survey responses against survey schema."""
    
    def __init__(self, survey_schema: Dict):
        self.schema = survey_schema
        self.validator = SurveyValidator()
    
    def validate_response(self, response_data: Dict) -> Dict[str, Any]:
        """
        Validate a complete survey response.
        
        Returns:
            Dict with 'valid' boolean and 'errors' list
        """
        errors = []
        
        
        required_fields = self.schema.get('required_fields', [])
        missing_fields = self.validator.validate_required_fields(response_data, required_fields)
        if missing_fields:
            errors.extend([f"Missing required field: {field}" for field in missing_fields])
        
        
        for field_name, field_config in self.schema.get('fields', {}).items():
            if field_name in response_data:
                field_errors = self._validate_field(field_name, response_data[field_name], field_config)
                errors.extend(field_errors)
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _validate_field(self, field_name: str, value: Any, field_config: Dict) -> List[str]:
        """Validate individual field based on its configuration."""
        errors = []
        field_type = field_config.get('type', 'text')
        
        if field_type == 'email':
            if not self.validator.validate_email(str(value)):
                errors.append(f"Invalid email format for field: {field_name}")
        
        elif field_type == 'phone':
            if not self.validator.validate_phone(str(value)):
                errors.append(f"Invalid phone format for field: {field_name}")
        
        elif field_type == 'rating':
            min_val = field_config.get('min', 1)
            max_val = field_config.get('max', 5)
            if not self.validator.validate_rating_scale(value, min_val, max_val):
                errors.append(f"Rating for {field_name} must be between {min_val} and {max_val}")
        
        elif field_type == 'text':
            min_len = field_config.get('min_length', 0)
            max_len = field_config.get('max_length', 1000)
            if not self.validator.validate_field_length(str(value), min_len, max_len):
                errors.append(f"Text length for {field_name} must be between {min_len} and {max_len} characters")
        
        return errors 