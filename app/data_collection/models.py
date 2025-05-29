"""
Data Collection Models

Database models for survey data collection and storage functionality.
"""

from datetime import datetime
from app import db
import json
from sqlalchemy import text


class SurveyData(db.Model):
    """Extended survey data storage for analytics and reporting."""
    __tablename__ = 'survey_data'
    
    id = db.Column(db.Integer, primary_key=True)
    survey_id = db.Column(db.Integer, db.ForeignKey('surveys.id'), nullable=False)
    response_id = db.Column(db.Integer, db.ForeignKey('survey_responses.id'), nullable=True)
    raw_data = db.Column(db.Text, nullable=False)  # Original JSON data
    processed_data = db.Column(db.Text, nullable=True)  # Processed/cleaned data
    survey_metadata = db.Column(db.Text, nullable=True)  # Additional metadata
    submission_ip = db.Column(db.String(45), nullable=True)  # IPv4/IPv6 address
    user_agent = db.Column(db.String(255), nullable=True)
    submission_time = db.Column(db.DateTime, default=datetime.utcnow)
    processing_status = db.Column(db.String(20), default='pending')  # pending, processed, failed
    
    def set_raw_data(self, data_dict):
        """Store raw survey data as JSON."""
        self.raw_data = json.dumps(data_dict)
    
    def get_raw_data(self):
        """Retrieve raw survey data as dictionary."""
        return json.loads(self.raw_data) if self.raw_data else {}
    
    def set_processed_data(self, data_dict):
        """Store processed survey data as JSON."""
        self.processed_data = json.dumps(data_dict)
    
    def get_processed_data(self):
        """Retrieve processed survey data as dictionary."""
        return json.loads(self.processed_data) if self.processed_data else {}
    
    def set_metadata(self, metadata_dict):
        """Store metadata as JSON."""
        self.survey_metadata = json.dumps(metadata_dict)
    
    def get_metadata(self):
        """Retrieve metadata as dictionary."""
        return json.loads(self.survey_metadata) if self.survey_metadata else {}


class DataExportLog(db.Model):
    """Track data exports for auditing purposes."""
    __tablename__ = 'data_export_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    export_type = db.Column(db.String(50), nullable=False)  # csv, json, excel
    file_path = db.Column(db.String(255), nullable=True)
    record_count = db.Column(db.Integer, nullable=False, default=0)
    exported_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    export_filters = db.Column(db.Text, nullable=True)  # JSON of applied filters
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='pending')  # pending, completed, failed
    
    def set_filters(self, filters_dict):
        """Store export filters as JSON."""
        self.export_filters = json.dumps(filters_dict)
    
    def get_filters(self):
        """Retrieve export filters as dictionary."""
        return json.loads(self.export_filters) if self.export_filters else {}


class DataStorageManager:
    """Handles data storage operations for survey data."""
    
    @staticmethod
    def store_survey_submission(survey_id, response_data, user_id=None, metadata=None):
        """Store a new survey submission with full tracking."""
        try:
            # Create SurveyData record for analytics
            survey_data = SurveyData(
                survey_id=survey_id,
                submission_ip=metadata.get('ip') if metadata else None,
                user_agent=metadata.get('user_agent') if metadata else None
            )
            survey_data.set_raw_data(response_data)
            if metadata:
                survey_data.set_metadata(metadata)
            
            db.session.add(survey_data)
            db.session.flush()  # Get the ID without committing
            
            # If user is logged in, also create a SurveyResponse record
            if user_id:
                from app.models import SurveyResponse
                survey_response = SurveyResponse(
                    user_id=user_id,
                    survey_id=survey_id
                )
                survey_response.set_answers(response_data)
                db.session.add(survey_response)
                db.session.flush()
                
                # Link the survey_data to the response
                survey_data.response_id = survey_response.id
            
            db.session.commit()
            return survey_data.id
            
        except Exception as e:
            db.session.rollback()
            raise e
    
    @staticmethod
    def get_survey_data(survey_id=None, start_date=None, end_date=None, limit=None):
        """Retrieve survey data with optional filtering."""
        query = SurveyData.query
        
        if survey_id:
            query = query.filter(SurveyData.survey_id == survey_id)
        
        if start_date:
            query = query.filter(SurveyData.submission_time >= start_date)
        
        if end_date:
            query = query.filter(SurveyData.submission_time <= end_date)
        
        query = query.order_by(SurveyData.submission_time.desc())
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    @staticmethod
    def get_response_statistics(survey_id):
        """Get basic statistics for survey responses."""
        total_responses = SurveyData.query.filter_by(survey_id=survey_id).count()
        
        # Get responses by status
        status_counts = db.session.query(
            SurveyData.processing_status,
            db.func.count(SurveyData.id)
        ).filter_by(survey_id=survey_id).group_by(SurveyData.processing_status).all()
        
        # Get daily response counts for last 30 days
        thirty_days_ago = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - \
                         datetime.timedelta(days=30)
        
        daily_counts = db.session.query(
            db.func.date(SurveyData.submission_time).label('date'),
            db.func.count(SurveyData.id).label('count')
        ).filter(
            SurveyData.survey_id == survey_id,
            SurveyData.submission_time >= thirty_days_ago
        ).group_by(db.func.date(SurveyData.submission_time)).all()
        
        return {
            'total_responses': total_responses,
            'status_breakdown': dict(status_counts),
            'daily_responses': [{'date': str(row.date), 'count': row.count} for row in daily_counts]
        }
    
    @staticmethod
    def log_export(export_type, record_count, exported_by=None, filters=None, file_path=None):
        """Log a data export operation."""
        export_log = DataExportLog(
            export_type=export_type,
            record_count=record_count,
            exported_by=exported_by,
            file_path=file_path,
            status='completed'
        )
        
        if filters:
            export_log.set_filters(filters)
        
        db.session.add(export_log)
        db.session.commit()
        return export_log.id 