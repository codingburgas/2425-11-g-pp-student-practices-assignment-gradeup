"""
Data Export Module

Handles exporting survey data in various formats (CSV, JSON, Excel).
"""

import csv
import json
import io
from datetime import datetime
from typing import List, Dict, Any, Optional
import xlsxwriter
from flask import make_response
from .models import DataStorageManager, DataExportLog


class DataExporter:
    """Handles data export operations in various formats."""
    
    @staticmethod
    def export_to_csv(survey_data: List, filename: str = None) -> str:
        """Export survey data to CSV format."""
        if not filename:
            filename = f"survey_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        output = io.StringIO()
        
        if not survey_data:
            return output.getvalue()
        
        # Get all possible field names from all responses
        all_fields = set()
        for data in survey_data:
            raw_data = data.get_raw_data()
            all_fields.update(raw_data.keys())
        
        # Add metadata fields
        meta_fields = ['id', 'submission_time', 'processing_status', 'submission_ip']
        all_fields = list(all_fields) + meta_fields
        
        writer = csv.DictWriter(output, fieldnames=all_fields)
        writer.writeheader()
        
        for data in survey_data:
            row = data.get_raw_data().copy()
            # Add metadata
            row.update({
                'id': data.id,
                'submission_time': data.submission_time.isoformat(),
                'processing_status': data.processing_status,
                'submission_ip': data.submission_ip
            })
            writer.writerow(row)
        
        return output.getvalue()
    
    @staticmethod
    def export_to_json(survey_data: List, filename: str = None) -> str:
        """Export survey data to JSON format."""
        if not filename:
            filename = f"survey_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'export_info': {
                'filename': filename,
                'export_time': datetime.now().isoformat(),
                'record_count': len(survey_data)
            },
            'data': []
        }
        
        for data in survey_data:
            record = {
                'id': data.id,
                'survey_id': data.survey_id,
                'submission_time': data.submission_time.isoformat(),
                'processing_status': data.processing_status,
                'submission_ip': data.submission_ip,
                'user_agent': data.user_agent,
                'raw_data': data.get_raw_data(),
                'processed_data': data.get_processed_data(),
                'metadata': data.get_metadata()
            }
            export_data['data'].append(record)
        
        return json.dumps(export_data, indent=2, ensure_ascii=False)
    
    @staticmethod
    def export_to_excel(survey_data: List, filename: str = None) -> bytes:
        """Export survey data to Excel format."""
        if not filename:
            filename = f"survey_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        output = io.BytesIO()
        workbook = xlsxwriter.Workbook(output, {'in_memory': True})
        
        # Create main data worksheet
        worksheet = workbook.add_worksheet('Survey Data')
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#D3D3D3',
            'border': 1
        })
        date_format = workbook.add_format({'num_format': 'yyyy-mm-dd hh:mm:ss'})
        
        if not survey_data:
            worksheet.write('A1', 'No data available', header_format)
            workbook.close()
            output.seek(0)
            return output.read()
        
        # Get all possible field names
        all_fields = set()
        for data in survey_data:
            raw_data = data.get_raw_data()
            all_fields.update(raw_data.keys())
        
        # Define column headers
        headers = ['ID', 'Survey ID', 'Submission Time', 'Status', 'IP Address'] + list(all_fields)
        
        # Write headers
        for col, header in enumerate(headers):
            worksheet.write(0, col, header, header_format)
        
        # Write data
        for row, data in enumerate(survey_data, 1):
            raw_data = data.get_raw_data()
            
            worksheet.write(row, 0, data.id)
            worksheet.write(row, 1, data.survey_id)
            worksheet.write(row, 2, data.submission_time, date_format)
            worksheet.write(row, 3, data.processing_status)
            worksheet.write(row, 4, data.submission_ip or '')
            
            # Write survey response fields
            for col, field in enumerate(all_fields, 5):
                value = raw_data.get(field, '')
                worksheet.write(row, col, str(value))
        
        # Auto-adjust column widths
        for col, header in enumerate(headers):
            max_width = max(len(header), 15)
            worksheet.set_column(col, col, max_width)
        
        # Create summary worksheet
        summary_sheet = workbook.add_worksheet('Summary')
        summary_sheet.write('A1', 'Export Summary', header_format)
        summary_sheet.write('A3', 'Total Records:')
        summary_sheet.write('B3', len(survey_data))
        summary_sheet.write('A4', 'Export Time:')
        summary_sheet.write('B4', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        summary_sheet.write('A5', 'Filename:')
        summary_sheet.write('B5', filename)
        
        workbook.close()
        output.seek(0)
        return output.read()


class ExportManager:
    """Manages export operations and logging."""
    
    @staticmethod
    def create_export(survey_id: int, export_format: str, 
                     start_date: datetime = None, end_date: datetime = None,
                     exported_by: int = None) -> Dict[str, Any]:
        """Create and return an export of survey data."""
        try:
            # Get survey data
            survey_data = DataStorageManager.get_survey_data(
                survey_id=survey_id,
                start_date=start_date,
                end_date=end_date
            )
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"survey_{survey_id}_data_{timestamp}"
            
            exporter = DataExporter()
            
            if export_format.lower() == 'csv':
                content = exporter.export_to_csv(survey_data, f"{filename}.csv")
                content_type = 'text/csv'
                file_extension = '.csv'
                
            elif export_format.lower() == 'json':
                content = exporter.export_to_json(survey_data, f"{filename}.json")
                content_type = 'application/json'
                file_extension = '.json'
                
            elif export_format.lower() == 'excel':
                content = exporter.export_to_excel(survey_data, f"{filename}.xlsx")
                content_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                file_extension = '.xlsx'
                
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            
            # Log the export
            filters = {
                'survey_id': survey_id,
                'start_date': start_date.isoformat() if start_date else None,
                'end_date': end_date.isoformat() if end_date else None
            }
            
            export_log_id = DataStorageManager.log_export(
                export_type=export_format.lower(),
                record_count=len(survey_data),
                exported_by=exported_by,
                filters=filters,
                file_path=f"{filename}{file_extension}"
            )
            
            return {
                'success': True,
                'content': content,
                'content_type': content_type,
                'filename': f"{filename}{file_extension}",
                'record_count': len(survey_data),
                'export_log_id': export_log_id
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    @staticmethod
    def get_export_history(limit: int = 50) -> List[Dict]:
        """Get history of data exports."""
        from app import db
        from .models import DataExportLog
        
        exports = DataExportLog.query.order_by(
            DataExportLog.created_at.desc()
        ).limit(limit).all()
        
        export_history = []
        for export in exports:
            export_history.append({
                'id': export.id,
                'export_type': export.export_type,
                'record_count': export.record_count,
                'created_at': export.created_at.isoformat(),
                'status': export.status,
                'exported_by': export.exported_by,
                'filters': export.get_filters(),
                'file_path': export.file_path
            })
        
        return export_history
    
    @staticmethod
    def create_flask_response(export_result: Dict) -> Any:
        """Create a Flask response object for file download."""
        if not export_result['success']:
            from flask import jsonify
            return jsonify({'error': export_result['error']}), 500
        
        response = make_response(export_result['content'])
        response.headers['Content-Type'] = export_result['content_type']
        response.headers['Content-Disposition'] = f"attachment; filename={export_result['filename']}"
        
        return response 