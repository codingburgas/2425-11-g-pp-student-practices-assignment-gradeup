"""
Recommendation Engine Blueprint

This module provides Flask routes for the recommendation engine functionality.
"""

from flask import Blueprint, request, jsonify, render_template, flash, redirect, url_for
from flask_login import login_required, current_user
from app.models import User, SurveyResponse
from .recommendation_engine import recommendation_engine
import json
from flask import current_app

recommendation_bp = Blueprint('recommendations', __name__, url_prefix='/api/recommendations')


@recommendation_bp.route('/universities', methods=['POST'])
@login_required
def get_university_recommendations():
    """Get university recommendations based on user preferences and survey data."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        user_preferences = data.get('user_preferences', {})
        survey_data = data.get('survey_data')
        top_k = data.get('top_k', 10)
        
        # Validate required fields
        if not user_preferences and not survey_data:
            return jsonify({'error': 'Either user_preferences or survey_data must be provided'}), 400
        
        recommendations = recommendation_engine.match_universities(
            user_preferences=user_preferences,
            survey_data=survey_data,
            top_k=top_k
        )
        
        return jsonify({
            'status': 'success',
            'recommendations': recommendations,
            'count': len(recommendations)
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in university recommendations: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@recommendation_bp.route('/programs', methods=['POST'])
@login_required
def get_program_recommendations():
    """Get program recommendations for the current user."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        survey_data = data.get('survey_data')
        if not survey_data:
            return jsonify({'error': 'Survey data required'}), 400
            
        user_preferences = data.get('preferences', {})
        top_k = data.get('top_k', 10)
        
        # Get user preferences from profile if not provided
        if not user_preferences and current_user.preferences:
            user_preferences = current_user.get_preferences()
            
        recommendations = recommendation_engine.recommend_programs(
            user_id=current_user.id,
            survey_data=survey_data,
            user_preferences=user_preferences,
            top_k=top_k
        )
        
        # Store recommendation history
        if recommendations:
            survey_response_id = data.get('survey_response_id')
            recommendation_engine.store_recommendation_history(
                user_id=current_user.id,
                survey_response_id=survey_response_id,
                recommendations=recommendations,
                recommendation_type='program'
            )
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'count': len(recommendations)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@recommendation_bp.route('/personalized', methods=['GET'])
@login_required
def get_personalized_suggestions():
    """Get personalized suggestions for the current user."""
    try:
        limit = request.args.get('limit', 5, type=int)
        
        suggestions = recommendation_engine.get_personalized_suggestions(
            user_id=current_user.id,
            limit=limit
        )
        
        return jsonify({
            'success': True,
            'suggestions': suggestions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@recommendation_bp.route('/history', methods=['GET'])
@login_required
def get_recommendation_history():
    """Get recommendation history for the current user."""
    try:
        limit = request.args.get('limit', 20, type=int)
        
        history = recommendation_engine.get_recommendation_history(
            user_id=current_user.id,
            limit=limit
        )
        
        return jsonify({
            'success': True,
            'history': history,
            'count': len(history)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@recommendation_bp.route('/patterns', methods=['GET'])
@login_required
def get_recommendation_patterns():
    """Get recommendation pattern analysis for the current user."""
    try:
        analysis = recommendation_engine.analyze_recommendation_patterns(
            user_id=current_user.id
        )
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@recommendation_bp.route('/survey-based/<int:survey_response_id>', methods=['GET'])
@login_required
def get_survey_based_recommendations(survey_response_id):
    """Get recommendations based on a specific survey response."""
    try:
        # Verify user owns this survey response
        survey_response = SurveyResponse.query.filter_by(
            id=survey_response_id,
            user_id=current_user.id
        ).first()
        
        if not survey_response:
            return jsonify({'error': 'Survey response not found'}), 404
            
        survey_data = survey_response.get_answers()
        top_k = request.args.get('top_k', 10, type=int)
        
        # Get user preferences
        user_preferences = {}
        if current_user.preferences:
            user_preferences = current_user.get_preferences()
            
        recommendations = recommendation_engine.recommend_programs(
            user_id=current_user.id,
            survey_data=survey_data,
            user_preferences=user_preferences,
            top_k=top_k
        )
        
        # Store recommendation history
        if recommendations:
            recommendation_engine.store_recommendation_history(
                user_id=current_user.id,
                survey_response_id=survey_response_id,
                recommendations=recommendations,
                recommendation_type='program'
            )
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'survey_response_id': survey_response_id,
            'count': len(recommendations)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500 