from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, SubmitField
from wtforms.validators import Optional, Length

class UserSearchForm(FlaskForm):
    """Form for searching and filtering users"""
    search = StringField('Search Users', 
                        validators=[Optional(), Length(max=100)],
                        render_kw={'placeholder': 'Search by username or location...'})
    
    location_filter = SelectField('Filter by Location', 
                                 choices=[('', 'All Locations')],
                                 validators=[Optional()])
    
    sort_by = SelectField('Sort By', 
                         choices=[
                             ('username', 'Username (A-Z)'),
                             ('created_at_desc', 'Newest First'),
                             ('created_at_asc', 'Oldest First'),
                             ('location', 'Location')
                         ],
                         default='username',
                         validators=[Optional()])
    
    submit = SubmitField('Search')
    
    def __init__(self, *args, **kwargs):
        super(UserSearchForm, self).__init__(*args, **kwargs)
        # Populate location choices dynamically if needed
        from app.models import User
        locations = User.query.filter(User.location.isnot(None), User.location != '').distinct(User.location).all()
        location_choices = [('', 'All Locations')]
        location_choices.extend([(loc.location, loc.location) for loc in locations])
        self.location_filter.choices = location_choices 