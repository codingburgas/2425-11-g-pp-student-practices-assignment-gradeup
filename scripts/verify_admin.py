from app import create_app, db
from app.models import User

app = create_app()
with app.app_context():
    admin = User.query.filter_by(username='admin').first()
    if admin:
        admin.email_verified = True
        db.session.commit()
        print("Admin user email marked as verified.")
    else:
        print("Admin user not found.") 