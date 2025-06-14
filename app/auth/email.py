from flask import current_app, render_template, url_for
from flask_mail import Message
from app import mail
import os
from threading import Thread


def send_async_email(app, msg):
    """Send email asynchronously"""
    try:
        with app.app_context():
            mail.send(msg)
    except Exception as e:
        # Log error in production, don't print to console
        pass


def send_email(subject, sender, recipients, text_body, html_body):
    """Send email with both text and HTML body"""
    try:
        # Use the authenticated Gmail account as sender
        actual_sender = current_app.config['MAIL_USERNAME']
        
        msg = Message(subject, sender=actual_sender, recipients=recipients)
        msg.body = text_body
        msg.html = html_body
        
        # Send email asynchronously to avoid blocking the request
        Thread(target=send_async_email, 
               args=(current_app._get_current_object(), msg)).start()
               
    except Exception as e:
        # Re-raise exception to be handled by calling function
        raise


def send_verification_email(user, sync=False):
    """Send email verification email to user"""
    # Use existing token if available, otherwise generate new one
    if not user.email_verification_token or user.is_email_verification_token_expired():
        token = user.generate_email_verification_token()
    else:
        token = user.email_verification_token
    
    # Build verification URL with proper server configuration
    try:
        verification_url = url_for('auth.verify_email', 
                                  token=token, 
                                  _external=True)
    except RuntimeError:
        # Fallback for when SERVER_NAME is not configured
        verification_url = f"http://localhost:5001/auth/verify_email/{token}"
    
    subject = 'Verify Your Email Address - School Recommendation System'
    
    text_body = f"""Dear {user.username},

Thank you for registering with the School Recommendation System!

To complete your registration, please verify your email address by clicking the link below:

{verification_url}

This verification link will expire in 24 hours.

If you did not create an account, please ignore this email.

Best regards,
School Recommendation System Team
"""
    
    html_body = f"""
    <html>
    <body>
        <h2>Welcome to School Recommendation System!</h2>
        <p>Dear <strong>{user.username}</strong>,</p>
        
        <p>Thank you for registering with our School Recommendation System!</p>
        
        <p>To complete your registration, please verify your email address by clicking the button below:</p>
        
        <div style="text-align: center; margin: 30px 0;">
            <a href="{verification_url}" 
               style="background-color: #007bff; color: white; padding: 12px 24px; 
                      text-decoration: none; border-radius: 5px; display: inline-block;">
                Verify Email Address
            </a>
        </div>
        
        <p>Or copy and paste this link into your browser:</p>
        <p><a href="{verification_url}">{verification_url}</a></p>
        
        <p><small>This verification link will expire in 24 hours.</small></p>
        
        <p>If you did not create an account, please ignore this email.</p>
        
        <hr>
        <p><small>Best regards,<br>School Recommendation System Team</small></p>
    </body>
    </html>
    """
    
    if sync:
        # Send synchronously for debugging
        try:
            actual_sender = current_app.config['MAIL_USERNAME']
            msg = Message(subject, sender=actual_sender, recipients=[user.email])
            msg.body = text_body
            msg.html = html_body
            
            mail.send(msg)
        except Exception as e:
            # Re-raise exception to be handled by calling function
            raise
    else:
        # Send asynchronously
        send_email(
            subject=subject,
            sender=current_app.config['MAIL_SENDER'],
            recipients=[user.email],
            text_body=text_body,
            html_body=html_body
        )


def send_welcome_email(user):
    """Send welcome email after successful verification"""
    subject = 'Welcome to School Recommendation System!'
    
    text_body = f"""Dear {user.username},

Your email has been successfully verified!

Welcome to the School Recommendation System. You can now:
- Complete surveys to get personalized school recommendations
- Browse schools and programs
- Save your favorite schools
        - Get detailed information about admission requirements

Start exploring: http://localhost:5001

Best regards,
School Recommendation System Team
"""
    
    html_body = f"""
    <html>
    <body>
        <h2>Welcome to School Recommendation System!</h2>
        <p>Dear <strong>{user.username}</strong>,</p>
        
        <p>🎉 Your email has been successfully verified!</p>
        
        <p>Welcome to the School Recommendation System. You can now:</p>
        <ul>
            <li>Complete surveys to get personalized school recommendations</li>
            <li>Browse schools and programs</li>
            <li>Save your favorite schools</li>
            <li>Get detailed information about admission requirements</li>
        </ul>
        
        <div style="text-align: center; margin: 30px 0;">
            <a href="http://localhost:5001" 
               style="background-color: #28a745; color: white; padding: 12px 24px; 
                      text-decoration: none; border-radius: 5px; display: inline-block;">
                Start Exploring
            </a>
        </div>
        
        <hr>
        <p><small>Best regards,<br>School Recommendation System Team</small></p>
    </body>
    </html>
    """
    
    send_email(
        subject=subject,
        sender=current_app.config['MAIL_SENDER'],
        recipients=[user.email],
        text_body=text_body,
        html_body=html_body
    )


def send_resend_verification_email(user):
    """Send new verification email when user requests resend"""
    # Generate new token
    token = user.generate_email_verification_token()
    
    # Build verification URL with proper server configuration
    try:
        verification_url = url_for('auth.verify_email', 
                                  token=token, 
                                  _external=True)
    except RuntimeError:
        # Fallback for when SERVER_NAME is not configured
        verification_url = f"http://localhost:5001/auth/verify_email/{token}"
    
    subject = 'Email Verification - School Recommendation System'
    
    text_body = f"""Dear {user.username},

You requested a new email verification link.

Please verify your email address by clicking the link below:

{verification_url}

This verification link will expire in 24 hours.

If you did not request this, please ignore this email.

Best regards,
School Recommendation System Team
"""
    
    html_body = f"""
    <html>
    <body>
        <h2>Email Verification</h2>
        <p>Dear <strong>{user.username}</strong>,</p>
        
        <p>You requested a new email verification link.</p>
        
        <p>Please verify your email address by clicking the button below:</p>
        
        <div style="text-align: center; margin: 30px 0;">
            <a href="{verification_url}" 
               style="background-color: #007bff; color: white; padding: 12px 24px; 
                      text-decoration: none; border-radius: 5px; display: inline-block;">
                Verify Email Address
            </a>
        </div>
        
        <p>Or copy and paste this link into your browser:</p>
        <p><a href="{verification_url}">{verification_url}</a></p>
        
        <p><small>This verification link will expire in 24 hours.</small></p>
        
        <p>If you did not request this, please ignore this email.</p>
        
        <hr>
        <p><small>Best regards,<br>School Recommendation System Team</small></p>
    </body>
    </html>
    """
    
    send_email(
        subject=subject,
        sender=current_app.config['MAIL_SENDER'],
        recipients=[user.email],
        text_body=text_body,
        html_body=html_body
    )


def send_password_reset_email(user):
    """Send password reset email to user"""
    # Use existing token if available and not expired, otherwise generate new one
    if not user.password_reset_token or user.is_password_reset_token_expired():
        token = user.generate_password_reset_token()
    else:
        token = user.password_reset_token
    
    # Build reset URL with proper server configuration
    try:
        reset_url = url_for('auth.reset_password', 
                           token=token, 
                           _external=True)
    except RuntimeError:
        # Fallback for when SERVER_NAME is not configured
        reset_url = f"http://localhost:5001/auth/reset_password/{token}"
    
    subject = 'Password Reset Request - School Recommendation System'
    
    text_body = f"""Dear {user.username},

You have requested to reset your password for the School Recommendation System.

To reset your password, please click the link below:

{reset_url}

This password reset link will expire in 1 hour.

If you did not request a password reset, please ignore this email. Your password will remain unchanged.

Best regards,
School Recommendation System Team
"""
    
    html_body = f"""
    <html>
    <body>
        <h2>Password Reset Request</h2>
        <p>Dear <strong>{user.username}</strong>,</p>
        
        <p>You have requested to reset your password for the School Recommendation System.</p>
        
        <p>To reset your password, please click the button below:</p>
        
        <div style="text-align: center; margin: 30px 0;">
            <a href="{reset_url}" 
               style="background-color: #dc3545; color: white; padding: 12px 24px; 
                      text-decoration: none; border-radius: 5px; display: inline-block;">
                Reset Password
            </a>
        </div>
        
        <p>Or copy and paste this link into your browser:</p>
        <p><a href="{reset_url}">{reset_url}</a></p>
        
        <p><small>This password reset link will expire in 1 hour.</small></p>
        
        <p>If you did not request a password reset, please ignore this email. Your password will remain unchanged.</p>
        
        <hr>
        <p><small>Best regards,<br>School Recommendation System Team</small></p>
    </body>
    </html>
    """
    
    send_email(
        subject=subject,
        sender=current_app.config['MAIL_SENDER'],
        recipients=[user.email],
        text_body=text_body,
        html_body=html_body
    ) 