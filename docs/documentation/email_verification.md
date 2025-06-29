# Email Verification Toggle Documentation

## Overview
The email verification toggle feature allows administrators to enable or disable the email verification requirement for user registration.

## Configuration
To configure the email verification toggle, add the following to your `.env` file:

```
# Feature Toggles
# Set to true to disable email verification (users can log in without verifying their email)
DISABLE_EMAIL_VERIFICATION=false
```

Set to `true` to disable email verification or `false` to require email verification.

## Implementation Details

### Configuration
The toggle is implemented in `config.py` as a configuration option:

```python
# Email verification toggle
DISABLE_EMAIL_VERIFICATION = os.environ.get('DISABLE_EMAIL_VERIFICATION', 'false').lower() in ['true', 'on', '1', 'yes']
```

### Authentication Routes
The toggle is checked in the authentication routes to determine whether email verification is required:

- During registration, users are automatically marked as verified if the toggle is enabled
- During login, the email verification check is skipped if the toggle is enabled
- Email verification endpoints return appropriate responses when the toggle is enabled

## Security Considerations
Disabling email verification reduces security by removing the email ownership verification step. This should only be used in development environments or when an alternative verification method is in place. 