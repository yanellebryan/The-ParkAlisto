import logging
import logging.config
import os

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard'
            },
            'file': {
                'level': 'INFO',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': 'logs/app.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'formatter': 'standard',
            },
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
        },
    }
    
    logging.config.dictConfig(logging_config)
