from config import Config

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def validate_video_source(source):
    # Could check if file exists or if URL is valid
    pass
