from flask import Blueprint

api_bp = Blueprint('api', __name__)
web_bp = Blueprint('web', __name__)

from . import api_routes, web_routes
