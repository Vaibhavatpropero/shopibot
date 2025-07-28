"""
WSGI config for Shopibot project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""

# import os

# from django.core.wsgi import get_wsgi_application

# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Shopibot.settings')

# application = get_wsgi_application()

# for azure

import os

from django.core.wsgi import get_wsgi_application

settings_module = "Shopibot.deployment" if "WEBSITE_HOSTNAME" in os.environ else "Shopibot.settings"
os.environ.setdefault('DJANGO_SETTINGS_MODULE', settings_module)

application = get_wsgi_application()
