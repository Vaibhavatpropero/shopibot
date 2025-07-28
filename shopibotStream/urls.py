from django.contrib import admin
from django.urls import path
from shopibotStream import views

urlpatterns = [
    path("", views.shopibot_dev, name="get_response")
]
