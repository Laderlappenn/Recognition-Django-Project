from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_photo, name='upload_photo'),
    path('about/', views.about, name='about'),
    # path('login/', views.login, name='login'),
    # path('profile/', views.profile, name='profile'),

]