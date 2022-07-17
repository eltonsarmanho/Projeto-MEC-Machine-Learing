"""machinelearning URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from prediction.views import PredicaoViewSet, DimensoesESTViewSet, FatoresESTViewSet, DimensoesViewSet, FatoresViewSet
from rest_framework import routers

router = routers.DefaultRouter()
router.register('predicao', PredicaoViewSet, basename='predicao')
router.register('dimensoes', DimensoesViewSet, basename='dimensoes')
router.register('fatores', FatoresViewSet, basename='fatores')
router.register('dimensoes_est', DimensoesESTViewSet, basename='dimensoes_est')
router.register('fatore_est', FatoresESTViewSet, basename='fatores_est')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include(router.urls)),
]
