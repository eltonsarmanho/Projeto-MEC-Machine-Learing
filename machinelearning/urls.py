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
from prediction import views
from rest_framework import routers
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from django.conf import settings
from django.conf.urls.static import static

schema_view = get_schema_view(
    openapi.Info(
        title="Machine Learning API",
        default_version='v1',
        description="Teste"
    ),
    public=True,
    permission_classes=[permissions.AllowAny],
)


router = routers.DefaultRouter()
router.register('dimensoes', views.DimensaoViewSet, basename='dimensoes')
router.register('fatores', views.FatorViewSet, basename='fatores')
router.register('dimensoes_est', views.DimensaoESTViewSet, basename='dimensoes_est')
router.register('fatore_est', views.FatorESTViewSet, basename='fatores_est')


urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include(router.urls)),
    path('processamento_from_json/', views.ProcessamentoFromJson.as_view()),
    path('', views.InitialView.as_view(), name='home'),
    path('sapdimensoes/', views.SapDimensoesView.as_view(), name='sapdimensoes'),
    path('sapfatores/', views.SapFatoresView.as_view(), name='sapfatores'),
    path('geral/', views.GeralView.as_view(), name='geral'),
    path('velofatores/', views.VeloFatoresView.as_view(), name='velofatores'),
    path('velodimensoes/', views.VeloDimensoesView.as_view(), name='velodimensoes'),
    path('radar/', views.SapRadarView.as_view(), name='radar'),
    path('radarcensosaeb/', views.SapRadarSaebCensoView.as_view(), name='radarcensosaeb'),

    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
