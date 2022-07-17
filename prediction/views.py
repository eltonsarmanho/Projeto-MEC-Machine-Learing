from prediction.serializer import DimensaoESTSerializer, FatorESTSerializer, DimensaoSerializer, FatorSerializer
from prediction.models import DimensaoEST, FatorEST, Dimensao, Fator
from rest_framework import viewsets

class DimensaoViewSet(viewsets.ModelViewSet):
  """Dimensões de risco para os alunos"""
  
  serializer_class = DimensaoSerializer
  http_method_names = ['get', 'post', 'put']
  
  def get_queryset(self):
    return Dimensao.objects.all()

class FatorViewSet(viewsets.ModelViewSet):
  """Fatores de risco para os alunos"""
  
  serializer_class = FatorSerializer
  http_method_names = ['get', 'post', 'put']
  
  def get_queryset(self):
    return Fator.objects.all()

class DimensaoESTViewSet(viewsets.ModelViewSet):
  """Dimensões do estudante"""
  
  serializer_class = DimensaoESTSerializer
  http_method_names = ['get', 'post', 'put']
  
  def get_queryset(self):
    return DimensaoEST.objects.all()
  
class FatorESTViewSet(viewsets.ModelViewSet):
  """Fatores do estudante"""
  
  serializer_class = FatorESTSerializer
  http_method_names = ['get', 'post', 'put']
  
  def get_queryset(self):
    return FatorEST.objects.all()