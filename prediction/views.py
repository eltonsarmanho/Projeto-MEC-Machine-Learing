from prediction.serializer import PredicaoSerializer, DimensoesESTSerializer, FatoresESTSerializer, DimensoesSerializer, FatoresSerializer
from prediction.models import Predicao, DimensoesEST, FatoresEST, Dimensoes, Fatores
from rest_framework import viewsets

class PredicaoViewSet(viewsets.ModelViewSet):
  """Fator de risco para os alunos"""
  
  serializer_class = PredicaoSerializer
  http_method_names = ['get', 'post', 'put']
  
  def get_queryset(self):
    return Predicao.objects.all()

class DimensoesViewSet(viewsets.ModelViewSet):
  """Dimensões de risco para os alunos"""
  
  serializer_class = DimensoesSerializer
  http_method_names = ['get', 'post', 'put']
  
  def get_queryset(self):
    return Dimensoes.objects.all()

class FatoresViewSet(viewsets.ModelViewSet):
  """Fatores de risco para os alunos"""
  
  serializer_class = FatoresSerializer
  http_method_names = ['get', 'post', 'put']
  
  def get_queryset(self):
    return Fatores.objects.all()

class DimensoesESTViewSet(viewsets.ModelViewSet):
  """Dimensões do estudante"""
  
  serializer_class = DimensoesESTSerializer
  http_method_names = ['get', 'post', 'put']
  
  def get_queryset(self):
    return DimensoesEST.objects.all()
  
class FatoresESTViewSet(viewsets.ModelViewSet):
  """Fatores do estudante"""
  
  serializer_class = FatoresESTSerializer
  http_method_names = ['get', 'post', 'put']
  
  def get_queryset(self):
    return FatoresEST.objects.all()