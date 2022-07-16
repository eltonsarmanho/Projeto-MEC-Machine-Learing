from prediction.serializer import PredicaoSerializer
from prediction.models import Predicao
from rest_framework import viewsets

class PredicaoViewSet(viewsets.ModelViewSet):
  """Fator de risco para os alunos"""
  
  serializer_class = PredicaoSerializer
  http_method_names = ['get', 'post', 'put']
  
  def get_queryset(self):
    return Predicao.objects.all()