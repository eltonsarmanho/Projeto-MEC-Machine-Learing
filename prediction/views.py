from prediction.serializer import DimensaoESTSerializer, FatorESTSerializer, DimensaoSerializer, FatorSerializer, ProcessamentoSerializer
from prediction.models import DimensaoEST, FatorEST, Dimensao, Fator
from rest_framework import viewsets
from rest_framework.views import APIView
from rest_framework.response import Response


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


class ProcessamentoFromJson(APIView):

    def post(self, request):
        serializer = ProcessamentoSerializer(data=request.data)
        if serializer.is_valid():
            return Response(serializer.processar(), status=201)

        return Response(serializer.errors, status=400)