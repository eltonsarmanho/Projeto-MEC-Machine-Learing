from prediction.serializer import DimensaoESTSerializer, FatorESTSerializer, DimensaoSerializer, FatorSerializer, ProcessamentoSerializer
from prediction.models import DimensaoEST, FatorEST, Dimensao, Fator
from rest_framework import viewsets
from rest_framework.views import APIView
from rest_framework.response import Response
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from django.views.generic import TemplateView


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
    request_schema_dict = openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            'alunos': openapi.Schema(type=openapi.TYPE_ARRAY, description=('Lista das respostas dos alunos'), 
                items=openapi.Schema(type=openapi.TYPE_OBJECT, description=('QE - questionário da escola, QA - questionário do aluno, QSD - questionário sócio demográfico'),
                    properties={
                        'IDALUNO': openapi.Schema(type=openapi.TYPE_INTEGER, description=('ID do aluno'), example=1),
                        'QE1': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 1 da escola"), example=1),
                        'QE2': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 2 da escola"), example=1),
                        'QE3': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 3 da escola"), example=1),
                        'QE4': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 4 da escola"), example=1),
                        'QE5': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 5 da escola"), example=1),
                        'QE6': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 6 da escola"), example=1),
                        'QE7': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 7 da escola"), example=1),
                        'QE8': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 8 da escola"), example=1),
                        'QE9': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 9 da escola"), example=1),
                        'QE10': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 10 da escola"), example=1),
                        'QE11': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 11 da escola"), example=1),
                        'QE12': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 12 da escola"), example=1),
                        'QE13': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 13 da escola"), example=1),
                        'QE14': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 14 da escola"), example=1),
                        'QE15': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 15 da escola"), example=1),
                        'QE16': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 16 da escola"), example=1),
                        'QE17': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 17 da escola"), example=1),
                        'QA1': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 1 do aluno"), example=1),
                        'QA2': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 2 do aluno"), example=1),
                        'QA3': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 3 do aluno"), example=1),
                        'QA4': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 4 do aluno"), example=1),
                        'QA5': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 5 do aluno"), example=1),
                        'QA6': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 6 do aluno"), example=1),
                        'QA7': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 7 do aluno"), example=1),
                        'QA8': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 8 do aluno"), example=1),
                        'QA9': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 9 do aluno"), example=1),
                        'QA10': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 10 do aluno"), example=1),
                        'QA11': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 11 do aluno"), example=1),
                        'QA12': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 12 do aluno"), example=1),
                        'QA13': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 13 do aluno"), example=1),
                        'QA14': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 14 do aluno"), example=1),
                        'QA15': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 15 do aluno"), example=1),
                        'QA16': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 16 do aluno"), example=1),
                        'QA17': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 17 do aluno"), example=1),
                        'QA18': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 18 do aluno"), example=1),
                        'QA19': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 19 do aluno"), example=1),
                        'QA20': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 20 do aluno"), example=1),
                        'QA21': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 21 do aluno"), example=1),
                        'QA22': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 22 do aluno"), example=1),
                        'QA23': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 23 do aluno"), example=1),
                        'QA24': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 24 do aluno"), example=1),
                        'QA25': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 25 do aluno"), example=1),
                        'QA26': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 26 do aluno"), example=1),
                        'QA27': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 27 do aluno"), example=1),
                        'QA28': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 28 do aluno"), example=1),
                        'QA29': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 29 do aluno"), example=1),
                        'QA30': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 30 do aluno"), example=1),
                        'QA31': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 31 do aluno"), example=1),
                        'QA32': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 32 do aluno"), example=1),
                        'QSD1': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 1 do sócio demográfico"), example=1),
                        'QSD2': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 2 do sócio demográfico"), example=1),
                        'QSD3': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 3 do sócio demográfico"), example=1),
                        'QSD4': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 4 do sócio demográfico"), example=1),
                        'QSD5': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 5 do sócio demográfico"), example=1),
                        'QSD6': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 6 do sócio demográfico"), example=1),
                        'QSD7': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 7 do sócio demográfico"), example=1),
                        'QSD8': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 8 do sócio demográfico"), example=1),
                        'QSD9': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 9 do sócio demográfico"), example=1),
                        'QSD10': openapi.Schema(type=openapi.TYPE_INTEGER, description=("Questão 10 do sócio demográfico"), example=1),               
                    }
                )
            ),
        }
    )

    response_schema_dict = {
        "201": openapi.Response(
            description="Resultado do processamento",
            examples={
                "application/json": [
                    [
                        {
                            "IDALUNO": 1,
                            "E-ESC1V": 1,
                            "E-ESC2V": 1,
                            "E-PROF1V": 1,
                            "E-PROF2V": 1,
                            "E-FAM1V": 1,
                            "E-FAM2V": 1,
                            "E-COM1V": 1,
                            "E-COM2V": 1,
                            "E-COM3V": 1,
                            "E-EST1V": 1,
                            "E-EST2V": 1,
                            "E-EST3V": 1,
                            "E-ESC1C": "Risco Baixo",
                            "E-ESC2C": "Risco Baixo",
                            "E-PROF1C": "Risco Baixo",
                            "E-PROF2C": "Risco Baixo",
                            "E-FAM1C": "Risco Baixo",
                            "E-FAM2C": "Risco Baixo",
                            "E-COM1C": "Risco Baixo",
                            "E-COM2C": "Risco Baixo",
                            "E-COM3C": "Risco Baixo",
                            "E-EST1C": "Risco Baixo",
                            "E-EST2C": "Risco Baixo",
                            "E-EST3C": "Risco Baixo"
                        },
                    ], 
                    [
                        {
                        "IDALUNO": 1,
                        "E-ESCV": 1,
                        "E-PROFV": 1,
                        "E-FAMV": 1,
                        "E-COMV": 1,
                        "E-ESTV": 1,
                        "E-ESCC": "Risco Baixo",
                        "E-PROFC": "Risco Baixo",
                        "E-FAMC": "Risco Baixo",
                        "E-COMC": "Risco Baixo",
                        "E-ESTC": "Risco Baixo"
                        },
                    ]
                ]
            }
        ),
    }

    @swagger_auto_schema(request_body=request_schema_dict, responses=response_schema_dict)
    def post(self, request):
        serializer = ProcessamentoSerializer(data=request.data)
        if serializer.is_valid():
            return Response(serializer.processar(), status=201)

        return Response(serializer.errors, status=400)


class InitialView(TemplateView):
    template_name = 'initial.html'
    

    def get_context_data(self, **kwargs):
        context = super(InitialView, self).get_context_data(**kwargs)
        from prediction.graphics import graph_bar_valor_por_pontuacao_segmentado
        
        
        context['titulo'] = 'Dashboard'
        context['graph'] = graph_bar_valor_por_pontuacao_segmentado().to_html()
        return context


class SapDimensoesView(TemplateView):
    template_name = 'initial.html'


    def get_context_data(self, **kwargs):
        context = super(SapDimensoesView, self).get_context_data(**kwargs)
        from prediction.graphics import grafico_risco_escola_dimensoes_barras
        from prediction.graphics import grafico_risco_escola_dimensoes_barras2
        from prediction.graphics import grafico_radar_sap

        context['titulo'] = 'SAP Dimensoes'
        context['graph'] = grafico_risco_escola_dimensoes_barras().to_html()
        context['graph2'] = grafico_risco_escola_dimensoes_barras2().to_html()
        context['graph3'] = grafico_radar_sap().to_html()

        return context

class SapFatoresView(TemplateView):
    template_name = 'initial.html'


    def get_context_data(self, **kwargs):
        context = super(SapFatoresView, self).get_context_data(**kwargs)
        from prediction.graphics import grafico_risco_escola_fatores_barras
        from prediction.graphics import grafico_risco_escola_fatores_barras2

        context['titulo'] = 'SAP Fatores'
        context['graph'] = grafico_risco_escola_fatores_barras().to_html()
        context['graph2'] = grafico_risco_escola_fatores_barras2().to_html()

        return context

class GeralView(TemplateView):
    template_name = 'initial.html'


    def get_context_data(self, **kwargs):
        context = super(GeralView, self).get_context_data(**kwargs)
        from prediction.graphics import texto_sap_quant_est_esc
        from prediction.graphics import texto_apa_quant_est_esc
        from prediction.graphics import table_apa_ciclo
        from prediction.graphics import media_dimensoes
        from prediction.graphics import digitalizacoes_apa
        #from prediction.graphics import dem_quantidades
        #from prediction.graphics import dem_quan_pont
        #from prediction.graphics import dem_quan_seg
        #from prediction.graphics import dem_quan_dig_status
        #from prediction.graphics import desc_estado_sap
        #import time
        
        #inicio = time.time()

        context['titulo'] = 'Visão geral dos sistemas atuais'
        context['text1'] = texto_sap_quant_est_esc()
        context['text2'] = texto_apa_quant_est_esc()
        

        context['table1'] = table_apa_ciclo().to_html()
        context['table2'] = media_dimensoes().to_html()
        context['table3'] = digitalizacoes_apa().to_html()
        #context['table4'] = dem_quantidades().to_html()
        #context['table5'] = dem_quan_pont().to_html()
        #context['table6'] = dem_quan_seg().to_html()
        #context['table7'] = dem_quan_dig_status().to_html()
        #context['table8'] = desc_estado_sap().to_html()

        #fim = time.time()
        #print(fim - inicio)
        return context

class VeloFatoresView(TemplateView):
    template_name = 'initial.html'


    def get_context_data(self, **kwargs):
        context = super(VeloFatoresView, self).get_context_data(**kwargs)
        from prediction.graphics import velocimetro_fator

        context['titulo'] = 'SAP F Velocimetro'
        #context['text1'] = texto_sap_quant_est_esc()
        #context['text2'] = texto_apa_quant_est_esc()
        context['graph'] = velocimetro_fator().to_html()

        return context

class VeloDimensoesView(TemplateView):
    template_name = 'initial.html'

    def get_context_data(self, **kwargs):
        context = super(VeloDimensoesView, self).get_context_data(**kwargs)
        from prediction.graphics import velocimetro_dimensao

        context['titulo'] = 'SAP D Velocimetro'
        context['graph'] = velocimetro_dimensao().to_html()

        return context