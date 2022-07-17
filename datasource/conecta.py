from django.db import connections
from collections import namedtuple
import json

def dictfetchall(cursor):
    "Return all rows from a cursor as a dict"
    columns = [col[0] for col in cursor.description]
    return [
        dict(zip(columns, row))
        for row in cursor.fetchall()
    ]

def get_respostas():
    resultado = []
    with connections['datasource'].cursor() as cursor:
        consulta = '''SELECT t.id_escola IDESCOLA, a.id_aluno AS IDALUNO, a.id_turma IDTURMA, 
                qe.questao1 QE1, qe.questao2 QE2, qe.questao3 QE3, qe.questao4 QE4, qe.questao5 QE5, qe.questao6 QE6, qe.questao7 QE7, qe.questao8 QE8, qe.questao9 QE9, 
                qe.questao10 QE10, qe.questao11 QE11, qe.questao12 QE12, qe.questao13 QE13, qe.questao14 QE14, qe.questao15 QE15, qe.questao16 QE16, qe.questao17 QE17, qe.questao18 QE18, 
                qa.questao1 QA1, qa.questao2 QA2, qa.questao3 QA3, qa.questao4 QA4, qa.questao5 QA5, qa.questao6 QA6, qa.questao7 QA7, qa.questao8 QA8, qa.questao9 QA9, qa.questao10 QA10,
                qa.questao11 QA11, qa.questao12 QA12, qa.questao13 QA13, qa.questao14 QA14, qa.questao15 QA15, qa.questao16 QA16, qa.questao17 QA17, qa.questao18 QA18, qa.questao19 QA19, 
                qa.questao20 QA20, qa.questao21 QA21, qa.questao22 QA22, qa.questao23 QA23, qa.questao24 QA24, qa.questao25 QA25, qa.questao26 QA26, qa.questao27 QA27, qa.questao28 QA28,
                qa.questao29 QA29, qa.questao30 QA30, qa.questao31 QA31, qa.questao32 QA32,
                qd.questao1 QSD1, qd.questao2 QSD2, qd.questao3 QSD3, qd.questao4 QSD4, qd.questao5 QSD5, qd.questao6 QSD6, qd.questao7 QSD7, qd.questao8 QSD8, qd.questao9 QSD9, qd.questao10 QSD10
            FROM aluno a 
                JOIN turma t ON t.id_turma = a.id_turma
                JOIN questionario_escola qe ON qe.id_aluno = a.id_aluno
                JOIN questionario_aluno qa ON qa.id_aluno = a.id_aluno
                JOIN questionario_socio_demografico qd ON qd.id_aluno = a.id_aluno'''
        cursor.execute(consulta)
        for result in dictfetchall(cursor):
            resultado.append(result)

    return json.dumps(resultado)