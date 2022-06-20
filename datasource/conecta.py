from django.db import connections
from collections import namedtuple

def dictfetchall(cursor):
    "Return all rows from a cursor as a dict"
    columns = [col[0] for col in cursor.description]
    return [
        dict(zip(columns, row))
        for row in cursor.fetchall()
    ]

def get_respostas():
    questionario_aluno = []
    questionario_escola = []
    with connections['datasource'].cursor as cursor:
        sql = 'select * from questionario_aluno'
        cursor.execute(sql)
        for result in dictfetchall(cursor):
            for k,v in result.items():
                if k.startswith('questao'):
                    questionario_aluno[k] = v

        sql2 = 'select * from questionario_escola'
        cursor.execute(sql2)
        for result in dictfetchall(cursor):
            for k,v in result.items():
                if k.startswith('questao'):
                    questionario_aluno[k] = v

    return questionario_aluno, questionario_escola