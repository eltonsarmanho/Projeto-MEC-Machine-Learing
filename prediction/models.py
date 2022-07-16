from django.db import models

class Predicao(models.Model):
    id_aluno = models.IntegerField('Id do aluno', null=True)
    risco_escola = models.CharField('Risco da Escola', max_length=1, null=True)
    risco_estudante = models.CharField('Risco do Estudante', max_length=1, null=True)
    risco_geral = models.CharField('Risco Geral', max_length=1, null=True)
    
    class Meta:
        ordering = ['id']
        db_table = 'predicao'
