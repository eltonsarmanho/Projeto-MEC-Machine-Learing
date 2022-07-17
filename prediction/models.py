from django.db import models

class Predicao(models.Model):
    id_aluno = models.IntegerField('Id do aluno', null=True)
    risco_escola = models.CharField('Risco da Escola', max_length=1, null=True)
    risco_estudante = models.CharField('Risco do Estudante', max_length=1, null=True)
    risco_geral = models.CharField('Risco Geral', max_length=1, null=True)
    
    class Meta:
        ordering = ['id']
        db_table = 'predicao'

class Medias(models.Model):
    medio_baixo = models.FloatField('Média de baixo', null=True)
    medio_alto = models.FloatField('Média de alto', null=True)
    
    class Meta:
        abstract = True
        
class Fatores(Medias):
    fator = models.CharField('Fator', max_length=20, null=True)
        
class Dimensoes(Medias):
    dimensao = models.CharField('Dimensão', max_length=20, null=True)

class DimensoesEST(models.Model):
    RISCOS = (
        ('1', 'Risco Baixo'),
        ('2', 'Risco Médio'),
        ('3', 'Risco Alto')
    )
    
    id_aluno = models.IntegerField('Id do aluno', null=True)
    E_ESCV = models.FloatField('E-ESCV', null=True)
    E_PROFV = models.FloatField('E-PROFV', null=True)
    E_FAMV = models.FloatField('E-FAMV', null=True)
    E_COMV = models.FloatField('E-COMV', null=True)
    E_ESCC = models.CharField('E-ESCC', max_length=1, choices=RISCOS, null=True)
    E_PROFC = models.CharField('E-PROFC', max_length=1, choices=RISCOS, null=True)
    E_FAMC = models.CharField('E-FAMC', max_length=1, choices=RISCOS, null=True)
    E_COMC = models.CharField('E-COMC', max_length=1, choices=RISCOS, null=True)
    E_ESTC = models.CharField('E-ESTC', max_length=1, choices=RISCOS, null=True)
    
    class Meta:
        ordering = ['id']
        db_table = 'dimensoes_est'

class FatoresEST(models.Model):
    RISCOS = (
        ('1', 'Risco Baixo'),
        ('2', 'Risco Médio'),
        ('3', 'Risco Alto')
    )
    
    id_aluno = models.IntegerField('Id do aluno', null=True)
    E_ESC1V = models.FloatField('E-ESC1V', null=True)
    E_ESC2V = models.FloatField('E-ESC2V', null=True)
    E_PROF21V = models.FloatField('E-PROF1V', null=True)
    E_PROF2V = models.FloatField('E-PROF2V', null=True)
    E_FAM1V = models.FloatField('E-FAM1V', null=True)
    E_FAM2V = models.FloatField('E-FAM2V', null=True)
    E_COM1V = models.FloatField('E-COM1V', null=True)
    E_COM2V = models.FloatField('E-COM2V', null=True)
    E_COM3V = models.FloatField('E-COM3V', null=True)
    E_EST1V = models.FloatField('E-EST1V', null=True)
    E_EST2V = models.FloatField('E-EST2V', null=True)
    E_EST3V = models.FloatField('E-EST3V', null=True)
    E_ESC1C = models.CharField('E-ESC1C', max_length=1, choices=RISCOS, null=True)
    E_ESC2C = models.CharField('E-ESC2C', max_length=1, choices=RISCOS, null=True)
    E_PROF1C = models.CharField('E-PROF1C', max_length=1, choices=RISCOS, null=True)
    E_PROF2C = models.CharField('E-PROF2C', max_length=1, choices=RISCOS, null=True)
    E_FAM1C = models.CharField('E-FAM1C', max_length=1, choices=RISCOS, null=True)
    E_FAM2C = models.CharField('E-FAM2C', max_length=1, choices=RISCOS, null=True)
    E_COM1C = models.CharField('E-COM1C', max_length=1, choices=RISCOS, null=True)
    E_COM2C = models.CharField('E-COM2C', max_length=1, choices=RISCOS, null=True)
    E_COM3C = models.CharField('E-COM3C', max_length=1, choices=RISCOS, null=True)
    E_EST1C = models.CharField('E-ESTC', max_length=1, choices=RISCOS, null=True)
    E_EST2C = models.CharField('E-ESTC', max_length=2, choices=RISCOS, null=True)
    E_EST3C = models.CharField('E-ESTC', max_length=3, choices=RISCOS, null=True)
    
    class Meta:
        ordering = ['id']
        db_table = 'fatores_est'