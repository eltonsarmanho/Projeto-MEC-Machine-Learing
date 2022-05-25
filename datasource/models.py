from operator import mod
from django.db import models

# Create your models here.
class Escola(models.Model):
    nome = models.CharField('Nome', max_length=150, null=False)
    estado = models.CharField('Estado', max_length=80, null=False)
    cidade = models.CharField('Cidade', max_length=100, null=False)
    endereco = models.CharField('Endereço', max_length=300, null=False)

class Aluno(models.Model):
    escola = models.ForeignKey(Escola, on_delete=models.CASCADE, null=False)
    matricula = models.IntegerField('Código de Matrícula', null=True)
    nome_turma = models.CharField('Nome da turma', max_length=80, null=True)
    modo_ensino = models.CharField('Nome modo de Ensino', max_length=50, null=True)
    etapa_ensino = models.CharField('Nome etapa de Ensino', max_length=80, null=True)
    nome = models.CharField('Nome', max_length=100, null=True)
    rg = models.IntegerField('RG', null=True)
    sexo = models.CharField('Sexo', max_length=50, null=True)
    data_nascimento = models.DateField('Data de Nascimento', null=True)
    nome_mae = models.CharField('Nome da Mãe', max_length=100, null=True)

class Representante(models.Model):
    escola = models.ManyToManyField(Escola, on_delete=models.CASCADE, null=False)
    nome = models.CharField('Nome do Representante', max_length=150, null=False)
    cpf = models.CharField('CPF', max_length=15, null=False)

class Turma(models.Model):
    escola = models.ForeignKey(Escola, on_delete=models.CASCADE, null=False)
    nome = models.CharField('Nome da Turma', max_length=50, null=False)
