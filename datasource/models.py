from operator import mod
from django.db import models

# Create your models here.
class Escola(models.Model):
    nome = models.CharField('Nome', max_length=150, null=False)
    estado = models.CharField('Estado', max_length=80, null=False)
    cidade = models.CharField('Cidade', max_length=100, null=False)
    endereco = models.CharField('Endereço', max_length=300, null=False)
