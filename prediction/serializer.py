from rest_framework import serializers
from prediction.models import DimensaoEST, FatorEST, Dimensao, Fator
from prediction.processamento import processa_from_dict


class DimensaoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dimensao
        fields = '__all__'

class FatorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Fator
        fields = '__all__'

class DimensaoESTSerializer(serializers.ModelSerializer):
    # E_ESCC = serializers.CharField(source='get_E_ESCC_display')
    class Meta:
        model = DimensaoEST
        fields = '__all__'
        
class FatorESTSerializer(serializers.ModelSerializer):
    class Meta:
        model = FatorEST
        fields = '__all__'

class ProcessamentoSerializer(serializers.Serializer):
    aluno = serializers.JSONField(required=False)
    alunos = serializers.JSONField(required=False)

    def validate(self, attrs):
        if 'aluno' not in attrs and 'alunos' not in attrs:
            raise serializers.ValidationError("Deve ser passado um aluno ou uma lista de alunos para realizar o processamento")
        if 'aluno' in attrs and 'alunos' in attrs:
            raise serializers.ValidationError("Deve ser passado um aluno ou uma lista de alunos para realizar o processamento, não ambos")
        return super().validate(attrs)

    def processar(self):
        if 'alunos' not in self.data:
            alunos = []
            alunos.append(self.data['aluno'])
        else:
            alunos = self.data['alunos']

        return processa_from_dict(alunos)
