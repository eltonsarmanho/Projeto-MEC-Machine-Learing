from rest_framework import serializers
from prediction.models import Predicao, DimensoesEST, FatoresEST, Dimensoes, Fatores
class PredicaoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Predicao
        fields = '__all__'

class DimensoesSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dimensoes
        fields = '__all__'

class FatoresSerializer(serializers.ModelSerializer):
    class Meta:
        model = Fatores
        fields = '__all__'

class DimensoesESTSerializer(serializers.ModelSerializer):
    # E_ESCC = serializers.CharField(source='get_E_ESCC_display')
    class Meta:
        model = DimensoesEST
        fields = '__all__'
        
class FatoresESTSerializer(serializers.ModelSerializer):
    class Meta:
        model = FatoresEST
        fields = '__all__'
