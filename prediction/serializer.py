from rest_framework import serializers
from prediction.models import DimensaoEST, FatorEST, Dimensao, Fator
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
