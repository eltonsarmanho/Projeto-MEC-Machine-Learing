from rest_framework import serializers
from prediction.models import Predicao

class PredicaoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Predicao
        fields = '__all__'
        