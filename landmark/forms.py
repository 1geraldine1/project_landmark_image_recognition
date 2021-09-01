from django.forms import ModelForm
from .models import imageModel

class ImageModelForm(ModelForm):
    class Meta:
        model = imageModel
        fields = ['imgfile']