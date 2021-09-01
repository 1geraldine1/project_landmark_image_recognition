from django.db import models

# Create your models here.
class imageModel(models.Model):
    imgfile = models.ImageField(null=True, upload_to="", blank=True)

