# Generated by Django 3.2.6 on 2021-09-01 05:16

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('landmark', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='fileupload',
            name='content',
        ),
        migrations.RemoveField(
            model_name='fileupload',
            name='title',
        ),
    ]
