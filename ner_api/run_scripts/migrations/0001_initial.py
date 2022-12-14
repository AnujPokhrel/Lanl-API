# Generated by Django 4.1 on 2022-08-12 22:27

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='NerModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('model_name', models.CharField(max_length=200, verbose_name='Name of the Model')),
                ('training_file', models.FileField(upload_to='training_files/')),
                ('testing_file', models.FileField(upload_to='testing_files/')),
                ('model_holder', models.FileField(upload_to='model_files/')),
                ('model_holder_made_at', models.IntegerField()),
                ('learning_rate', models.FloatField()),
                ('prob_threshold', models.FloatField()),
                ('max_epochs', models.IntegerField()),
                ('epochs_pr_train_cycle', models.IntegerField()),
                ('time_finished', models.DateTimeField()),
                ('base_model_holder', models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, to='run_scripts.nermodel')),
            ],
        ),
    ]
