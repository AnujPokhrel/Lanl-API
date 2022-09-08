from django.db import models
from django.contrib.postgres.fields import ArrayField


class NerModel(models.Model):
    model_name = models.CharField('Name of the Model', max_length=200)
    training_file = models.FileField(upload_to ='training_files/')
    testing_file = models.FileField(upload_to='testing_files/')
    model_holder = models.FileField(upload_to='model_files/')
    model_holder_made_at = models.IntegerField()
    base_model_holder = models.ForeignKey('self', on_delete=models.DO_NOTHING)
    learning_rate = models.FloatField()
    prob_threshold = models.FloatField()
    max_epochs = models.IntegerField()
    epochs_pr_train_cycle = models.IntegerField()
    time_finished = models.DateTimeField()
    # training_done = models.BooleanField(default=False)