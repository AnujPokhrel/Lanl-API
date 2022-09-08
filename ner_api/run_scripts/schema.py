from importlib.metadata import requires
import graphene
from graphene_django import DjangoObjectType
from graphene_file_upload.scalars import Upload
from .get_tags import *


from .models import NerModel

class NerModelType(DjangoObjectType):
    class Meta:
        model = NerModel

class NerModelMutation(graphene.Mutation):
    class Arguments:
        model_name =  graphene.String(required=True)
        learning_rate = graphene.Float(required=True)
        prob_threshold = graphene.Float(required=True)
        max_epochs = graphene.Int(required=True)
        epochs_pr_train_cycle = graphene.Int(required=True)
        training_file = Upload(required=True)
        testing_file = Upload(required=True)
    
    id = graphene.ID()
    text = graphene.String()
    
    def mutate(self, info, training_file, testing_file, **kwargs):
        return NerModelMutation(id=1, text=str(training_file))

class GetTagsMutation(graphene.Mutation):
    class Arguments:
        text = graphene.String(required=True)
        pr_threshold = graphene.Float(required=True)
        training_file_name = graphene.String(required=True)
    
    # info = graphene.String
    text = graphene.String()
    pr_threshold = graphene.Float()
    training_file_name = graphene.String()
    tags = graphene.List(graphene.String)

    def mutate(self, info, text, pr_threshold, training_file_name, **kwargs):
        # cud_available = "cuda" if torch.cuda.is_available() else "cpu"
        ner_tags = start_tagging(training_file_name, text, pr_threshold)
        # tags = [info, text, str(pr_threshold), training_file_name, ner_tags]
        return GetTagsMutation(text=text, pr_threshold=pr_threshold, training_file_name=training_file_name, tags=ner_tags)

    

class Query(graphene.ObjectType):
    all_ner_models = graphene.List(NerModelType)

    def resolve_all_ner_models(root, info):
        return NerModel.objects.all()

class Mutation(graphene.ObjectType):
    ner_model_make = NerModelMutation.Field()
    get_tags = GetTagsMutation.Field()