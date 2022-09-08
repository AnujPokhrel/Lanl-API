import graphene
import run_scripts.schema

class Query(
    run_scripts.schema.Query
):
    pass

class Mutation(
    run_scripts.schema.Mutation
):
    pass

schema = graphene.Schema(query=Query, mutation=Mutation)