from django.shortcuts import render
from rest_framework.response import response
from rest_framework.decorators import api_view
from rest_framework import status
from django.http import JsonResponse

@api_view['GET']
def front_page(request):
    if request.method == 'GET':
        data = {'ABC': 1234}
        return JsonResponse({data})


# Create your views here.
