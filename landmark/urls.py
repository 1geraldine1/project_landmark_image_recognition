from django.urls import path
from .views import *

urlpatterns = [
	path('',index,name="index"),
	path('result',result,name="result"),
	path('recommand',recommand,name="recommand")

]